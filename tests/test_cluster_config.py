# SPDX-License-Identifier: Apache-2.0
"""The operator surface: reachability, the config write, and the hot-apply.

The property that matters most here is the one that is easiest to lose: the
admin API has to exist on a machine where clustering has never been enabled,
because it is what enables it. Every test that matters therefore starts from
`enabled=False` - the shipped default - rather than from a configured cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin.auth import require_admin
from omlx.cluster import bootstrap, routes


# =============================================================================
# Fakes
# =============================================================================


@dataclass
class FakeClusterSettings:
    enabled: bool = False
    cluster_key: str = ""
    backend: str = "auto"
    model: str = ""
    pipeline: bool = False
    discovery_interval_seconds: float = 5.0
    max_batch_size: int = 8


@dataclass
class FakeServerSettings:
    port: int = 8888


@dataclass
class FakeAuthSettings:
    skip_api_key_verification: bool = False


@dataclass
class FakeSettings:
    cluster: FakeClusterSettings = field(default_factory=FakeClusterSettings)
    server: FakeServerSettings = field(default_factory=FakeServerSettings)
    auth: FakeAuthSettings = field(default_factory=FakeAuthSettings)
    saves: int = 0

    def save(self) -> None:
        self.saves += 1


class FakeDiscovery:
    """Stands in for Bonjour. Real discovery shells out to `dns-sd`."""

    instances: list[FakeDiscovery] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.peers: list[object] = []
        FakeDiscovery.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakePeerInfo:
    def __init__(self, node_id: str, fingerprint: str, hostname: str = "") -> None:
        self.node_id = node_id
        self.hostname = hostname
        self.port = 8888
        self.chip = "Apple M3 Ultra"
        self.ram_gb = 96
        self.version = "0.5.3"
        self.key_fingerprint = fingerprint


class FakePeer:
    def __init__(self, node_id: str, fingerprint: str, hostname: str = "") -> None:
        self.info = FakePeerInfo(node_id, fingerprint, hostname)
        self.host = "192.168.5.28"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_bootstrap(monkeypatch):
    """Reset the module-level cluster state around every test."""
    from omlx.cluster import discovery as discovery_module

    FakeDiscovery.instances = []
    monkeypatch.setattr(discovery_module, "ClusterDiscovery", FakeDiscovery)
    # Skip the sysctl calls behind the local node description.
    monkeypatch.setattr(
        routes,
        "_local_node",
        {
            "node_id": "test-node",
            "chip": "Apple M5 Max",
            "ram_gb": 128,
            "version": "0.5.3",
        },
    )
    for name, value in (
        ("_discovery", None),
        ("_manager", None),
        ("_settings", None),
        ("_admin_installed", False),
        ("_peer_installed", False),
    ):
        monkeypatch.setattr(bootstrap, name, value)
    monkeypatch.setattr(routes, "_follower", None)
    yield
    routes.reset_resolved()


@pytest.fixture
def node(clean_bootstrap):
    """A daemon with clustering off - the configuration everyone ships with."""

    async def _allow_admin():
        return True

    settings = FakeSettings()
    app = FastAPI()
    app.dependency_overrides[require_admin] = _allow_admin
    advertising = bootstrap.install(app, settings)
    assert advertising is False
    return TestClient(app), settings, app


# =============================================================================
# Reachability
# =============================================================================


def test_admin_surface_exists_when_clustering_is_off(node):
    """The default install can still be asked about itself."""
    client, _settings, _app = node

    response = client.get("/admin/api/cluster/status")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["configured"] is False
    assert body["local"]["node_id"] == "test-node"


def test_peer_surface_is_absent_when_clustering_is_off(node):
    """A node that has not opted in serves no path that can spawn a rank."""
    client, _settings, _app = node

    assert client.post("/cluster/report", json={"model": ""}).status_code == 404
    assert client.post("/cluster/ranks/stop").status_code == 404


def test_config_and_preflight_answer_when_clustering_is_off(node):
    client, _settings, _app = node

    config = client.get("/admin/api/cluster/config")

    assert config.status_code == 200
    assert config.json()["enabled"] is False
    assert "jaccl" in config.json()["backends"]


def test_enabling_registers_the_peer_surface_without_a_restart(node):
    """The regression this whole split exists to prevent.

    A single install flag would have been set by the admin router at startup,
    so the peer router would never be added and the leader's `/cluster/report`
    would 404 - a failure that reads like a version mismatch, not like
    clustering being off.
    """
    client, settings, _app = node

    response = client.post(
        "/admin/api/cluster/config",
        json={"enabled": True, "cluster_key": "k" * 20},
    )

    assert response.status_code == 200
    assert response.json()["advertising"] is True
    assert settings.cluster.enabled is True
    assert settings.saves == 1

    # Present, and refusing on the key rather than on the path.
    report = client.post("/cluster/report", json={"model": ""})
    assert report.status_code == 403


def test_disabling_leaves_the_peer_surface_fail_closed(node):
    """FastAPI cannot remove a route, so the check has to be at call time."""
    client, _settings, _app = node
    client.post(
        "/admin/api/cluster/config",
        json={"enabled": True, "cluster_key": "k" * 20},
    )

    client.post("/admin/api/cluster/config", json={"enabled": False})

    response = client.post(
        "/cluster/report", json={"model": ""}, headers={"X-Cluster-Key": "k" * 20}
    )
    assert response.status_code == 403
    assert "not enabled" in response.json()["detail"]


# =============================================================================
# The config write
# =============================================================================


def test_enabling_without_a_key_is_refused(node):
    """Otherwise discovery starts and the peer list never fills in."""
    client, settings, _app = node

    response = client.post("/admin/api/cluster/config", json={"enabled": True})

    assert response.status_code == 400
    assert settings.cluster.enabled is False


@pytest.mark.parametrize(
    "key,reason",
    [
        ("short", "at least"),
        ("has whitespace in it here", "whitespace"),
        ("café-" + "x" * 20, "ASCII"),
    ],
)
def test_weak_or_unusable_keys_are_refused(node, key, reason):
    """A key is brute-forceable from the digest we broadcast, and it travels
    as an HTTP header the ASGI layer decodes as latin-1."""
    client, settings, _app = node

    response = client.post("/admin/api/cluster/config", json={"cluster_key": key})

    assert response.status_code == 400
    assert reason in response.json()["detail"]
    assert settings.cluster.cluster_key == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"backend": "infiniband"},
        {"max_batch_size": 0},
        {"max_batch_size": 999},
        {"discovery_interval_seconds": 0.1},
        {"discovery_interval_seconds": 10000},
    ],
)
def test_out_of_range_values_are_refused(node, payload):
    client, settings, _app = node

    response = client.post("/admin/api/cluster/config", json=payload)

    assert response.status_code == 400
    assert settings.saves == 0


def test_a_partial_write_leaves_other_fields_alone(node):
    client, settings, _app = node
    settings.cluster.model = "big-model"
    settings.cluster.backend = "jaccl"

    response = client.post("/admin/api/cluster/config", json={"max_batch_size": 4})

    assert response.status_code == 200
    assert settings.cluster.model == "big-model"
    assert settings.cluster.backend == "jaccl"
    assert settings.cluster.max_batch_size == 4


def test_a_serving_cluster_refuses_the_write(node, monkeypatch):
    """Applying means tearing ranks down. Not while one holds a request."""
    client, _settings, _app = node

    class _Status:
        formed = True
        busy = True

    class _Manager:
        def status(self):
            return _Status()

    monkeypatch.setattr(bootstrap, "_manager", _Manager())

    response = client.post("/admin/api/cluster/config", json={"max_batch_size": 2})

    assert response.status_code == 409


def test_the_write_is_refused_when_admin_auth_is_switched_off(node):
    """`skip_api_key_verification` makes every caller an admin. Letting one
    write a cluster key would turn that into remote process spawn."""
    client, settings, _app = node
    settings.auth.skip_api_key_verification = True

    write = client.post("/admin/api/cluster/config", json={"max_batch_size": 2})
    keygen = client.post("/admin/api/cluster/key")

    assert write.status_code == 403
    assert keygen.status_code == 403
    # Reading stays available; it discloses nothing this mode has not already.
    assert client.get("/admin/api/cluster/status").status_code == 200


def test_generated_keys_are_long_and_unique(node):
    client, _settings, _app = node

    first = client.post("/admin/api/cluster/key").json()["cluster_key"]
    second = client.post("/admin/api/cluster/key").json()["cluster_key"]

    assert first != second
    assert len(first) >= routes.KEY_MIN_LENGTH
    assert client.post(
        "/admin/api/cluster/config", json={"cluster_key": first}
    ).status_code == 200


# =============================================================================
# Disclosure
# =============================================================================


def test_the_status_poll_never_carries_the_key(node):
    """It is fetched every few seconds by the browser. A secret does not
    belong in a heartbeat, even one behind admin auth."""
    client, _settings, _app = node
    secret = "s3cret-" + "z" * 20
    client.post(
        "/admin/api/cluster/config", json={"enabled": True, "cluster_key": secret}
    )

    body = client.get("/admin/api/cluster/status").text

    assert secret not in body
    assert client.get("/admin/api/cluster/config").json()["cluster_key"] == secret


# =============================================================================
# Pairing
# =============================================================================


def test_a_peer_with_a_different_key_is_shown_rather_than_hidden(node, monkeypatch):
    """An empty peer list and a mistyped key look identical otherwise."""
    from omlx.cluster.discovery import fingerprint

    client, _settings, _app = node
    ours = "matching-key-" + "m" * 10
    client.post(
        "/admin/api/cluster/config", json={"enabled": True, "cluster_key": ours}
    )
    monkeypatch.setattr(
        bootstrap,
        "peers",
        lambda: [
            FakePeer("agrees", fingerprint(ours)),
            FakePeer("disagrees", fingerprint("some-other-key-entirely")),
        ],
    )

    peers = {p["node_id"]: p for p in client.get("/admin/api/cluster/status").json()["peers"]}

    assert peers["agrees"]["key_match"] is True
    assert peers["disagrees"]["key_match"] is False


def test_key_match_is_unknown_rather_than_false_without_a_key(node, monkeypatch):
    """Nothing to compare against is not the same as a mismatch."""
    client, _settings, _app = node
    monkeypatch.setattr(bootstrap, "peers", lambda: [FakePeer("someone", "abc123")])

    peers = client.get("/admin/api/cluster/status").json()["peers"]

    assert peers[0]["key_match"] is None


def test_the_peer_check_does_not_call_a_mismatched_peer(node, monkeypatch):
    """It would hand our key to a machine that is not in this cluster, and
    come back with a raw 403 the page could already have predicted."""
    from omlx.cluster.discovery import fingerprint

    client, _settings, _app = node
    ours = "our-shared-key-" + "o" * 10
    client.post("/admin/api/cluster/config", json={"enabled": True, "cluster_key": ours})
    monkeypatch.setattr(
        bootstrap,
        "peers",
        lambda: [FakePeer("stranger", fingerprint("a-different-key"), "Someone")],
    )

    result = client.post(
        "/admin/api/cluster/peers/check", json={"model": "big-model"}
    ).json()

    assert result["peers"][0]["ok"] is False
    assert result["peers"][0]["key_match"] is False
    assert "different cluster key" in result["peers"][0]["error"]


def test_a_stranger_on_the_lan_cannot_break_formation(clean_bootstrap):
    """A peer with a different key is skipped, not called.

    Calling it sends our key, takes a 403 from `/cluster/report`, and raises
    out of `_collect_reports` - so one unrelated oMLX install on the network
    would stop this cluster forming at all.
    """
    from omlx.cluster.discovery import fingerprint
    from omlx.cluster.manager import ClusterManager

    ours = "our-shared-key-" + "o" * 10
    settings = FakeSettings()
    settings.cluster.cluster_key = ours
    peers = [
        FakePeer("ours", fingerprint(ours), "Studio"),
        FakePeer("stranger", fingerprint("someone-elses-key"), "Someone-Else"),
    ]
    manager = ClusterManager(settings, lambda: peers)

    clients = manager._peer_clients(ours)

    assert set(clients) == {"ours"}


# =============================================================================
# Hot-apply
# =============================================================================


def test_applying_rebuilds_discovery_rather_than_mutating_it(node):
    """Discovery reads its port and interval once, at construction."""
    client, _settings, _app = node
    client.post(
        "/admin/api/cluster/config",
        json={"enabled": True, "cluster_key": "k" * 20},
    )
    first = FakeDiscovery.instances[-1]

    client.post("/admin/api/cluster/config", json={"discovery_interval_seconds": 30})
    second = FakeDiscovery.instances[-1]

    assert first is not second
    assert first.stopped is True
    assert second.kwargs["poll_interval"] == 30


def test_applying_forgets_paths_resolved_under_the_old_settings(node):
    """A path resolved under one set of model directories is not a path under
    the next set; spawning a worker on it would fail late and unreadably."""
    client, _settings, _app = node
    routes._resolved["big-model"] = "/Volumes/Models/big-model"

    client.post("/admin/api/cluster/config", json={"model": "big-model"})

    assert routes._resolved == {}


def test_applying_evicts_the_engine_bound_to_the_discarded_manager(node, monkeypatch):
    """`ClusterEngine` holds its manager by reference. Left in the pool, it
    would keep answering through a cluster that has been torn down - and while
    clustering is off the pool will not rebuild it either."""
    client, settings, _app = node
    settings.cluster.model = "big-model"
    evicted: list[str] = []

    async def _evict(model_id: str) -> None:
        evicted.append(model_id)

    monkeypatch.setattr(bootstrap, "_evict_cluster_engine", _evict)

    client.post("/admin/api/cluster/config", json={"model": "other-model"})

    assert evicted == ["big-model"]
