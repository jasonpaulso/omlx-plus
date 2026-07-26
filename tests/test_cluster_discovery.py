# SPDX-License-Identifier: Apache-2.0
"""Tests for `omlx.cluster.discovery`.

None of these touch the real network: `DnsSdBackend`'s subprocess calls are
mocked, and the peer-table tests use a `FakeBackend` test double instead of
shelling out to `dns-sd` at all. The `_parse_*` tests are checked against
output captured from a real `dns-sd -R` / `-B` / `-L` round trip on this
machine, not hand-written guesses at the format.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from omlx.cluster import discovery
from omlx.cluster.discovery import (
    ClusterDiscovery,
    DiscoveryUnavailableError,
    DnsSdBackend,
    Peer,
    PeerInfo,
    _parse_browse_line,
    _parse_resolve_output,
    _parse_txt_tokens,
    decode_txt,
    encode_txt,
    fingerprint,
    matches_fingerprint,
)

# =============================================================================
# Fixtures / helpers
# =============================================================================


class FakeBackend:
    """A `DiscoveryBackend` test double with no subprocess, no thread."""

    def __init__(self, is_available: bool = True) -> None:
        self.is_available = is_available
        self.advertised: tuple[str, int, dict[str, str]] | None = None
        self.stopped_advertising = False
        self.stopped = False
        self.scan_results: list[dict[str, tuple[str, dict[str, str]]]] = []

    def available(self) -> bool:
        return self.is_available

    def advertise(self, node_id: str, port: int, txt: dict[str, str]) -> None:
        self.advertised = (node_id, port, txt)

    def stop_advertising(self) -> None:
        self.stopped_advertising = True

    def scan(self) -> dict[str, tuple[str, dict[str, str]]]:
        if not self.scan_results:
            return {}
        return self.scan_results.pop(0)

    def stop(self) -> None:
        self.stopped = True


def _txt_for(node_id: str, port: int = 8888, key: str = "shared-secret") -> dict[str, str]:
    info = PeerInfo(
        node_id=node_id,
        version="0.5.3",
        port=port,
        chip="Apple M5 Max",
        ram_gb=128,
        key_fingerprint=fingerprint(key),
    )
    return encode_txt(info)


# =============================================================================
# TXT record encode/decode round trip
# =============================================================================


def test_encode_decode_round_trip():
    info = PeerInfo(
        node_id="DB6FC0C8-FCE2-5498-B721-D140F6082D9B",
        version="0.5.3",
        port=8888,
        chip="Apple M5 Max",
        ram_gb=128,
        key_fingerprint=fingerprint("shared-secret"),
    )
    assert decode_txt(encode_txt(info)) == info


def test_decode_txt_rejects_missing_required_field():
    txt = _txt_for("node-a")
    del txt["node_id"]
    assert decode_txt(txt) is None


def test_decode_txt_rejects_non_integer_port():
    txt = _txt_for("node-a")
    txt["port"] = "not-a-port"
    assert decode_txt(txt) is None


def test_decode_txt_defaults_optional_fields():
    # A minimal record from a hypothetical older/newer peer still decodes.
    txt = {"node_id": "node-a", "port": "8888"}
    info = decode_txt(txt)
    assert info is not None
    assert info.chip == "unknown"
    assert info.ram_gb == 0
    assert info.key_fingerprint == ""


# =============================================================================
# Cluster-key fingerprint match/mismatch
# =============================================================================


def test_fingerprint_is_deterministic_and_short():
    assert fingerprint("shared-secret") == fingerprint("shared-secret")
    assert len(fingerprint("shared-secret")) == 16


def test_fingerprint_does_not_leak_the_key():
    assert "shared-secret" not in fingerprint("shared-secret")


def test_matches_fingerprint_true_for_same_key():
    assert matches_fingerprint("shared-secret", fingerprint("shared-secret"))


def test_matches_fingerprint_false_for_different_key():
    assert not matches_fingerprint("shared-secret", fingerprint("a-different-key"))


# =============================================================================
# Parsing, checked against real captured `dns-sd` output
# =============================================================================


def test_parse_browse_line_add():
    line = "15:07:12.727  Add        3  31 local.               _omlx._tcp.          omlx-discovery-test-19865"
    assert _parse_browse_line(line) == ("Add", "omlx-discovery-test-19865")


def test_parse_browse_line_rmv():
    line = "15:07:20.001  Rmv        2  31 local.               _omlx._tcp.          omlx-discovery-test-19865"
    assert _parse_browse_line(line) == ("Rmv", "omlx-discovery-test-19865")


def test_parse_browse_line_preserves_spaces_in_instance_name():
    line = "15:07:12.727  Add        3  31 local.               _omlx._tcp.          Jason's Mac Studio"
    assert _parse_browse_line(line) == ("Add", "Jason's Mac Studio")


@pytest.mark.parametrize(
    "line",
    [
        "Browsing for _omlx._tcp.local",
        "DATE: ---Sun 26 Jul 2026---",
        "15:07:12.726  ...STARTING...",
        "Timestamp     A/R    Flags  if Domain               Service Type         Instance Name",
        "",
    ],
)
def test_parse_browse_line_skips_banners_and_headers(line):
    assert _parse_browse_line(line) is None


def test_parse_browse_line_ignores_other_service_types():
    line = "15:07:12.727  Add        3  31 local.               _http._tcp.          some-other-service"
    assert _parse_browse_line(line) is None


def test_parse_txt_tokens_plain():
    assert _parse_txt_tokens("node_id=testnode123 version=9.9.9 ram_gb=192") == {
        "node_id": "testnode123",
        "version": "9.9.9",
        "ram_gb": "192",
    }


def test_parse_txt_tokens_unescapes_backslash_space():
    # Captured verbatim from `dns-sd -L` resolving a TXT value with a real space.
    assert _parse_txt_tokens(r"node_id=abc chip=Apple\ M3\ Ultra ram_gb=192") == {
        "node_id": "abc",
        "chip": "Apple M3 Ultra",
        "ram_gb": "192",
    }


_RESOLVE_SAMPLE = (
    "Lookup omlx-space-test._omlx._tcp.local\n"
    "DATE: ---Sun 26 Jul 2026---\n"
    "15:07:42.934  ...STARTING...\n"
    "15:07:42.935  omlx-space-test._omlx._tcp.local. can be reached at "
    "Jasons-MacBook-Pro.local.:18889 (interface 30) Flags: 1\n"
    r" node_id=abc chip=Apple\ M3\ Ultra ram_gb=192" "\n"
    "15:07:42.935  omlx-space-test._omlx._tcp.local. can be reached at "
    "Jasons-MacBook-Pro.local.:18889 (interface 30) Flags: 1\n"
    r" node_id=abc chip=Apple\ M3\ Ultra ram_gb=192" "\n"
)


def test_parse_resolve_output_real_sample():
    result = _parse_resolve_output(_RESOLVE_SAMPLE)
    assert result == (
        "Jasons-MacBook-Pro.local",
        {"node_id": "abc", "chip": "Apple M3 Ultra", "ram_gb": "192"},
    )


def test_parse_resolve_output_empty():
    assert _parse_resolve_output("") is None
    assert _parse_resolve_output("Lookup foo\nDATE: ---today---\n") is None


# =============================================================================
# DnsSdBackend
# =============================================================================


def test_dns_sd_backend_available_checks_path(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda name: "/usr/bin/dns-sd")
    assert DnsSdBackend().available()
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    assert not DnsSdBackend().available()


def test_dns_sd_backend_advertise_builds_expected_argv():
    backend = DnsSdBackend()
    with patch.object(discovery.subprocess, "Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        backend.advertise("node-a", 8888, {"node_id": "node-a", "port": "8888"})

    argv = mock_popen.call_args.args[0]
    assert argv[:5] == ["dns-sd", "-R", "node-a", "_omlx._tcp", "local"]
    assert "8888" in argv
    assert "node_id=node-a" in argv


def test_dns_sd_backend_advertise_is_a_noop_when_already_advertising():
    backend = DnsSdBackend()
    with patch.object(discovery.subprocess, "Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        backend.advertise("node-a", 8888, {})
        backend.advertise("node-a", 8888, {})
    assert mock_popen.call_count == 1


def test_dns_sd_backend_stop_advertising_terminates_process():
    backend = DnsSdBackend()
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    with patch.object(discovery.subprocess, "Popen", return_value=fake_proc):
        backend.advertise("node-a", 8888, {})
    backend.stop_advertising()
    fake_proc.terminate.assert_called_once()
    # Idempotent: a second call must not blow up or re-terminate.
    backend.stop_advertising()
    fake_proc.terminate.assert_called_once()


def test_dns_sd_backend_stop_advertising_escalates_to_kill_on_timeout():
    backend = DnsSdBackend()
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="dns-sd", timeout=5)
    with patch.object(discovery.subprocess, "Popen", return_value=fake_proc):
        backend.advertise("node-a", 8888, {})
    backend.stop_advertising()
    fake_proc.kill.assert_called_once()


def test_dns_sd_backend_advertise_handles_missing_binary_gracefully():
    backend = DnsSdBackend()
    with patch.object(discovery.subprocess, "Popen", side_effect=OSError("no such file")):
        backend.advertise("node-a", 8888, {})  # must not raise
    backend.stop_advertising()  # must not raise either


# =============================================================================
# Constructing without start() spawns nothing
# =============================================================================


def test_construction_spawns_no_thread_or_subprocess():
    with patch.object(discovery.threading, "Thread") as mock_thread, patch.object(
        discovery.subprocess, "Popen"
    ) as mock_popen:
        ClusterDiscovery(
            node_id="node-a",
            port=8888,
            version="0.5.3",
            cluster_key="shared-secret",
        )
    mock_thread.assert_not_called()
    mock_popen.assert_not_called()


def test_construction_with_fake_backend_does_not_call_start_methods():
    backend = FakeBackend()
    ClusterDiscovery(
        node_id="node-a", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    assert backend.advertised is None
    assert not backend.stopped
    assert not backend.stopped_advertising


# =============================================================================
# start() / stop() lifecycle
# =============================================================================


def test_start_raises_when_backend_unavailable():
    backend = FakeBackend(is_available=False)
    cd = ClusterDiscovery(
        node_id="node-a", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    with pytest.raises(DiscoveryUnavailableError):
        cd.start()
    assert backend.advertised is None


def test_start_advertises_this_nodes_own_txt_record():
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="node-a",
        port=8888,
        version="0.5.3",
        cluster_key="shared-secret",
        chip="Apple M5 Max",
        ram_gb=128,
        backend=backend,
    )
    cd.start()
    try:
        assert backend.advertised is not None
        node_id, port, txt = backend.advertised
        assert node_id == "node-a"
        assert port == 8888
        assert txt["key_fingerprint"] == fingerprint("shared-secret")
        assert txt["chip"] == "Apple M5 Max"
    finally:
        cd.stop()


def test_stop_is_idempotent_and_non_hanging():
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="node-a", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    cd.start()
    cd.stop()
    cd.stop()  # must not raise
    assert backend.stopped_advertising
    assert backend.stopped
    assert not cd.started


def test_stop_without_start_is_a_noop():
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="node-a", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    cd.stop()  # must not raise
    assert not backend.stopped_advertising


def test_stop_joins_the_browse_thread(monkeypatch):
    # A very short poll interval so the background thread loops quickly, then
    # confirm stop() actually leaves no thread running behind it.
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="node-a",
        port=8888,
        version="0.5.3",
        cluster_key="k",
        backend=backend,
        poll_interval=0.01,
    )
    cd.start()
    thread = cd._thread
    cd.stop()
    assert thread is not None
    assert not thread.is_alive()


# =============================================================================
# Peer add / expire on last-seen timeout (direct reconcile, deterministic clock)
# =============================================================================


def test_peer_joins_and_is_reported():
    backend = FakeBackend()
    events: list[tuple[str, Peer]] = []
    cd = ClusterDiscovery(
        node_id="me",
        port=8888,
        version="0.5.3",
        cluster_key="shared-secret",
        backend=backend,
        on_event=lambda event, peer: events.append((event, peer)),
    )
    backend.scan_results.append({"peer-a": ("peer-a.local", _txt_for("peer-a"))})

    cd._scan_and_reconcile()

    assert [p.info.node_id for p in cd.peers] == ["peer-a"]
    assert events == [("joined", cd.peers[0])]


def test_peer_does_not_discover_itself():
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="me", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    backend.scan_results.append({"me": ("me.local", _txt_for("me"))})

    cd._scan_and_reconcile()

    assert cd.peers == []


def test_peer_last_seen_refreshes_on_repeated_scan(monkeypatch):
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="me", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    clock = [100.0]
    monkeypatch.setattr(discovery, "_now", lambda: clock[0])

    backend.scan_results.append({"peer-a": ("peer-a.local", _txt_for("peer-a"))})
    cd._scan_and_reconcile()
    first_seen = cd.peers[0].last_seen

    clock[0] = 110.0
    backend.scan_results.append({"peer-a": ("peer-a.local", _txt_for("peer-a"))})
    cd._scan_and_reconcile()

    assert cd.peers[0].last_seen == 110.0
    assert cd.peers[0].last_seen != first_seen


def test_peer_expires_after_timeout_without_a_goodbye(monkeypatch):
    backend = FakeBackend()
    events: list[str] = []
    cd = ClusterDiscovery(
        node_id="me",
        port=8888,
        version="0.5.3",
        cluster_key="k",
        backend=backend,
        expire_after=30.0,
        on_event=lambda event, peer: events.append(event),
    )
    clock = [0.0]
    monkeypatch.setattr(discovery, "_now", lambda: clock[0])

    # Peer seen once, then vanishes without ever sending a "Rmv".
    backend.scan_results.append({"peer-a": ("peer-a.local", _txt_for("peer-a"))})
    cd._scan_and_reconcile()
    assert [p.info.node_id for p in cd.peers] == ["peer-a"]

    # Not yet past the expiry window: still present.
    clock[0] = 29.0
    backend.scan_results.append({})
    cd._scan_and_reconcile()
    assert [p.info.node_id for p in cd.peers] == ["peer-a"]

    # Past the window: expired and reported as "left".
    clock[0] = 31.0
    backend.scan_results.append({})
    cd._scan_and_reconcile()
    assert cd.peers == []
    assert events == ["joined", "left"]


def test_duplicate_node_id_logs_a_warning_but_keeps_one_entry(caplog):
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="me", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    backend.scan_results.append({"instance-a": ("host-a.local", _txt_for("peer-a"))})
    cd._scan_and_reconcile()

    with caplog.at_level("WARNING", logger="omlx.cluster.discovery"):
        backend.scan_results.append({"instance-b": ("host-b.local", _txt_for("peer-a"))})
        cd._scan_and_reconcile()

    assert len(cd.peers) == 1
    assert cd.peers[0].host == "host-b.local"
    assert any("advertised by both" in record.message for record in caplog.records)


def test_malformed_txt_from_a_peer_is_ignored_not_fatal():
    backend = FakeBackend()
    cd = ClusterDiscovery(
        node_id="me", port=8888, version="0.5.3", cluster_key="k", backend=backend
    )
    backend.scan_results.append({"bad-peer": ("host.local", {"port": "not-a-number"})})

    cd._scan_and_reconcile()  # must not raise

    assert cd.peers == []


def test_observer_exception_does_not_break_reconciliation():
    backend = FakeBackend()

    def bad_observer(event, peer):
        raise ValueError("boom")

    cd = ClusterDiscovery(
        node_id="me",
        port=8888,
        version="0.5.3",
        cluster_key="k",
        backend=backend,
        on_event=bad_observer,
    )
    backend.scan_results.append({"peer-a": ("peer-a.local", _txt_for("peer-a"))})

    cd._scan_and_reconcile()  # must not raise

    assert [p.info.node_id for p in cd.peers] == ["peer-a"]


# =============================================================================
# compatible-cluster check used by callers consuming the peer table
# =============================================================================


def test_peer_with_matching_key_fingerprint_is_compatible():
    peer = Peer(
        info=PeerInfo(
            node_id="peer-a",
            version="0.5.3",
            port=8888,
            chip="Apple M5 Max",
            ram_gb=128,
            key_fingerprint=fingerprint("shared-secret"),
        ),
        host="peer-a.local",
        last_seen=0.0,
    )
    assert matches_fingerprint("shared-secret", peer.info.key_fingerprint)
    assert not matches_fingerprint("wrong-secret", peer.info.key_fingerprint)


# =============================================================================
# Default local-facts probes degrade gracefully without the binaries
# =============================================================================


def test_default_node_id_falls_back_to_hostname_without_ioreg(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    monkeypatch.setattr(discovery.socket, "gethostname", lambda: "fallback-host")
    assert discovery.default_node_id() == "fallback-host"


def test_default_chip_returns_unknown_without_sysctl(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    assert discovery.default_chip() == "unknown"


def test_default_ram_gb_returns_zero_without_sysctl(monkeypatch):
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    assert discovery.default_ram_gb() == 0
