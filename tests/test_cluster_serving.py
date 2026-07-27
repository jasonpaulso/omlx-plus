# SPDX-License-Identifier: Apache-2.0
"""The serving path: launcher plumbing, formation, routes, and the engine.

Nothing here spawns a rank or joins a collective. What is worth testing at this
level is the wiring that fails *quietly* in production - an abort that never
reaches the pipe, a peer told to start before rank 0 is listening, a cluster
model admitted against the local memory ceiling - not mlx itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from dataclasses import dataclass, field

import pytest

from omlx.cluster import bootstrap, routes
from omlx.cluster.launcher import LocalCluster, RankProcess, resolve_python
from omlx.cluster.manager import (
    ClusterFormationError,
    ClusterManager,
    NodeSlot,
    local_ip,
)
from omlx.cluster.protocol import GenerationSpec


# =============================================================================
# Fakes
# =============================================================================


@dataclass
class FakeClusterSettings:
    enabled: bool = True
    cluster_key: str = "shared-secret"
    backend: str = "auto"
    model: str = "big-model"
    pipeline: bool = False
    # Real default is 5s. Formation waits three polls for Bonjour, and the
    # tests would rather not.
    discovery_interval_seconds: float = 0.05


@dataclass
class FakeServerSettings:
    port: int = 8888


@dataclass
class FakeSettings:
    cluster: FakeClusterSettings = field(default_factory=FakeClusterSettings)
    server: FakeServerSettings = field(default_factory=FakeServerSettings)

    def get_effective_model_dirs(self):
        return []


class _FakePeerInfo:
    def __init__(self, node_id: str, port: int) -> None:
        self.node_id = node_id
        self.port = port
        self.chip = "Apple M3 Ultra"
        self.ram_gb = 96
        self.version = "0.0.0"


class _FakePeer:
    def __init__(self, node_id: str, host: str, port: int) -> None:
        self.info = _FakePeerInfo(node_id, port)
        self.host = host


@pytest.fixture
def quick_formation(monkeypatch):
    """Formation without the real disk, interpreter probe or discovery grace."""
    monkeypatch.setattr(
        "omlx.cluster.manager.resolve_model_path", lambda s, m: "/models/big"
    )
    monkeypatch.setattr(
        "omlx.cluster.manager.resolve_python", lambda *a: sys.executable
    )
    monkeypatch.setattr("omlx.cluster.manager.PEER_DISCOVERY_GRACE_S", 0.3)
    # Peer addressing has its own tests; here the fake peers are already
    # literal addresses.
    monkeypatch.setattr(
        "omlx.cluster.manager.resolve_ipv4", lambda host, port: host
    )
    return None


class FakeProcess:
    """Just enough of `subprocess.Popen` for the launcher's bookkeeping."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.stdin = None
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


# =============================================================================
# Launcher
# =============================================================================


class TestResolvePython:
    def test_accepts_an_interpreter_that_can_import_omlx(self):
        assert resolve_python(sys.executable) == sys.executable

    def test_rejects_one_that_cannot(self, tmp_path):
        """The failure that otherwise surfaces on the *other* machine.

        A rank whose interpreter cannot import oMLX dies before it listens, and
        the peer reports `[ring] Couldn't connect` - which reads as a firewall
        fault and is not one.
        """
        fake = tmp_path / "python"
        fake.write_text("#!/bin/sh\nexit 1\n")
        fake.chmod(0o755)
        with pytest.raises(RuntimeError, match="cannot import omlx"):
            resolve_python(str(fake))


class TestAbortSignal:
    def test_no_ranks_means_nothing_to_abort(self):
        assert LocalCluster(model_path="m", world_size=2).abort() is False

    def test_reaches_the_control_pipe(self):
        read_fd, write_fd = os.pipe()
        try:
            cluster = LocalCluster(model_path="m", world_size=2)
            cluster.ranks.append(
                RankProcess(rank=0, process=FakeProcess(), control_w=write_fd)
            )
            assert cluster.abort() is True
            assert b"abort" in os.read(read_fd, 4096)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_node_without_rank_zero_cannot_abort(self):
        """Peers learn about an abort through the collective, from rank 0."""
        cluster = LocalCluster(model_path="m", world_size=2)
        cluster.ranks.append(RankProcess(rank=1, process=FakeProcess()))
        assert cluster.abort() is False

    def test_a_dead_rank_zero_cannot_be_aborted(self):
        read_fd, write_fd = os.pipe()
        try:
            cluster = LocalCluster(model_path="m", world_size=2)
            cluster.ranks.append(
                RankProcess(
                    rank=0, process=FakeProcess(alive=False), control_w=write_fd
                )
            )
            assert cluster.abort() is False
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_stop_closes_the_control_pipe(self):
        read_fd, write_fd = os.pipe()
        cluster = LocalCluster(model_path="m", world_size=2)
        cluster.ranks.append(
            RankProcess(rank=0, process=FakeProcess(), control_w=write_fd)
        )
        cluster.stop(timeout=0.1)
        # The worker must see the pipe close when its daemon goes away.
        with pytest.raises(OSError):
            os.write(write_fd, b"x")
        os.close(read_fd)


# =============================================================================
# Formation
# =============================================================================


class TestRankOrder:
    def test_this_node_becomes_rank_zero(self):
        order = ClusterManager._rank_order(["a", "b", "c"], "b")
        assert order[0] == "b"

    def test_rotation_preserves_ring_adjacency(self):
        """`jaccl-ring` needs consecutive ranks physically adjacent.

        Rotating a cycle keeps every node next to the neighbours it was cabled
        to; re-sorting would not.
        """
        planned = ["a", "b", "c", "d"]
        assert ClusterManager._rank_order(planned, "c") == ["c", "d", "a", "b"]

    def test_an_unplanned_node_still_leads(self):
        assert ClusterManager._rank_order(["a", "b"], "z") == ["z", "a", "b"]


class TestClusterManager:
    def test_starts_unformed(self):
        manager = ClusterManager(FakeSettings())
        status = manager.status()
        assert status.enabled is True
        assert status.formed is False
        assert status.world_size == 0

    def test_reports_disabled_when_the_setting_is_off(self):
        settings = FakeSettings()
        settings.cluster.enabled = False
        assert ClusterManager(settings).status().enabled is False

    def test_forming_with_clustering_off_is_refused(self):
        settings = FakeSettings()
        settings.cluster.enabled = False
        with pytest.raises(ClusterFormationError, match="enabled is false"):
            ClusterManager(settings).form("big-model")

    def test_forming_without_a_key_is_refused(self):
        settings = FakeSettings()
        settings.cluster.cluster_key = ""
        with pytest.raises(ClusterFormationError, match="cluster_key"):
            ClusterManager(settings).form("big-model")

    def test_a_cluster_of_one_is_refused(self, monkeypatch, quick_formation):
        """No peers is not a degenerate cluster, it is a local model."""
        with pytest.raises(ClusterFormationError, match="no peers"):
            ClusterManager(FakeSettings(), peers=list).form("big-model")

    def test_a_failed_formation_leaves_nothing_running(
        self, monkeypatch, quick_formation
    ):
        manager = ClusterManager(FakeSettings(), peers=list)
        with pytest.raises(ClusterFormationError):
            manager.form("big-model")
        assert manager.formed is False
        assert manager.status().error

    def test_waits_for_bonjour_rather_than_failing_instantly(
        self, monkeypatch, quick_formation
    ):
        """Discovery is a poll, and the engine pool makes a failed load sticky.

        Failing on the first empty peer table would turn "Bonjour has not
        answered yet" into "this model is broken until you reload".
        """
        calls = {"n": 0}

        def peers_appear_late():
            calls["n"] += 1
            if calls["n"] < 3:
                return []
            # Refuses instantly, so formation fails at the report call rather
            # than spending a timeout on an address that swallows packets.
            return [_FakePeer("studio", "127.0.0.1", 1)]

        manager = ClusterManager(FakeSettings(), peers=peers_appear_late)
        # Formation gets past the peer check and fails later, on the report
        # call to a peer that does not exist - which is the point.
        with pytest.raises(ClusterFormationError) as exc:
            manager.form("big-model")
        assert "no peers" not in str(exc.value)
        assert calls["n"] >= 3

    def test_streaming_without_a_cluster_is_refused(self):
        manager = ClusterManager(FakeSettings())
        with pytest.raises(ClusterFormationError, match="no cluster"):
            list(manager.stream(GenerationSpec(prompt_ids=[1])))

    def test_teardown_is_safe_when_nothing_is_formed(self):
        ClusterManager(FakeSettings()).teardown()

    def test_abort_without_a_cluster_is_false(self):
        assert ClusterManager(FakeSettings()).abort() is False


def test_local_ip_is_an_address():
    address = local_ip()
    assert address.count(".") == 3


# =============================================================================
# Peer control plane
# =============================================================================


class TestClusterKeyAuth:
    def test_matching_key_is_accepted(self):
        routes.configure(lambda: FakeSettings(), lambda: None)
        assert routes.verify_cluster_key("shared-secret") is True

    def test_wrong_key_is_refused(self):
        from fastapi import HTTPException

        routes.configure(lambda: FakeSettings(), lambda: None)
        with pytest.raises(HTTPException) as exc:
            routes.verify_cluster_key("guess")
        assert exc.value.status_code == 403

    def test_a_node_with_clustering_off_refuses_outright(self):
        from fastapi import HTTPException

        settings = FakeSettings()
        settings.cluster.enabled = False
        routes.configure(lambda: settings, lambda: None)
        with pytest.raises(HTTPException, match="not enabled"):
            routes.verify_cluster_key("shared-secret")

    def test_an_empty_key_never_authenticates(self):
        """A node that has not been given a key is not a node with key ''."""
        from fastapi import HTTPException

        settings = FakeSettings()
        settings.cluster.cluster_key = ""
        routes.configure(lambda: settings, lambda: None)
        with pytest.raises(HTTPException):
            routes.verify_cluster_key("")

    def test_stopping_ranks_is_safe_when_none_are_running(self):
        assert asyncio.run(routes.stop_ranks()) == {"ok": True, "stopped": 0}


# =============================================================================
# Engine selection
# =============================================================================


class TestEngineSelection:
    def test_no_manager_means_no_cluster_model(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "_manager", None)
        assert bootstrap.serves_cluster_model("big-model") is False

    def test_only_the_configured_model_is_cluster_served(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "_manager", ClusterManager(FakeSettings()))
        assert bootstrap.serves_cluster_model("big-model") is True
        assert bootstrap.serves_cluster_model("some-other-model") is False

    def test_an_unset_cluster_model_matches_nothing(self, monkeypatch):
        """An empty `cluster.model` must not swallow an empty model id."""
        settings = FakeSettings()
        settings.cluster.model = ""
        monkeypatch.setattr(bootstrap, "_manager", ClusterManager(settings))
        assert bootstrap.serves_cluster_model("") is False

    def test_disabled_clustering_serves_nothing_over_the_cluster(self, monkeypatch):
        settings = FakeSettings()
        settings.cluster.enabled = False
        monkeypatch.setattr(bootstrap, "_manager", ClusterManager(settings))
        assert bootstrap.serves_cluster_model("big-model") is False

    def test_build_engine_returns_none_for_local_models(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "_manager", ClusterManager(FakeSettings()))
        assert (
            bootstrap.build_engine(
                model_id="local-model",
                model_path="/models/local",
                trust_remote_code=False,
                model_settings=None,
            )
            is None
        )

    def test_build_engine_returns_a_cluster_engine_for_the_cluster_model(
        self, monkeypatch
    ):
        from omlx.cluster.engine import ClusterEngine

        monkeypatch.setattr(bootstrap, "_manager", ClusterManager(FakeSettings()))
        engine = bootstrap.build_engine(
            model_id="big-model",
            model_path="/models/big",
            trust_remote_code=False,
            model_settings=None,
        )
        assert isinstance(engine, ClusterEngine)


# =============================================================================
# The engine
# =============================================================================


class FakeManager:
    """A formed cluster that replays a scripted reply stream."""

    def __init__(self, replies, *, stall: bool = False) -> None:
        self._replies = replies
        self._stall = stall
        self.aborted = 0
        self.formed_for = None
        self.torn_down = 0

    def form(self, model_id):
        self.formed_for = model_id

    def teardown(self):
        self.torn_down += 1

    def abort(self):
        self.aborted += 1
        return True

    def status(self):
        from omlx.cluster.manager import ClusterStatus

        return ClusterStatus(
            enabled=True,
            formed=True,
            backend="ring",
            world_size=2,
            nodes=[
                NodeSlot("macbook", "10.0.0.1", 8888, 0, True),
                NodeSlot("studio", "10.0.0.2", 8888, 1),
            ],
        )

    def stream(self, spec):
        yield from self._replies


class FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]


def build_engine(replies) -> "object":
    from omlx.cluster.engine import ClusterEngine

    manager = FakeManager(replies)
    engine = ClusterEngine(
        model_name="/models/big", model_id="big-model", manager=manager
    )
    engine._tokenizer = FakeTokenizer()
    engine._loaded = True
    return engine, manager


COMPLETE = [
    {"ok": True, "chunk": "Hello", "tokens": 1},
    {"ok": True, "chunk": " world", "tokens": 2},
    {
        "ok": True,
        "done": True,
        "text": "Hello world",
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "finish_reason": "stop",
    },
]


class TestClusterEngine:
    def test_streams_incremental_text(self):
        engine, _ = build_engine(COMPLETE)

        async def run():
            return [o async for o in engine.stream_generate("hi", max_tokens=8)]

        outputs = asyncio.run(run())
        assert [o.new_text for o in outputs[:-1]] == ["Hello", " world"]
        assert outputs[-1].finished is True
        assert outputs[-1].text == "Hello world"
        assert outputs[-1].finish_reason == "stop"

    def test_reports_usage_from_the_worker(self):
        engine, _ = build_engine(COMPLETE)

        async def run():
            return await engine.generate("hi")

        output = asyncio.run(run())
        assert output.prompt_tokens == 3
        assert output.completion_tokens == 2

    def test_a_completed_run_is_not_aborted(self):
        engine, manager = build_engine(COMPLETE)

        async def run():
            return await engine.generate("hi")

        asyncio.run(run())
        assert manager.aborted == 0

    def test_a_client_that_leaves_early_aborts_the_cluster(self):
        """Nothing else can be served until the ranks leave the decode loop."""
        engine, manager = build_engine(COMPLETE)

        async def run():
            async for _ in engine.stream_generate("hi"):
                break

        asyncio.run(run())
        assert manager.aborted == 1

    def test_a_worker_error_surfaces(self):
        engine, _ = build_engine([{"ok": True, "chunk": "x", "tokens": 1}])

        def explode(spec):
            yield {"ok": True, "chunk": "x", "tokens": 1}
            raise RuntimeError("rank 1 died")

        engine._manager.stream = explode

        async def run():
            async for _ in engine.stream_generate("hi"):
                pass

        with pytest.raises(RuntimeError, match="rank 1 died"):
            asyncio.run(run())

    def test_no_prefix_cache_in_cluster_mode(self):
        """A local cache hit is exactly the state ranks may not branch on."""
        engine, _ = build_engine(COMPLETE)
        assert engine.prefix_cache_enabled is False
        assert engine.get_cache_stats() is None

    def test_structured_output_is_refused_rather_than_ignored(self):
        engine, _ = build_engine(COMPLETE)
        assert engine.grammar_compiler is None

    def test_stats_describe_the_cluster(self):
        engine, _ = build_engine(COMPLETE)
        stats = engine.get_stats()
        assert stats["engine_type"] == "cluster"
        assert stats["backend"] == "ring"
        assert stats["nodes"] == ["macbook", "studio"]

    def test_neutral_penalties_are_sent_as_none(self):
        """mlx-lm reads "no penalty" as None, not as 1.0/0.0.

        Passing the neutral value builds a logits processor that runs on every
        token to change nothing.
        """
        engine, manager = build_engine(COMPLETE)
        captured = {}

        def capture(spec):
            captured["spec"] = spec
            yield from COMPLETE

        manager.stream = capture

        async def run():
            await engine.generate("hi", repetition_penalty=1.0, presence_penalty=0.0)

        asyncio.run(run())
        assert captured["spec"].repetition_penalty is None
        assert captured["spec"].presence_penalty is None

    def test_real_penalties_are_forwarded(self):
        engine, manager = build_engine(COMPLETE)
        captured = {}

        def capture(spec):
            captured["spec"] = spec
            yield from COMPLETE

        manager.stream = capture

        async def run():
            await engine.generate("hi", repetition_penalty=1.2, presence_penalty=0.5)

        asyncio.run(run())
        assert captured["spec"].repetition_penalty == 1.2
        assert captured["spec"].presence_penalty == 0.5

    def test_stop_returns_the_cluster(self):
        engine, manager = build_engine(COMPLETE)
        asyncio.run(engine.stop())
        assert manager.torn_down == 1
        assert engine._loaded is False


# =============================================================================
# Peer addressing
# =============================================================================


class FakeProbe:
    """A socket whose `connect_ex` succeeds only for one address."""

    reachable: str = ""
    attempts: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def settimeout(self, _timeout):
        pass

    def connect_ex(self, address):
        FakeProbe.attempts.append(address[0])
        return 0 if address[0] == FakeProbe.reachable else 1


class TestResolveIpv4:
    """A Mac with a Thunderbolt cable publishes several A records.

    The first is routinely a link-local `169.254.x.x` belonging to a bridge
    nobody serves on. A rank handed that address binds where its peers cannot
    see it, and the run dies with `[ring] Couldn't connect (error: 60)` - a
    timeout that reads like a firewall fault and is not one. This was a real
    two-machine failure, not a hypothetical.
    """

    def _addresses(self, monkeypatch, addresses):
        import socket as socket_module

        monkeypatch.setattr(
            "omlx.cluster.manager.socket.getaddrinfo",
            lambda *a, **k: [
                (socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", (addr, 0))
                for addr in addresses
            ],
        )

    def _probe(self, monkeypatch, reachable):
        FakeProbe.reachable = reachable
        FakeProbe.attempts = []
        monkeypatch.setattr("omlx.cluster.manager.socket.socket", FakeProbe)

    def test_skips_link_local(self, monkeypatch):
        from omlx.cluster import manager as manager_module

        self._addresses(monkeypatch, ["169.254.214.89", "192.168.4.32"])
        self._probe(monkeypatch, "192.168.4.32")
        assert manager_module.resolve_ipv4("studio.local", 8888) == "192.168.4.32"
        assert "169.254.214.89" not in FakeProbe.attempts

    def test_prefers_the_address_that_answers(self, monkeypatch):
        """Evidence, not ordering: a route that works is one that connects."""
        from omlx.cluster import manager as manager_module

        self._addresses(monkeypatch, ["10.9.9.9", "192.168.5.28"])
        self._probe(monkeypatch, "192.168.5.28")
        assert manager_module.resolve_ipv4("studio.local", 8888) == "192.168.5.28"

    def test_falls_back_rather_than_refusing_to_form(self, monkeypatch):
        """A peer mid-restart is not the same as a peer that is unreachable."""
        from omlx.cluster import manager as manager_module

        self._addresses(monkeypatch, ["192.168.4.32", "192.168.5.28"])
        self._probe(monkeypatch, "nothing answers")
        assert manager_module.resolve_ipv4("studio.local", 8888) == "192.168.4.32"

    def test_no_routable_address_is_an_error(self, monkeypatch):
        from omlx.cluster import manager as manager_module

        self._addresses(monkeypatch, ["169.254.1.1", "127.0.0.1"])
        with pytest.raises(ClusterFormationError, match="routable"):
            manager_module.resolve_ipv4("studio.local", 8888)

    def test_a_name_that_does_not_resolve_names_itself(self, monkeypatch):
        from omlx.cluster import manager as manager_module

        def boom(*args, **kwargs):
            raise OSError("nodename nor servname provided")

        monkeypatch.setattr("omlx.cluster.manager.socket.getaddrinfo", boom)
        with pytest.raises(ClusterFormationError, match="studio.local"):
            manager_module.resolve_ipv4("studio.local", 8888)


# =============================================================================
# Stale ranks
# =============================================================================


class TestOrphanSweep:
    """A worker left alive by a failed run holds its ring port.

    The next rank 0 then quietly fails to own it and the whole cluster dies
    with a connect timeout that looks like a network fault.
    """

    @staticmethod
    def _orphan(argv_tag: str) -> int:
        """Spawn a decoy whose parent exits, so it reparents to launchd.

        A decoy started directly by the test would be *our* child, and the
        sweep deliberately spares those.
        """
        import subprocess
        import time

        marker = f"/tmp/omlx-sweep-test-{argv_tag}-{os.getpid()}"
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys, os\n"
                    f"p = subprocess.Popen([sys.executable, '-c',"
                    f" 'import time; time.sleep(60)', {argv_tag!r}])\n"
                    f"open({marker!r}, 'w').write(str(p.pid))\n"
                ),
            ]
        )
        launcher.wait(timeout=10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                return int(open(marker).read())
            except (OSError, ValueError):
                time.sleep(0.05)
        raise AssertionError("decoy never reported its pid")

    @staticmethod
    def _alive(pid: int) -> bool:
        import psutil

        return psutil.pid_exists(pid)

    def test_kills_an_orphaned_rank(self):
        import time

        from omlx.cluster.launcher import WORKER_MODULE, sweep_orphaned_ranks

        pid = self._orphan(WORKER_MODULE)
        try:
            deadline = time.monotonic() + 5
            while self._alive(pid) and time.monotonic() < deadline:
                sweep_orphaned_ranks()
                time.sleep(0.1)
            assert not self._alive(pid)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)

    def test_spares_a_rank_this_process_owns(self):
        """Two daemons on one machine is how this is tested without a 2nd Mac.

        A peer forming its rank must not kill the leader's rank 0 next to it.
        """
        import subprocess

        from omlx.cluster.launcher import WORKER_MODULE, sweep_orphaned_ranks

        mine = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", WORKER_MODULE]
        )
        try:
            sweep_orphaned_ranks()
            assert mine.poll() is None
        finally:
            mine.kill()
            mine.wait(timeout=5)

    def test_leaves_everything_else_alone(self):
        pid = self._orphan("some.other.module")
        try:
            from omlx.cluster.launcher import sweep_orphaned_ranks

            sweep_orphaned_ranks()
            assert self._alive(pid)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)


# =============================================================================
# Reply reading
# =============================================================================


class TestReplyReader:
    """The idle timeout has to be honest about buffered data.

    `select` reports on the file descriptor, and a buffered reader routinely
    pulls several replies out of the pipe at once - so a wrapper-based
    implementation would sit waiting for bytes it already had.
    """

    def test_reads_lines_in_order(self):
        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"a": 1}\n{"b": 2}\n')
            reader = ReplyReader(read_fd)
            assert reader.readline(1.0) == '{"a": 1}'
            assert reader.readline(1.0) == '{"b": 2}'
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_second_buffered_line_does_not_time_out(self):
        """The exact failure a `select`-on-the-wrapper version would have."""
        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"a": 1}\n{"b": 2}\n')
            reader = ReplyReader(read_fd)
            reader.readline(1.0)
            # Nothing new will ever arrive on the fd, but line two is in hand.
            assert reader.readline(0.05) == '{"b": 2}'
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_silence_times_out(self):
        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        try:
            assert ReplyReader(read_fd).readline(0.05) is None
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_closed_pipe_is_eof_not_a_timeout(self):
        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            assert ReplyReader(read_fd).readline(1.0) == ""
        finally:
            os.close(read_fd)

    def test_no_timeout_blocks_until_the_line_arrives(self):
        import threading

        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        try:
            timer = threading.Timer(0.2, lambda: os.write(write_fd, b'{"late": 1}\n'))
            timer.start()
            assert ReplyReader(read_fd).readline(None) == '{"late": 1}'
            timer.join()
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_partial_line_is_not_a_line(self):
        from omlx.cluster.launcher import ReplyReader

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"half"')
            reader = ReplyReader(read_fd)
            assert reader.readline(0.05) is None
            os.write(write_fd, b': 1}\n')
            assert reader.readline(1.0) == '{"half": 1}'
        finally:
            os.close(read_fd)
            os.close(write_fd)
