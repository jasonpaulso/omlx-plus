# SPDX-License-Identifier: Apache-2.0
"""Rank death must fail fast, not at the idle timeout.

A rank that dies leaves every other rank blocked inside a collective - mlx has
no fault tolerance - and before the deathwatch existed the only thing that
ended the wait was the generate idle timeout (minutes) or the load timeout
(longer). These tests pin the whole fast-fail path: detection, the hard kill
that closes the reply pipe, the error a waiting request sees, and the
re-formation the next request triggers.
"""

import asyncio
import threading
import time

import pytest

from omlx.cluster import routes
from omlx.cluster.launcher import DeathWatch, LocalCluster, RankProcess
from omlx.cluster.manager import ClusterFormationError, ClusterManager, NodeSlot
from omlx.cluster.protocol import GenerationSpec

from test_cluster_serving import FakeProcess, FakeSettings

# Fast enough that a whole watch lifecycle fits in a test, slow enough that a
# loaded CI machine still gets its polls in.
TICK = 0.02


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class Recorder:
    def __init__(self):
        self.deaths = []

    def __call__(self, label, reason):
        self.deaths.append((label, reason))


# =============================================================================
# The watch itself
# =============================================================================


class TestDeathWatch:
    def test_a_definitive_death_fires_on_the_first_poll(self):
        on_death = Recorder()
        watch = DeathWatch([("rank 1", lambda: False)], on_death, interval=TICK)
        watch.start()
        assert wait_for(lambda: on_death.deaths)
        assert on_death.deaths == [("rank 1", "reported dead")]

    def test_unreachable_is_patience_not_death(self):
        """A LAN blip must not kill a healthy formation."""
        on_death = Recorder()
        polls = []
        watch = DeathWatch(
            [("peer", lambda: polls.append(1) and None)],
            on_death,
            interval=TICK,
            strikes=3,
        )
        watch.start()
        assert wait_for(lambda: len(on_death.deaths) == 1)
        assert len(polls) >= 3
        assert "unreachable" in on_death.deaths[0][1]

    def test_one_good_answer_resets_the_strikes(self):
        on_death = Recorder()
        script = [None, None, True, None, None, None]
        polls = []

        def check():
            polls.append(1)
            return script[len(polls) - 1] if len(polls) <= len(script) else None

        watch = DeathWatch([("peer", check)], on_death, interval=TICK, strikes=3)
        watch.start()
        assert wait_for(lambda: on_death.deaths)
        # Fired on the sixth poll, not the third: the True in between reset it.
        assert len(polls) == 6

    def test_a_raising_check_counts_as_unreachable(self):
        on_death = Recorder()

        def broken():
            raise OSError("no route to host")

        watch = DeathWatch([("peer", broken)], on_death, interval=TICK, strikes=2)
        watch.start()
        assert wait_for(lambda: on_death.deaths)
        assert "unreachable" in on_death.deaths[0][1]

    def test_fires_once_then_stands_down(self):
        on_death = Recorder()
        watch = DeathWatch(
            [("a", lambda: False), ("b", lambda: False)], on_death, interval=TICK
        )
        watch.start()
        assert wait_for(lambda: on_death.deaths)
        assert wait_for(lambda: not watch.is_alive())
        assert len(on_death.deaths) == 1

    def test_a_stopped_watch_never_fires(self):
        on_death = Recorder()
        watch = DeathWatch([("rank", lambda: False)], on_death, interval=0.5)
        watch.start()
        watch.stop()
        time.sleep(TICK)
        assert on_death.deaths == []

    def test_the_callback_may_stop_the_watch_without_deadlock(self):
        """Teardown stops the watch, and teardown runs *in* the callback."""
        watch = DeathWatch([("rank", lambda: False)], lambda l, r: watch.stop(), interval=TICK)
        watch.start()
        assert wait_for(lambda: not watch.is_alive())

    def test_healthy_checks_keep_the_watch_alive(self):
        on_death = Recorder()
        watch = DeathWatch([("rank", lambda: True)], on_death, interval=TICK)
        watch.start()
        time.sleep(TICK * 5)
        assert watch.is_alive()
        assert on_death.deaths == []
        watch.stop()


# =============================================================================
# The hard kill
# =============================================================================


class TestKill:
    def _cluster(self, alive=(True, True)):
        cluster = LocalCluster(model_path="/m", world_size=2)
        cluster.ranks = [
            RankProcess(rank=i, process=FakeProcess(alive=a))
            for i, a in enumerate(alive)
        ]
        return cluster

    def test_kill_is_immediate_and_impolite(self):
        cluster = self._cluster()
        cluster.kill()
        assert all(r.process.killed for r in cluster.ranks)

    def test_alive_ranks_reads_the_process_table(self):
        cluster = self._cluster(alive=(True, False))
        assert cluster.alive_ranks() == [0]

    def test_dead_ranks_are_not_killed_again(self):
        cluster = self._cluster(alive=(False,))
        cluster.kill()
        assert cluster.ranks[0].process.killed is False


# =============================================================================
# The manager's side: detection -> kill -> teardown -> the error a request sees
# =============================================================================


def _formed_manager():
    """A manager hand-wired into the formed state, no real processes."""
    manager = ClusterManager(FakeSettings())
    cluster = LocalCluster(model_path="/m", world_size=2)
    cluster.ranks = [RankProcess(rank=0, process=FakeProcess())]
    manager._cluster = cluster
    manager._slots = [
        NodeSlot("macbook", "10.0.0.1", 8888, 0, is_local=True),
        NodeSlot("studio", "10.0.0.2", 8888, 1),
    ]
    manager._model_id = "big-model"
    return manager, cluster


class TestManagerFastFail:
    @pytest.fixture(autouse=True)
    def no_real_http(self, monkeypatch):
        """Teardown tells peers to stop; these tests have no peers to tell."""
        monkeypatch.setattr(
            "omlx.cluster.manager.PeerClient.post",
            lambda self, path, payload=None: {"ok": True},
        )

    @pytest.fixture
    def as_the_watch(self, monkeypatch):
        """Make this test thread pass the callback's identity check.

        `_on_rank_death` only acts when it runs on the thread the manager
        still owns as its watch, and teardown then stops that watch - so the
        test thread needs a no-op `stop`.
        """
        thread = threading.current_thread()
        monkeypatch.setattr(thread, "stop", lambda: None, raising=False)
        return thread

    def test_a_death_kills_teardown_and_records_why(self, as_the_watch):
        manager, cluster = _formed_manager()
        manager._watch = as_the_watch

        manager._on_rank_death("rank 1 on studio", "reported dead")

        assert cluster.ranks == [] or all(not r.alive for r in cluster.ranks)
        assert manager.formed is False
        assert "rank 1 on studio" in manager._error

    def test_a_stale_watch_cannot_kill_the_next_formation(self):
        """The race: old watch fires just as a new formation replaces it."""
        manager, cluster = _formed_manager()
        manager._watch = None  # a new formation already owns the state

        manager._on_rank_death("rank 1 on studio", "reported dead")

        assert manager.formed is True
        assert manager._error == ""

    def test_a_request_arriving_after_the_death_gets_the_reason(self, as_the_watch):
        manager, _ = _formed_manager()
        manager._watch = as_the_watch
        manager._on_rank_death("rank 1 on studio", "unreachable for 5 checks")

        with pytest.raises(ClusterFormationError, match="rank 1 on studio"):
            list(manager.stream(GenerationSpec(prompt_ids=[1])))

    def test_alive_local_ranks_answers_from_the_process_table(self):
        manager, cluster = _formed_manager()
        assert manager.alive_local_ranks() == [0]
        cluster.ranks[0].process.kill()
        assert manager.alive_local_ranks() == []

    def test_no_cluster_means_no_local_ranks(self):
        assert ClusterManager(FakeSettings()).alive_local_ranks() == []


# =============================================================================
# The liveness endpoint every deathwatch polls
# =============================================================================


class TestRanksAlive:
    def _follower(self, alive=(True,)):
        cluster = LocalCluster(model_path="/m", world_size=2)
        cluster.ranks = [
            RankProcess(rank=i + 1, process=FakeProcess(alive=a))
            for i, a in enumerate(alive)
        ]
        return cluster

    def test_reports_follower_ranks(self, monkeypatch):
        monkeypatch.setattr(routes, "_follower", self._follower())
        routes.configure(lambda: FakeSettings(), lambda: None)
        assert asyncio.run(routes.ranks_alive()) == {"ranks": [1]}

    def test_reports_the_leaders_own_ranks(self, monkeypatch):
        manager, _ = _formed_manager()
        monkeypatch.setattr(routes, "_follower", None)
        routes.configure(lambda: FakeSettings(), lambda: manager)
        assert asyncio.run(routes.ranks_alive()) == {"ranks": [0]}

    def test_dead_ranks_are_not_listed(self, monkeypatch):
        monkeypatch.setattr(routes, "_follower", self._follower(alive=(False,)))
        routes.configure(lambda: FakeSettings(), lambda: None)
        assert asyncio.run(routes.ranks_alive()) == {"ranks": []}

    def test_no_ranks_anywhere_is_an_empty_list(self, monkeypatch):
        monkeypatch.setattr(routes, "_follower", None)
        routes.configure(lambda: FakeSettings(), lambda: None)
        assert asyncio.run(routes.ranks_alive()) == {"ranks": []}


# =============================================================================
# The follower watching its own ranks, and its leader
# =============================================================================


class TestFollowerWatch:
    def test_a_dead_local_rank_stops_the_follower(self, monkeypatch):
        cluster = LocalCluster(model_path="/m", world_size=2)
        cluster.ranks = [RankProcess(rank=1, process=FakeProcess(alive=False))]
        monkeypatch.setattr(routes, "_follower", cluster)
        monkeypatch.setattr(routes, "_get_settings", lambda: FakeSettings())
        monkeypatch.setattr("omlx.cluster.launcher.DEATHWATCH_INTERVAL_S", TICK)

        routes._start_follower_watch(cluster, {"ips": ["10.0.0.1"]})
        try:
            assert wait_for(lambda: routes._follower is None)
        finally:
            routes._follower_watch = None
            routes._follower = None

    def test_without_a_leader_port_only_local_ranks_are_watched(self, monkeypatch):
        cluster = LocalCluster(model_path="/m", world_size=2)
        cluster.ranks = [RankProcess(rank=1, process=FakeProcess())]
        monkeypatch.setattr(routes, "_get_settings", lambda: FakeSettings())

        routes._start_follower_watch(cluster, {"ips": ["10.0.0.1"]})
        try:
            assert len(routes._follower_watch._checks) == 1
        finally:
            routes._follower_watch.stop()
            routes._follower_watch = None

    def test_with_a_leader_port_the_leader_is_watched_too(self, monkeypatch):
        cluster = LocalCluster(model_path="/m", world_size=2)
        cluster.ranks = [RankProcess(rank=1, process=FakeProcess())]
        monkeypatch.setattr(routes, "_get_settings", lambda: FakeSettings())

        routes._start_follower_watch(
            cluster, {"ips": ["10.0.0.1"], "leader_port": 8888}
        )
        try:
            assert len(routes._follower_watch._checks) == 2
        finally:
            routes._follower_watch.stop()
            routes._follower_watch = None
