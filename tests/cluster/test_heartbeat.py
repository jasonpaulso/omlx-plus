# SPDX-License-Identifier: Apache-2.0
"""Tests for the worker-side heartbeat loop."""

import asyncio

import pytest

from omlx.cluster.client import ClusterClientError
from omlx.cluster.heartbeat import HeartbeatSender
from omlx.cluster.state import WorkerIdentity

from .conftest import FakeClusterClient

IDENTITY = WorkerIdentity(
    member_id="m-1", secret="s" * 64, head_url="http://10.0.0.1:8000", joined_at=1.0
)


def sender(replies=None, interval_s=0.01) -> tuple[HeartbeatSender, list]:
    created: list[FakeClusterClient] = []

    def factory(url: str) -> FakeClusterClient:
        client = FakeClusterClient(url, replies or {"/v1/cluster/heartbeat": {}})
        created.append(client)
        return client

    return (
        HeartbeatSender(IDENTITY, interval_s=interval_s, client_factory=factory),
        created,
    )


class TestPayload:
    async def test_sequence_increases_within_one_epoch(self):
        beat, created = sender()
        await beat.send_once()
        await beat.send_once()
        await beat.send_once()

        payloads = [call[3] for client in created for call in client.calls]
        assert [p["seq"] for p in payloads] == [1, 2, 3]
        assert len({p["epoch"] for p in payloads}) == 1

    async def test_epoch_is_minted_per_runtime_not_per_beat(self):
        first, _ = sender()
        second, _ = sender()
        assert first.epoch != second.epoch
        assert len(first.epoch) == 16

    async def test_the_member_secret_authenticates_the_beat(self):
        beat, created = sender()
        await beat.send_once()
        method, path, token, _payload = created[0].calls[0]
        assert (method, path, token) == ("POST", "/v1/cluster/heartbeat", "s" * 64)

    async def test_sequence_still_advances_after_a_failure(self):
        """A retry must not reuse a sequence the head may already have seen."""
        beat, created = sender(
            replies={"/v1/cluster/heartbeat": ClusterClientError("boom")}
        )
        assert await beat.send_once() is None
        assert beat.last_error is not None
        assert beat.seq == 1

        beat._client_factory = lambda url: FakeClusterClient(
            url, {"/v1/cluster/heartbeat": {"status": "active"}}
        )
        assert await beat.send_once() == {"status": "active"}
        assert beat.seq == 2
        assert beat.last_error is None


class TestLoop:
    async def test_loop_sends_repeatedly_until_stopped(self):
        beat, created = sender(interval_s=0.01)
        await beat.start()
        try:
            for _ in range(100):
                if beat.seq >= 3:
                    break
                await asyncio.sleep(0.01)
        finally:
            await beat.stop()

        assert beat.seq >= 3
        assert not beat.running
        sent = beat.seq
        await asyncio.sleep(0.05)
        assert beat.seq == sent

    async def test_loop_survives_a_failing_head(self):
        beat, _ = sender(
            replies={"/v1/cluster/heartbeat": ClusterClientError("head down")},
            interval_s=0.01,
        )
        await beat.start()
        try:
            await asyncio.sleep(0.05)
            assert beat.running
            assert beat.last_error is not None
        finally:
            await beat.stop()

    async def test_start_is_idempotent(self):
        beat, _ = sender()
        await beat.start()
        await beat.start()
        try:
            assert beat.running
        finally:
            await beat.stop()

    async def test_stop_without_start_is_safe(self):
        beat, _ = sender()
        await beat.stop()
        assert not beat.running


class TestNodeState:
    """S4 D1: advisory node_state attached to each heartbeat."""

    async def test_node_state_is_attached_when_provider_is_set(self):
        beat, created = sender()
        beat._node_state_provider = lambda: {
            "total_memory": 1000,
            "memory_ceiling": 100,
            "models_present": {"m": 50},
        }
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert payload["node_state"] == {
            "total_memory": 1000,
            "memory_ceiling": 100,
            "models_present": {"m": 50},
        }

    async def test_node_state_is_omitted_when_provider_returns_none(self):
        beat, created = sender()
        beat._node_state_provider = lambda: None
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert "node_state" not in payload

    async def test_node_state_is_omitted_when_no_provider_is_set(self):
        beat, created = sender()
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert "node_state" not in payload

    async def test_a_raising_provider_never_fails_the_beat(self):
        beat, created = sender()

        def boom():
            raise RuntimeError("scan failed")

        beat._node_state_provider = boom
        reply = await beat.send_once()
        assert reply is not None
        payload = created[0].calls[0][3]
        assert "node_state" not in payload


class TestRanksStatus:
    """S6 D1: the worker-side ranks provider riding each heartbeat."""

    async def test_ranks_is_attached_when_provider_is_set(self):
        beat, created = sender()
        beat._ranks_provider = lambda: {"alive": [1], "dead": []}
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert payload["ranks"] == {"alive": [1], "dead": []}

    async def test_ranks_is_omitted_when_provider_returns_none(self):
        beat, created = sender()
        beat._ranks_provider = lambda: None
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert "ranks" not in payload

    async def test_ranks_is_omitted_when_no_provider_is_set(self):
        beat, created = sender()
        await beat.send_once()
        payload = created[0].calls[0][3]
        assert "ranks" not in payload

    async def test_a_raising_ranks_provider_never_fails_the_beat(self):
        beat, created = sender()

        def boom():
            raise RuntimeError("cluster read failed")

        beat._ranks_provider = boom
        reply = await beat.send_once()
        assert reply is not None
        payload = created[0].calls[0][3]
        assert "ranks" not in payload


class TestBackoff:
    """S6 D2: heartbeat failure backoff, pinned to the negotiated interval."""

    def test_delay_sequence_at_the_default_interval(self):
        beat, _ = sender(interval_s=5.0)
        assert beat._next_delay() == 5.0  # healthy: the base interval
        beat._consecutive_failures = 1
        assert beat._next_delay() == 5.0
        beat._consecutive_failures = 2
        assert beat._next_delay() == 10.0
        beat._consecutive_failures = 3
        assert beat._next_delay() == 20.0
        beat._consecutive_failures = 4
        assert beat._next_delay() == 20.0  # capped at 4x
        beat._consecutive_failures = 50
        assert beat._next_delay() == 20.0  # still capped, deep backoff

    def test_delay_pins_the_negotiated_interval_not_a_different_default(self):
        # A NON-default negotiated interval (10s): cap is 4x THAT interval
        # (40s), not the 5s-default cap.
        beat, _ = sender(interval_s=10.0)
        beat._consecutive_failures = 1
        assert beat._next_delay() == 10.0
        beat._consecutive_failures = 2
        assert beat._next_delay() == 20.0
        beat._consecutive_failures = 3
        assert beat._next_delay() == 40.0
        beat._consecutive_failures = 10
        assert beat._next_delay() == 40.0

    async def test_a_failure_then_success_resets_the_backoff_to_base(self):
        """The counting/reset itself lives in `_loop` (`send_once` only
        reports success/failure per call), so this drives the real loop
        rather than calling `send_once` directly."""
        attempts = {"n": 0}

        def factory(url):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return FakeClusterClient(
                    url, {"/v1/cluster/heartbeat": ClusterClientError("down")}
                )
            return FakeClusterClient(
                url, {"/v1/cluster/heartbeat": {"status": "active"}}
            )

        # A comfortable interval between beats: attempt 1 fails, then the
        # loop sleeps `_next_delay()` (~0.3s, the base interval, at
        # failures==1) before attempt 2 -- plenty of window to observe the
        # counted failure before it resets on success.
        beat = HeartbeatSender(IDENTITY, interval_s=0.3, client_factory=factory)
        await beat.start()
        try:
            deadline = asyncio.get_event_loop().time() + 2.0
            while beat.seq < 1 and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert beat.seq >= 1, "the loop never reached its first (failing) beat"
            assert beat._consecutive_failures == 1  # the failure WAS counted

            while beat.seq < 2 and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert beat.seq >= 2, "the loop never reached a second (successful) beat"
            assert beat._consecutive_failures == 0
            assert beat._next_delay() == beat._base_interval_s
        finally:
            await beat.stop()

    @pytest.mark.parametrize("interval", [0.5, 1.0, 5.0, 5.01, 10.0, 30.0, 100.0])
    def test_the_cap_never_exceeds_the_real_command_timeout(self, interval):
        """rev4's provable invariant: `4i < max(30, 6i)` for any interval --
        a backed-off worker can never miss a formation command window
        because of its own backoff. Ties the heartbeat's real cap to
        FormationManager's real `_command_timeout` formula, not two
        independently-reimplemented constants.
        """
        import types

        from omlx.cluster.formation import FormationManager

        beat, _ = sender(interval_s=interval)
        beat._consecutive_failures = 1000  # deep backoff -- definitely capped
        cap = beat._next_delay()

        fm = FormationManager.__new__(FormationManager)
        fm._manager = types.SimpleNamespace(
            settings=types.SimpleNamespace(heartbeat_interval_s=interval)
        )
        command_timeout = fm._command_timeout

        assert cap == pytest.approx(interval * 4.0)
        assert cap < command_timeout


class TestStatus:
    async def test_status_reports_epoch_and_sequence_without_the_secret(self):
        beat, _ = sender()
        await beat.send_once()
        status = beat.status()
        assert status["seq"] == 1
        assert status["epoch"] == beat.epoch
        assert "s" * 64 not in str(status)
