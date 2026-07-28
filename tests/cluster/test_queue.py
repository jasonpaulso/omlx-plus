# SPDX-License-Identifier: Apache-2.0
"""Tests for the head's serialized command queue (E6)."""

import asyncio

import pytest

from omlx.cluster.queue import ClusterCommandQueue


class TestSerialization:
    async def test_interleaved_commands_apply_serially(self):
        """Commands never interleave, even when each one awaits."""
        queue = ClusterCommandQueue()
        await queue.start()
        events: list[str] = []

        def command(name: str, delay: float):
            async def _run() -> str:
                events.append(f"start:{name}")
                await asyncio.sleep(delay)
                events.append(f"end:{name}")
                return name

            return _run

        try:
            results = await asyncio.gather(
                queue.submit("a", command("a", 0.03)),
                queue.submit("b", command("b", 0.0)),
                queue.submit("c", command("c", 0.0)),
            )
        finally:
            await queue.stop()

        assert results == ["a", "b", "c"]
        assert events == [
            "start:a",
            "end:a",
            "start:b",
            "end:b",
            "start:c",
            "end:c",
        ]

    async def test_result_is_returned_to_the_caller(self):
        queue = ClusterCommandQueue()
        await queue.start()

        async def _run() -> int:
            return 42

        try:
            assert await queue.submit("answer", _run) == 42
        finally:
            await queue.stop()

    async def test_failure_propagates_to_the_submitter_only(self):
        queue = ClusterCommandQueue()
        await queue.start()

        async def _boom() -> None:
            raise ValueError("nope")

        async def _fine() -> str:
            return "ok"

        try:
            with pytest.raises(ValueError, match="nope"):
                await queue.submit("boom", _boom)
            assert await queue.submit("fine", _fine) == "ok"
        finally:
            await queue.stop()


class TestLifecycle:
    async def test_submit_before_start_is_refused(self):
        queue = ClusterCommandQueue()

        async def _run() -> None:
            return None

        with pytest.raises(RuntimeError, match="not running"):
            await queue.submit("x", _run)

    async def test_start_is_idempotent(self):
        queue = ClusterCommandQueue()
        await queue.start()
        await queue.start()
        try:
            assert queue.running
        finally:
            await queue.stop()

    async def test_stop_makes_the_queue_unusable(self):
        queue = ClusterCommandQueue()
        await queue.start()
        await queue.stop()

        async def _run() -> None:
            return None

        assert not queue.running
        with pytest.raises(RuntimeError):
            await queue.submit("x", _run)

    async def test_stop_is_safe_without_start(self):
        await ClusterCommandQueue().stop()
