# SPDX-License-Identifier: Apache-2.0
"""Serialized command queue owned by the head (E6).

Every cluster state mutation goes through one consumer task, so two
concurrent formation or membership sequences cannot interleave. Callers
await their command's result, so the queue is invisible at the call site
apart from the ordering guarantee it provides.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ClusterCommandQueue:
    """Runs submitted commands one at a time, in submission order."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, Callable[[], Any], asyncio.Future[Any]]]
        self._queue = asyncio.Queue()
        self._consumer: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._consumer is not None and not self._consumer.done()

    async def start(self) -> None:
        if self.running:
            return
        self._consumer = asyncio.create_task(self._run())

    async def stop(self) -> None:
        consumer = self._consumer
        self._consumer = None
        if consumer is None:
            return
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        while not self._queue.empty():
            _, _, future = self._queue.get_nowait()
            if not future.done():
                future.cancel()

    async def submit(self, name: str, command: Callable[[], Awaitable[T]]) -> T:
        """Enqueue a command and await its result."""
        if not self.running:
            raise RuntimeError("Cluster command queue is not running")
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        await self._queue.put((name, command, future))
        return await future

    async def _run(self) -> None:
        while True:
            name, command, future = await self._queue.get()
            try:
                result = await command()
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                logger.warning("Cluster command %s failed: %s", name, exc)
                if not future.done():
                    future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)
            finally:
                self._queue.task_done()
