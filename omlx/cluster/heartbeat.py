# SPDX-License-Identifier: Apache-2.0
"""Worker-side heartbeat loop.

Each heartbeat carries a runtime ``epoch`` plus a strictly increasing
``seq``. The epoch is minted once per worker cluster-runtime start, so a
worker restart is visible to the head as a new epoch and resets the
sequence, while replay of an old heartbeat inside a live epoch is
rejected (CL-06, partial). The epoch itself is only accepted from a
request that already authenticated with the member secret, so an epoch
reset is not an unauthenticated replay vector.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from .client import ClusterClient, ClusterClientError
from .credentials import generate_epoch
from .state import WorkerIdentity

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = "/v1/cluster/heartbeat"


class HeartbeatSender:
    """Posts heartbeats to the head until stopped."""

    def __init__(
        self,
        identity: WorkerIdentity,
        *,
        interval_s: float,
        client_factory: Callable[[str], ClusterClient] | None = None,
    ) -> None:
        self.identity = identity
        self.interval_s = max(0.1, interval_s)
        self.epoch = generate_epoch()
        self.seq = 0
        self.last_success_at: float | None = None
        self.last_error: str | None = None
        self._client_factory = client_factory or (lambda url: ClusterClient(url))
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def send_once(self) -> dict[str, Any] | None:
        """Send one heartbeat. Returns the head's reply, or None on failure."""
        self.seq += 1
        client = self._client_factory(self.identity.head_url)
        try:
            reply = await client.post_json(
                HEARTBEAT_PATH,
                token=self.identity.secret,
                payload={"seq": self.seq, "epoch": self.epoch},
            )
        except ClusterClientError as exc:
            self.last_error = str(exc)
            logger.warning("Heartbeat to %s failed: %s", self.identity.head_url, exc)
            return None
        self.last_error = None
        self.last_success_at = time.time()
        return reply

    async def _loop(self) -> None:
        while True:
            try:
                await self.send_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                self.last_error = str(exc)
                logger.warning("Heartbeat loop error: %s", exc)
            await asyncio.sleep(self.interval_s)

    def status(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "seq": self.seq,
            "running": self.running,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
        }
