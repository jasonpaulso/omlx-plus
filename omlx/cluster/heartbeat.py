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

from ..admin.auth import fingerprint_key
from .client import ClusterClient, ClusterClientError
from .credentials import digest_secret, generate_epoch, verify_command_response
from .state import WorkerIdentity

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = "/v1/cluster/heartbeat"


class HeartbeatSender:
    """Posts heartbeats to the head until stopped.

    When wired with a ``command_sink``/``job_updates_provider`` (the worker's
    command executor) the heartbeat becomes the D2 command channel: the request
    carries pending ``job_updates`` and the response may carry ``commands``.
    A commands-bearing response is acted on only after it (1) echoes the exact
    epoch+seq of the heartbeat just sent (CL2-06) and (2) carries a valid HMAC
    derived from this worker's own secret (CL2-05). Absent either callback the
    behaviour is exactly S1's.
    """

    def __init__(
        self,
        identity: WorkerIdentity,
        *,
        interval_s: float,
        client_factory: Callable[[str], ClusterClient] | None = None,
        command_sink: Callable[[list[Any]], None] | None = None,
        job_updates_provider: Callable[[], list[dict[str, Any]]] | None = None,
        transfer_updates_provider: Callable[[], list[dict[str, Any]]] | None = None,
        node_state_provider: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.identity = identity
        self.interval_s = max(0.1, interval_s)
        self.epoch = generate_epoch()
        self.seq = 0
        self.last_success_at: float | None = None
        self.last_error: str | None = None
        self._client_factory = client_factory or (lambda url: ClusterClient(url))
        self._command_sink = command_sink
        self._job_updates_provider = job_updates_provider
        # S5 D1b: a SIBLING channel to job_updates -- per-file transfer
        # progress/terminal state, not an ack (a TRANSFER_ROUND's ack only
        # confirms the round started; its outcome may arrive many
        # heartbeats later, here).
        self._transfer_updates_provider = transfer_updates_provider
        # S4 D1: advisory capacity/inventory attached to every heartbeat. The
        # provider is best-effort — returning None (or raising, caught below)
        # simply omits the field, exactly S1's heartbeat shape.
        self._node_state_provider = node_state_provider
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
        sent_seq = self.seq
        payload: dict[str, Any] = {"seq": sent_seq, "epoch": self.epoch}
        if self._job_updates_provider is not None:
            payload["job_updates"] = self._job_updates_provider()
        if self._transfer_updates_provider is not None:
            payload["transfer_updates"] = self._transfer_updates_provider()
        if self._node_state_provider is not None:
            try:
                node_state = self._node_state_provider()
            except Exception as exc:  # noqa: BLE001 - advisory, never blocks the beat
                node_state = None
                logger.debug("cluster: node_state provider failed: %s", exc)
            if node_state is not None:
                payload["node_state"] = node_state
        # CL2-11: the head address was pinned into WorkerIdentity at join and
        # is never changed by a command, so a heartbeat only ever reaches the
        # join-resolved head.
        client = self._client_factory(self.identity.head_url)
        try:
            reply = await client.post_json(
                HEARTBEAT_PATH,
                token=self.identity.secret,
                payload=payload,
            )
        except ClusterClientError as exc:
            self.last_error = str(exc)
            logger.warning("Heartbeat to %s failed: %s", self.identity.head_url, exc)
            return None
        self.last_error = None
        self.last_success_at = time.time()
        self._handle_commands(reply, sent_seq)
        return reply

    def _handle_commands(self, reply: dict[str, Any], sent_seq: int) -> None:
        """Act on a commands-bearing response, or discard it (CL2-05/CL2-06)."""
        commands = reply.get("commands")
        if not commands:
            return
        if self._command_sink is None:
            # S1 behaviour: a node with no command executor ignores commands.
            return
        # CL2-06: the response must echo the exact heartbeat we just sent, or it
        # is a replayed/reordered response and cannot be trusted.
        if reply.get("command_epoch") != self.epoch or reply.get("command_seq") != (
            sent_seq
        ):
            logger.warning(
                "cluster: discarding commands with mismatched epoch/seq echo"
            )
            return
        # CL2-05: a commands-bearing response must carry a valid HMAC derived
        # from this worker's own secret. Absent or wrong -> discard + log.
        digest = digest_secret(self.identity.secret)
        signature = str(reply.get("command_sig") or "")
        if not verify_command_response(
            digest, commands, epoch=self.epoch, seq=sent_seq, signature=signature
        ):
            logger.warning(
                "cluster: discarding commands with absent/invalid response "
                "signature (fp=%s)",
                fingerprint_key(signature),
            )
            return
        self._command_sink(commands)

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
