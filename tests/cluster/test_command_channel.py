# SPDX-License-Identifier: Apache-2.0
"""The D2 head->worker command channel: HMAC (CL2-05), replay echo (CL2-06),
attribution (CL2-07), and S1 heartbeat compatibility.
"""

from __future__ import annotations

from omlx.cluster.credentials import (
    digest_secret,
    sign_command_response,
    verify_command_response,
)
from omlx.cluster.heartbeat import HeartbeatSender
from omlx.cluster.protocol import PROTOCOL_VERSION
from omlx.cluster.state import WorkerIdentity

from .conftest import FakeClusterClient, running_manager

SECRET = "s" * 64
IDENTITY = WorkerIdentity(
    member_id="m-1", secret=SECRET, head_url="http://10.0.0.1:8000", joined_at=1.0
)
COMMANDS = [
    {"kind": "sweep", "schema_version": PROTOCOL_VERSION, "job_id": "j", "step": 1}
]


def _worker(reply_fn, *, with_sink=True):
    sunk: list = []

    def factory(url):
        return FakeClusterClient(url, {"/v1/cluster/heartbeat": reply_fn})

    sender = HeartbeatSender(
        IDENTITY,
        interval_s=0.01,
        client_factory=factory,
        command_sink=(sunk.append if with_sink else None),
        job_updates_provider=(lambda: []) if with_sink else None,
    )
    return sender, sunk


def _signed_reply(payload):
    epoch, seq = payload["epoch"], payload["seq"]
    digest = digest_secret(SECRET)
    return {
        "status": "active",
        "commands": COMMANDS,
        "command_epoch": epoch,
        "command_seq": seq,
        "command_sig": sign_command_response(digest, COMMANDS, epoch=epoch, seq=seq),
    }


# -- CL2-05 HMAC round-trip ---------------------------------------------------


def test_derived_key_signature_round_trips():
    digest = digest_secret(SECRET)
    sig = sign_command_response(digest, COMMANDS, epoch="ep", seq=3)
    assert verify_command_response(digest, COMMANDS, epoch="ep", seq=3, signature=sig)


def test_signature_binds_epoch_and_seq():
    digest = digest_secret(SECRET)
    sig = sign_command_response(digest, COMMANDS, epoch="ep", seq=3)
    assert not verify_command_response(
        digest, COMMANDS, epoch="ep", seq=4, signature=sig
    )
    assert not verify_command_response(
        digest, COMMANDS, epoch="other", seq=3, signature=sig
    )


def test_absent_or_wrong_signature_fails():
    digest = digest_secret(SECRET)
    assert not verify_command_response(
        digest, COMMANDS, epoch="ep", seq=3, signature=""
    )
    assert not verify_command_response(
        digest, COMMANDS, epoch="ep", seq=3, signature="deadbeef"
    )


def test_wrong_secret_cannot_forge():
    good = sign_command_response(digest_secret(SECRET), COMMANDS, epoch="ep", seq=3)
    assert not verify_command_response(
        digest_secret("z" * 64), COMMANDS, epoch="ep", seq=3, signature=good
    )


# -- worker accepts / discards command-bearing responses ----------------------


async def test_worker_accepts_correctly_signed_response():
    sender, sunk = _worker(_signed_reply)
    await sender.send_once()
    assert sunk == [COMMANDS]


async def test_worker_discards_unsigned_commands():
    def unsigned(payload):
        return {"status": "active", "commands": COMMANDS}

    sender, sunk = _worker(unsigned)
    await sender.send_once()
    assert sunk == []


async def test_worker_discards_wrong_signature():
    def bad_sig(payload):
        return {
            "status": "active",
            "commands": COMMANDS,
            "command_epoch": payload["epoch"],
            "command_seq": payload["seq"],
            "command_sig": "0" * 64,
        }

    sender, sunk = _worker(bad_sig)
    await sender.send_once()
    assert sunk == []


async def test_worker_discards_mismatched_epoch_echo():
    def wrong_epoch(payload):
        digest = digest_secret(SECRET)
        return {
            "status": "active",
            "commands": COMMANDS,
            "command_epoch": "not-mine",
            "command_seq": payload["seq"],
            "command_sig": sign_command_response(
                digest, COMMANDS, epoch="not-mine", seq=payload["seq"]
            ),
        }

    sender, sunk = _worker(wrong_epoch)
    await sender.send_once()
    assert sunk == []


# -- S1 compatibility ---------------------------------------------------------


async def test_s1_worker_without_sink_ignores_commands():
    sender, sunk = _worker(_signed_reply, with_sink=False)
    reply = await sender.send_once()
    assert reply is not None and reply.get("commands") == COMMANDS
    assert sunk == []


async def test_s1_worker_omits_job_updates_field():
    def echo(payload):
        return {"status": "active", "_seen": payload}

    sender, _ = _worker(echo, with_sink=False)
    reply = await sender.send_once()
    assert "job_updates" not in reply["_seen"]


async def test_command_worker_includes_job_updates_field():
    seen: dict = {}

    def echo(payload):
        seen.update(payload)
        return {"status": "active"}

    def factory(url):
        return FakeClusterClient(url, {"/v1/cluster/heartbeat": echo})

    sender = HeartbeatSender(
        IDENTITY,
        interval_s=0.01,
        client_factory=factory,
        command_sink=lambda c: None,
        job_updates_provider=lambda: [{"job_id": "j", "step": 1, "status": "spawned"}],
    )
    await sender.send_once()
    assert seen["job_updates"] == [{"job_id": "j", "step": 1, "status": "spawned"}]


# -- head side: signing + CL2-07 attribution ----------------------------------


class _FakeFormation:
    def __init__(self, commands):
        self._commands = commands
        self.updates: list = []

    def commands_for(self, member_id):
        return list(self._commands)

    def record_job_updates(self, member, updates):
        self.updates.append((member.id, updates))

    def snapshot(self):
        return {}

    async def stop(self):
        return None


async def test_head_signs_commands_and_attributes_updates(head_settings):
    async with running_manager(head_settings) as manager:
        reply = await manager.join(
            peer_host="10.1.2.3",
            port=40404,
            name="w",
            versions=manager.versions.to_dict(),
        )
        member = manager.state.member(reply["member_id"])
        assert member is not None
        digest = digest_secret(reply["member_secret"])
        manager._formation = _FakeFormation(COMMANDS)

        hb = manager.record_heartbeat(
            member,
            seq=5,
            epoch="ep",
            job_updates=[{"job_id": "j", "step": 1, "status": "x", "member": "SPOOF"}],
        )

        assert hb["commands"] == COMMANDS
        assert hb["command_epoch"] == "ep" and hb["command_seq"] == 5
        assert verify_command_response(
            digest, COMMANDS, epoch="ep", seq=5, signature=hb["command_sig"]
        )
        # CL2-07: the head attributed the update to the AUTHENTICATED member,
        # never the body-supplied "SPOOF".
        assert len(manager._formation.updates) == 1
        attributed_member_id, _updates = manager._formation.updates[0]
        assert attributed_member_id == member.id


async def test_head_without_pending_commands_returns_s1_reply(head_settings):
    async with running_manager(head_settings) as manager:
        reply = await manager.join(
            peer_host="10.1.2.3",
            port=40404,
            name="w",
            versions=manager.versions.to_dict(),
        )
        member = manager.state.member(reply["member_id"])
        assert member is not None
        # A formation with nothing queued for this member: the reply is exactly
        # the S1 shape (no commands, no signature).
        manager._formation = _FakeFormation([])
        hb = manager.record_heartbeat(member, seq=1, epoch="ep")
        assert "commands" not in hb
        assert "command_sig" not in hb
        assert hb["status"] == "active"
