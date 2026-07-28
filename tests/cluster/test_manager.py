# SPDX-License-Identifier: Apache-2.0
"""Tests for the cluster manager: membership, liveness, revocation."""

import json

import pytest

from omlx.cluster.client import ClusterClientError
from omlx.cluster.credentials import cluster_state_path, load_state, verify_secret
from omlx.cluster.manager import ClusterError, ClusterManager
from omlx.cluster.state import MemberLiveness
from omlx.cluster.versions import PackageVersion, VersionInfo, collect_versions

from .conftest import FakeClusterClient, make_settings, running_manager

SKEWED = VersionInfo(
    omlx="0.0.1", mlx="0.0.1", mlx_lm=PackageVersion("0.0.1", "0000000")
).to_dict()


def local_versions() -> dict:
    return collect_versions().to_dict()


async def admit(manager, *, host="10.0.0.9", port=8000, name="worker-a"):
    return await manager.join(
        peer_host=host, port=port, name=name, versions=local_versions()
    )


class TestStartupGate:
    async def test_cluster_role_without_api_key_refuses_to_start(self, tmp_path):
        """CL-01: no opt-out — a cluster role needs a configured API key."""
        for role in ("head", "worker"):
            settings = make_settings(tmp_path / role, role=role)
            settings.auth.api_key = None
            manager = ClusterManager(settings)
            with pytest.raises(RuntimeError, match="requires auth.api_key"):
                await manager.start()

    async def test_role_off_starts_nothing(self, tmp_path):
        settings = make_settings(tmp_path / "off", role="off")
        settings.auth.api_key = None
        manager = ClusterManager(settings)
        await manager.start()
        try:
            assert not cluster_state_path(settings.base_path).exists()
            # Acceptance: no new background work on a default install.
            assert manager._scrub_task is None
            assert manager._heartbeat is None
            assert not manager._queue.running
        finally:
            await manager.stop()


class TestJoin:
    async def test_join_admits_a_member_and_returns_the_secret_once(
        self, head_settings
    ):
        async with running_manager(head_settings) as manager:
            result = await admit(manager)

            assert result["member_secret"]
            assert result["heartbeat_interval_s"] == pytest.approx(0.05)
            member = manager.state.member(result["member_id"])
            assert member is not None
            assert str(member.address) == "10.0.0.9"
            assert verify_secret(
                result["member_secret"], manager.state.member_digests[member.id]
            )

    async def test_secret_is_never_persisted(self, head_settings):
        async with running_manager(head_settings) as manager:
            result = await admit(manager)
            raw = cluster_state_path(head_settings.base_path).read_text(
                encoding="utf-8"
            )
            assert result["member_secret"] not in raw

    async def test_membership_survives_a_restart_of_the_manager(self, head_settings):
        async with running_manager(head_settings) as manager:
            result = await admit(manager)
        async with running_manager(head_settings) as revived:
            assert revived.state.member(result["member_id"]) is not None

    async def test_version_skew_is_rejected(self, head_settings):
        async with running_manager(head_settings) as manager:
            with pytest.raises(ClusterError) as exc:
                await manager.join(
                    peer_host="10.0.0.9", port=8000, name="x", versions=SKEWED
                )
            assert exc.value.status_code == 409
            assert "0.0.1" in exc.value.detail
            assert manager.state.members == ()

    async def test_loopback_is_rejected_unless_allowed(self, tmp_path):
        settings = make_settings(tmp_path / "strict", role="head", allow_loopback=False)
        async with running_manager(settings) as manager:
            with pytest.raises(ClusterError) as exc:
                await admit(manager, host="127.0.0.1")
            assert exc.value.status_code == 400
            assert "Loopback" in exc.value.detail

    async def test_loopback_is_admitted_in_test_mode(self, head_settings):
        async with running_manager(head_settings) as manager:
            result = await admit(manager, host="127.0.0.1")
            assert str(manager.state.member(result["member_id"]).address) == "127.0.0.1"

    async def test_invalid_port_is_rejected(self, head_settings):
        async with running_manager(head_settings) as manager:
            with pytest.raises(ClusterError, match="port"):
                await admit(manager, port=0)

    async def test_member_name_is_sanitized(self, head_settings):
        async with running_manager(head_settings) as manager:
            result = await admit(manager, name="rack\n1\x00" + "x" * 200)
            name = manager.state.member(result["member_id"]).name
            assert "\n" not in name and "\x00" not in name
            assert len(name) <= 64


class TestHeartbeatAndScrub:
    async def test_heartbeat_records_liveness_without_touching_disk(
        self, head_settings
    ):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            before = cluster_state_path(head_settings.base_path).read_text("utf-8")

            # A non-hex epoch: member ids and digests are hex, so a short hex
            # epoch would collide with them by chance and make the
            # absence assertion meaningless.
            manager.record_heartbeat(member, seq=1, epoch="epoch-zulu")

            assert manager.liveness(member.id).status == "active"
            after = cluster_state_path(head_settings.base_path).read_text("utf-8")
            assert before == after
            assert "epoch-zulu" not in after

    async def test_replayed_sequence_within_an_epoch_is_rejected(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=5, epoch="e1")

            for replayed in (5, 4, 0):
                with pytest.raises(ClusterError) as exc:
                    manager.record_heartbeat(member, seq=replayed, epoch="e1")
                assert exc.value.status_code == 409
            assert manager.liveness(member.id).last_seq == 5

    async def test_new_epoch_resets_the_sequence(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=9, epoch="e1")

            manager.record_heartbeat(member, seq=1, epoch="e2")

            live = manager.liveness(member.id)
            assert (live.epoch, live.last_seq, live.status) == ("e2", 1, "active")

    async def test_empty_epoch_is_rejected(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            with pytest.raises(ClusterError, match="epoch"):
                manager.record_heartbeat(member, seq=1, epoch="")

    async def test_scrub_marks_silent_members_lost_without_revoking(
        self, head_settings
    ):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=1, epoch="e1")
            manager._liveness[member.id] = MemberLiveness(
                epoch="e1", last_seq=1, last_heartbeat_at=0.0, status="active"
            )

            expired = await manager.scrub()

            assert expired == [member.id]
            assert manager.liveness(member.id).status == "lost"
            # A timeout is a liveness statement, not a trust decision.
            assert member.id in manager.state.member_digests

    async def test_lost_member_revives_on_the_next_heartbeat(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=1, epoch="e1")
            manager._liveness[member.id] = MemberLiveness(
                epoch="e1", last_seq=1, last_heartbeat_at=0.0, status="lost"
            )

            manager.record_heartbeat(member, seq=2, epoch="e1")

            assert manager.liveness(member.id).status == "active"

    async def test_fresh_head_reports_persisted_members_as_lost(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=7, epoch="e1")

        async with running_manager(head_settings) as restarted:
            member = restarted.state.member(joined["member_id"])
            assert restarted.liveness(member.id) is None
            assert restarted.snapshot()["members"][0]["status"] == "lost"

            # A same-epoch worker continues; a restarted one resets.
            restarted.record_heartbeat(member, seq=8, epoch="e1")
            assert restarted.liveness(member.id).last_seq == 8
            restarted.record_heartbeat(member, seq=1, epoch="e2")
            assert restarted.liveness(member.id).epoch == "e2"


class TestRevocation:
    async def test_leave_revokes_only_the_caller(self, head_settings):
        async with running_manager(head_settings) as manager:
            first = await admit(manager, host="10.0.0.9")
            second = await admit(manager, host="10.0.0.10")
            member = manager.state.member(first["member_id"])

            await manager.member_leave(member)

            assert manager.state.member(first["member_id"]) is None
            assert first["member_id"] not in manager.state.member_digests
            assert manager.state.member(second["member_id"]) is not None
            assert verify_secret(
                second["member_secret"],
                manager.state.member_digests[second["member_id"]],
            )

    async def test_operator_removal_revokes_the_secret(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            await manager.remove_member(joined["member_id"])
            assert manager.state.member_digests == {}

    async def test_removal_persists(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            await manager.remove_member(joined["member_id"])
        assert load_state(cluster_state_path(head_settings.base_path)).members == ()

    async def test_removing_an_unknown_member_is_404(self, head_settings):
        async with running_manager(head_settings) as manager:
            with pytest.raises(ClusterError) as exc:
                await manager.remove_member("nope")
            assert exc.value.status_code == 404


class TestBootstrapToken:
    async def test_mint_returns_the_value_once_and_stores_a_digest(self, head_settings):
        async with running_manager(head_settings) as manager:
            result = await manager.mint_bootstrap_token()
            raw = cluster_state_path(head_settings.base_path).read_text("utf-8")
            assert result["token"] not in raw
            assert manager.state.bootstrap is not None

    async def test_renewal_replaces_the_previous_token(self, head_settings):
        async with running_manager(head_settings) as manager:
            first = await manager.mint_bootstrap_token()
            second = await manager.mint_bootstrap_token()
            assert first["token"] != second["token"]

    async def test_revoke_clears_the_record(self, head_settings):
        async with running_manager(head_settings) as manager:
            await manager.mint_bootstrap_token()
            assert (await manager.revoke_bootstrap_token())["revoked"] is True
            assert manager.state.bootstrap is None
            assert (await manager.revoke_bootstrap_token())["revoked"] is False


class TestWorkerSide:
    def _factory(self, replies):
        created: list[FakeClusterClient] = []

        def factory(url: str) -> FakeClusterClient:
            client = FakeClusterClient(url, replies)
            created.append(client)
            return client

        return factory, created

    async def test_local_join_persists_the_credential_and_starts_heartbeats(
        self, worker_settings
    ):
        factory, created = self._factory(
            {
                "/v1/cluster/join": {
                    "member_id": "m-1",
                    "member_secret": "s" * 64,
                    "heartbeat_interval_s": 0.05,
                },
                "/v1/cluster/heartbeat": {"status": "active"},
            }
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            result = await manager.local_join("http://10.0.0.1:8000", "join-token")

            assert result["member_id"] == "m-1"
            identity = manager.state.worker
            assert identity.secret == "s" * 64
            assert identity.head_url == "http://10.0.0.1:8000"
            assert manager._heartbeat.running
            # The join body carries the port, never an address (CL-10).
            method, path, token, payload = created[0].calls[0]
            assert (method, path, token) == ("POST", "/v1/cluster/join", "join-token")
            assert set(payload) == {"port", "name", "versions"}

    async def test_worker_state_file_holds_no_membership_table(self, worker_settings):
        factory, _ = self._factory(
            {"/v1/cluster/join": {"member_id": "m-1", "member_secret": "s" * 64}}
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            await manager.local_join("http://10.0.0.1:8000", "join-token")
            document = json.loads(
                cluster_state_path(worker_settings.base_path).read_text("utf-8")
            )
            assert document["members"] == []
            assert document["worker"]["member_id"] == "m-1"

    async def test_local_join_requires_a_token(self, worker_settings):
        factory, _ = self._factory({})
        async with running_manager(worker_settings, client_factory=factory) as manager:
            with pytest.raises(ClusterError, match="token"):
                await manager.local_join("http://10.0.0.1:8000", "")

    async def test_local_join_rejects_a_bad_head_url(self, worker_settings):
        factory, _ = self._factory({})
        async with running_manager(worker_settings, client_factory=factory) as manager:
            with pytest.raises(ClusterError) as exc:
                await manager.local_join("ftp://head", "token")
            assert exc.value.status_code == 400

    async def test_join_rejection_by_the_head_surfaces_its_status(
        self, worker_settings
    ):
        factory, _ = self._factory(
            {"/v1/cluster/join": ClusterClientError("rejected", status_code=401)}
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            with pytest.raises(ClusterError) as exc:
                await manager.local_join("http://10.0.0.1:8000", "bad")
            assert exc.value.status_code == 401
            assert manager.state.worker is None

    async def test_local_leave_notifies_the_head_and_drops_the_credential(
        self, worker_settings
    ):
        factory, created = self._factory(
            {
                "/v1/cluster/join": {"member_id": "m-1", "member_secret": "s" * 64},
                "/v1/cluster/heartbeat": {},
                "/v1/cluster/leave": {"removed": True},
            }
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            await manager.local_join("http://10.0.0.1:8000", "join-token")

            result = await manager.local_leave()

            assert result["head_notified"] is True
            assert manager.state.worker is None
            assert manager._heartbeat is None
            leave_calls = [
                c
                for client in created
                for c in client.calls
                if c[1] == "/v1/cluster/leave"
            ]
            assert leave_calls[0][2] == "s" * 64

    async def test_leave_drops_the_credential_even_if_the_head_is_unreachable(
        self, worker_settings
    ):
        factory, _ = self._factory(
            {
                "/v1/cluster/join": {"member_id": "m-1", "member_secret": "s" * 64},
                "/v1/cluster/heartbeat": {},
                "/v1/cluster/leave": ClusterClientError("no route"),
            }
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            await manager.local_join("http://10.0.0.1:8000", "join-token")
            result = await manager.local_leave()
            assert result["head_notified"] is False
            assert manager.state.worker is None

    async def test_leave_without_membership_is_a_client_error(self, worker_settings):
        factory, _ = self._factory({})
        async with running_manager(worker_settings, client_factory=factory) as manager:
            with pytest.raises(ClusterError) as exc:
                await manager.local_leave()
            assert exc.value.status_code == 400

    async def test_worker_resumes_heartbeats_after_a_restart(self, worker_settings):
        factory, _ = self._factory(
            {
                "/v1/cluster/join": {"member_id": "m-1", "member_secret": "s" * 64},
                "/v1/cluster/heartbeat": {},
            }
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            await manager.local_join("http://10.0.0.1:8000", "join-token")
            first_epoch = manager._heartbeat.epoch

        async with running_manager(worker_settings, client_factory=factory) as revived:
            assert revived.state.worker.member_id == "m-1"
            assert revived._heartbeat.running
            # A process restart is a new epoch.
            assert revived._heartbeat.epoch != first_epoch

    async def test_local_status_never_exposes_the_secret(self, worker_settings):
        factory, _ = self._factory(
            {
                "/v1/cluster/join": {"member_id": "m-1", "member_secret": "s" * 64},
                "/v1/cluster/heartbeat": {},
            }
        )
        async with running_manager(worker_settings, client_factory=factory) as manager:
            await manager.local_join("http://10.0.0.1:8000", "join-token")
            assert "s" * 64 not in json.dumps(manager.local_status())


class TestSnapshot:
    async def test_snapshot_carries_no_credential_material(self, head_settings):
        async with running_manager(head_settings) as manager:
            token = (await manager.mint_bootstrap_token())["token"]
            joined = await admit(manager)

            body = json.dumps(manager.snapshot())

            assert token not in body
            assert joined["member_secret"] not in body
            for digest in manager.state.member_digests.values():
                assert digest not in body
            assert manager.state.bootstrap.digest not in body

    async def test_snapshot_reports_counts_and_bootstrap_presence(self, head_settings):
        async with running_manager(head_settings) as manager:
            joined = await admit(manager)
            member = manager.state.member(joined["member_id"])
            manager.record_heartbeat(member, seq=1, epoch="e1")

            snapshot = manager.snapshot()

            assert snapshot["role"] == "head"
            assert (snapshot["member_count"], snapshot["active_count"]) == (1, 1)
            assert snapshot["bootstrap_token"] == {
                "configured": False,
                "expires_at": None,
            }
