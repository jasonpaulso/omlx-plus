# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed cluster state model."""

import ipaddress
import json
import time

import pytest

from omlx.cluster.credentials import cluster_state_path, load_state, save_state
from omlx.cluster.state import (
    BootstrapTokenRecord,
    ClusterState,
    FileManifestEntry,
    Member,
    MemberLiveness,
    TransferJob,
    WorkerIdentity,
    parse_member_address,
)
from omlx.cluster.versions import PackageVersion, VersionInfo

VERSIONS = VersionInfo(
    omlx="0.5.3", mlx="0.32.0", mlx_lm=PackageVersion("0.31.3", "ab1806e")
)


def _member(**overrides):
    defaults = dict(
        id="m1",
        address=ipaddress.ip_address("10.0.0.7"),
        port=8000,
        name="studio",
        versions=VERSIONS,
        joined_at=1700000000.0,
        peer_cert_fingerprint=None,
    )
    defaults.update(overrides)
    return Member(**defaults)


class TestMember:
    def test_round_trip_through_cluster_json(self, tmp_path):
        """A member survives a save/load cycle through cluster.json intact."""
        member = _member(peer_cert_fingerprint="ab:cd:ef")
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState(members=(member,)))

        loaded = load_state(path)

        assert loaded.members == (member,)
        assert loaded.members[0].peer_cert_fingerprint == "ab:cd:ef"
        assert isinstance(loaded.members[0].address, ipaddress.IPv4Address)

    def test_cert_fingerprint_defaults_to_none(self):
        """The CL-05 TLS seam field is present and unused in v1."""
        assert _member().peer_cert_fingerprint is None

    def test_ipv6_endpoint_is_bracketed(self):
        member = _member(address=ipaddress.ip_address("fd00::1"))
        assert member.endpoint == "[fd00::1]:8000"

    def test_ipv4_endpoint(self):
        assert _member().endpoint == "10.0.0.7:8000"


class TestPersistedShape:
    def test_liveness_fields_are_absent_from_the_persisted_file(self, tmp_path):
        """Heartbeat state is runtime-only and must never reach disk."""
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState(members=(_member(),)))

        raw = path.read_text(encoding="utf-8")
        document = json.loads(raw)

        for field in ("epoch", "last_seq", "last_heartbeat_at", "status"):
            assert field not in raw
            assert field not in document["members"][0]

    def test_full_document_round_trip(self, tmp_path):
        state = ClusterState(
            members=(_member(),),
            member_digests={"m1": "d" * 64},
            bootstrap=BootstrapTokenRecord(
                digest="a" * 64, created_at=1.0, expires_at=2.0
            ),
            worker=WorkerIdentity(
                member_id="m1", secret="s" * 64, head_url="http://h:8000", joined_at=3.0
            ),
            jobs=(
                TransferJob(
                    id="j1",
                    kind="transfer",
                    status="queued",
                    created_at=4.0,
                    manifest=(FileManifestEntry("a/b.safetensors", 12, "c" * 64),),
                ),
            ),
        )
        path = cluster_state_path(tmp_path)
        save_state(path, state)

        assert load_state(path) == state

    def test_missing_file_loads_as_empty_state(self, tmp_path):
        loaded = load_state(cluster_state_path(tmp_path))
        assert loaded == ClusterState()
        assert loaded.members == ()


class TestLookups:
    def test_member_lookup(self):
        state = ClusterState(members=(_member(), _member(id="m2")))
        assert state.member("m2").id == "m2"
        assert state.member("nope") is None

    def test_liveness_is_a_separate_record(self):
        live = MemberLiveness(
            epoch="e1", last_seq=3, last_heartbeat_at=time.time(), status="active"
        )
        assert live.to_dict()["status"] == "active"
        assert not hasattr(ClusterState(), "liveness")


class TestParseMemberAddress:
    def test_accepts_routable_address(self):
        assert parse_member_address("192.168.1.5") == ipaddress.ip_address(
            "192.168.1.5"
        )

    def test_rejects_loopback_by_default(self):
        with pytest.raises(ValueError, match="Loopback"):
            parse_member_address("127.0.0.1")

    def test_allows_loopback_in_test_mode(self):
        assert parse_member_address("127.0.0.1", allow_loopback=True)

    def test_rejects_unspecified(self):
        with pytest.raises(ValueError, match="Unspecified"):
            parse_member_address("0.0.0.0", allow_loopback=True)

    def test_rejects_multicast(self):
        with pytest.raises(ValueError, match="Multicast"):
            parse_member_address("224.0.0.1", allow_loopback=True)

    def test_rejects_non_address_string(self):
        with pytest.raises(ValueError):
            parse_member_address("10.0.0.1 evil-hostfile-line")
