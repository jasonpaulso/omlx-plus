# SPDX-License-Identifier: Apache-2.0
"""Tests for cluster credential handling and the cluster.json store."""

import json
import stat

import pytest

from omlx.cluster.credentials import (
    bootstrap_token_matches,
    cluster_state_path,
    digest_secret,
    generate_epoch,
    generate_secret,
    load_state,
    mint_bootstrap_token,
    save_state,
    verify_secret,
)
from omlx.cluster.state import ClusterState, WorkerIdentity


class TestSecrets:
    def test_generated_secret_is_256_bit_hex(self):
        secret = generate_secret()
        assert len(secret) == 64
        int(secret, 16)

    def test_secrets_are_unique(self):
        assert generate_secret() != generate_secret()

    def test_epoch_is_short_hex(self):
        epoch = generate_epoch()
        assert len(epoch) == 16
        int(epoch, 16)

    def test_verify_secret_accepts_matching_secret(self):
        secret = generate_secret()
        assert verify_secret(secret, digest_secret(secret))

    def test_verify_secret_rejects_wrong_secret(self):
        assert not verify_secret(generate_secret(), digest_secret(generate_secret()))

    def test_verify_secret_rejects_empty_inputs(self):
        assert not verify_secret("", digest_secret("x"))
        assert not verify_secret("x", "")

    def test_verify_secret_tolerates_non_ascii(self):
        """A junk credential must yield 401, never a 500 from the compare."""
        assert not verify_secret("café🔥", digest_secret(generate_secret()))


class TestBootstrapToken:
    def test_mint_returns_value_and_record_with_digest_only(self):
        token, record = mint_bootstrap_token(900.0, now=1000.0)
        assert record.digest == digest_secret(token)
        assert token not in json.dumps(record.to_dict())
        assert record.expires_at == 1900.0

    def test_valid_token_matches(self):
        token, record = mint_bootstrap_token(900.0, now=1000.0)
        assert bootstrap_token_matches(record, token, now=1100.0)

    def test_expired_token_is_rejected(self):
        token, record = mint_bootstrap_token(900.0, now=1000.0)
        assert not bootstrap_token_matches(record, token, now=1900.1)

    def test_wrong_token_is_rejected(self):
        _token, record = mint_bootstrap_token(900.0, now=1000.0)
        assert not bootstrap_token_matches(record, generate_secret(), now=1100.0)

    def test_absent_record_is_rejected(self):
        assert not bootstrap_token_matches(None, generate_secret())

    def test_renewal_replaces_the_digest(self):
        first, first_record = mint_bootstrap_token(900.0, now=1000.0)
        _second, second_record = mint_bootstrap_token(900.0, now=1001.0)
        assert first_record.digest != second_record.digest
        assert not bootstrap_token_matches(second_record, first, now=1002.0)


class TestStore:
    def test_state_file_is_0600(self, tmp_path):
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState())
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, oct(mode)

    def test_rewrite_keeps_0600(self, tmp_path):
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState())
        save_state(path, ClusterState(member_digests={"m1": "d" * 64}))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert load_state(path).member_digests == {"m1": "d" * 64}

    def test_write_leaves_no_temp_files_behind(self, tmp_path):
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState())
        assert [p.name for p in tmp_path.iterdir()] == ["cluster.json"]

    def test_missing_parent_directory_is_created(self, tmp_path):
        path = cluster_state_path(tmp_path / "nested" / "deeper")
        save_state(path, ClusterState())
        assert path.exists()

    def test_head_stores_digests_not_secrets(self, tmp_path):
        secret = generate_secret()
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState(member_digests={"m1": digest_secret(secret)}))
        assert secret not in path.read_text(encoding="utf-8")

    def test_worker_identity_round_trips(self, tmp_path):
        identity = WorkerIdentity(
            member_id="m1",
            secret=generate_secret(),
            head_url="http://head:8000",
            joined_at=5.0,
        )
        path = cluster_state_path(tmp_path)
        save_state(path, ClusterState(worker=identity))
        assert load_state(path).worker == identity

    def test_corrupt_file_is_not_silently_reset(self, tmp_path):
        path = cluster_state_path(tmp_path)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_state(path)
