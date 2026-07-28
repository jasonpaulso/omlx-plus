# SPDX-License-Identifier: Apache-2.0
"""Tests for the E10 version handshake."""

from omlx.cluster.versions import (
    PackageVersion,
    VersionInfo,
    collect_versions,
    compare_versions,
    vcs_commit_id,
)


def info(omlx="0.5.3", mlx="0.32.0", mlx_lm_version="0.31.3", commit="ab1806e"):
    return VersionInfo(
        omlx=omlx, mlx=mlx, mlx_lm=PackageVersion(mlx_lm_version, commit)
    )


class TestCompare:
    def test_identical_stacks_match(self):
        assert compare_versions(info(), info()) is None

    def test_omlx_mismatch_is_rejected(self):
        error = compare_versions(info(), info(omlx="0.5.4"))
        assert error is not None
        assert "0.5.3" in error and "0.5.4" in error

    def test_mlx_mismatch_is_rejected(self):
        assert compare_versions(info(), info(mlx="0.31.0")) is not None

    def test_mlx_lm_version_mismatch_is_rejected(self):
        assert compare_versions(info(), info(mlx_lm_version="0.31.2")) is not None

    def test_commit_mismatch_is_rejected_when_versions_agree(self):
        """The exact skew E10 exists to catch: same version, different commit."""
        error = compare_versions(info(), info(commit="deadbee"))
        assert error is not None
        assert "ab1806e" in error and "deadbee" in error

    def test_both_commits_absent_falls_back_to_version_compare(self):
        assert compare_versions(info(commit=None), info(commit=None)) is None

    def test_exactly_one_commit_absent_is_rejected(self):
        assert compare_versions(info(), info(commit=None)) is not None
        assert compare_versions(info(commit=None), info()) is not None

    def test_error_names_both_sides(self):
        error = compare_versions(info(), info(omlx="9.9.9", commit="f00"))
        assert "head has [" in error
        assert "joining node has [" in error


class TestSerialization:
    def test_round_trip(self):
        original = info()
        assert VersionInfo.from_dict(original.to_dict()) == original

    def test_missing_commit_round_trips_as_none(self):
        original = info(commit=None)
        restored = VersionInfo.from_dict(original.to_dict())
        assert restored.mlx_lm.commit_id is None
        assert restored == original

    def test_from_dict_tolerates_missing_keys(self):
        restored = VersionInfo.from_dict({})
        assert restored.omlx == ""
        assert restored.mlx_lm.commit_id is None

    def test_describe_marks_unknown_commit(self):
        assert info(commit=None).mlx_lm.describe().endswith("@unknown")


class TestCollect:
    def test_collect_reports_the_installed_stack(self):
        collected = collect_versions()
        assert collected.omlx
        assert collected.mlx
        assert collected.mlx_lm.version

    def test_mlx_lm_commit_comes_from_pep_610_metadata(self):
        """mlx-lm is a git pin, so its commit id must be discoverable."""
        assert vcs_commit_id("mlx-lm") is not None

    def test_unknown_distribution_has_no_commit(self):
        assert vcs_commit_id("definitely-not-installed-xyz") is None
