# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.cluster.hostfile."""

import json

import pytest

from omlx.cluster.hostfile import (
    build,
    jaccl_env,
    ring_addresses,
    ring_env,
    scrubbed_parent_env,
    write_ring_hostfile,
)

_DISTRIBUTED_VARS = [
    "MLX_RANK",
    "MLX_HOSTFILE",
    "MLX_JACCL_COORDINATOR",
    "MLX_IBV_DEVICES",
    "MLX_JACCL_RING",
    "JACCL_RANK",
    "JACCL_COORDINATOR",
    "JACCL_IBV_DEVICES",
    "JACCL_RING",
]


class TestRingAddresses:
    def test_assigns_one_link_per_rank_incrementing_port(self):
        assert ring_addresses(["10.0.0.1", "10.0.0.2"]) == [
            ["10.0.0.1:41100"],
            ["10.0.0.2:41101"],
        ]

    def test_colocated_ranks_share_ip_different_ports(self):
        # This is what makes a whole cluster testable on a single box.
        assert ring_addresses(["127.0.0.1", "127.0.0.1"]) == [
            ["127.0.0.1:41100"],
            ["127.0.0.1:41101"],
        ]

    def test_custom_base_port(self):
        assert ring_addresses(["10.0.0.1"], base_port=50000) == [["10.0.0.1:50000"]]


class TestWriteRingHostfile:
    def test_round_trips_valid_json(self, tmp_path):
        addresses = [["10.0.0.1:41100"], ["10.0.0.2:41101"]]
        path = write_ring_hostfile(tmp_path / "hostfile.json", addresses)
        assert json.loads(path.read_text()) == addresses

    def test_creates_parent_directories(self, tmp_path):
        path = write_ring_hostfile(
            tmp_path / "nested" / "dir" / "hostfile.json", [["10.0.0.1:41100"]]
        )
        assert path.exists()
        assert json.loads(path.read_text()) == [["10.0.0.1:41100"]]


class TestBuild:
    def test_ring_without_hostfile_raises(self):
        with pytest.raises(ValueError, match="hostfile"):
            build(backend="ring", rank=0, world_size=2)

    def test_jaccl_without_coordinator_raises(self):
        with pytest.raises(ValueError, match="coordinator"):
            build(backend="jaccl", rank=0, world_size=2, ibv_devices="/tmp/ibv.json")

    def test_jaccl_without_ibv_devices_raises(self):
        with pytest.raises(ValueError, match="coordinator"):
            build(
                backend="jaccl",
                rank=0,
                world_size=2,
                coordinator="10.0.0.1:41200",
            )

    def test_jaccl_ring_without_coordinator_raises(self):
        with pytest.raises(ValueError, match="coordinator"):
            build(
                backend="jaccl-ring",
                rank=0,
                world_size=2,
                ibv_devices="/tmp/ibv.json",
            )

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unsupported backend"):
            build(backend="mpi", rank=0, world_size=2)

    def test_ring_build_succeeds(self):
        launch = build(
            backend="ring", rank=1, world_size=2, hostfile="/tmp/hostfile.json"
        )
        assert launch.rank == 1
        assert launch.world_size == 2
        assert launch.backend == "ring"
        assert launch.env["MLX_HOSTFILE"] == "/tmp/hostfile.json"

    def test_jaccl_build_succeeds(self):
        launch = build(
            backend="jaccl",
            rank=0,
            world_size=2,
            coordinator="10.0.0.1:41200",
            ibv_devices="/tmp/ibv.json",
        )
        assert launch.backend == "jaccl"
        assert "MLX_JACCL_RING" not in launch.env

    def test_jaccl_ring_build_succeeds(self):
        launch = build(
            backend="jaccl-ring",
            rank=0,
            world_size=3,
            coordinator="10.0.0.1:41200",
            ibv_devices="/tmp/ibv.json",
        )
        assert launch.env["MLX_JACCL_RING"] == "1"


class TestJacclEnv:
    def test_ring_true_sets_flag(self):
        env = jaccl_env(0, "10.0.0.1:41200", "/tmp/ibv.json", ring=True)
        assert env["MLX_JACCL_RING"] == "1"

    def test_ring_false_omits_flag(self):
        env = jaccl_env(0, "10.0.0.1:41200", "/tmp/ibv.json", ring=False)
        assert "MLX_JACCL_RING" not in env

    def test_ring_defaults_to_false(self):
        env = jaccl_env(0, "10.0.0.1:41200", "/tmp/ibv.json")
        assert "MLX_JACCL_RING" not in env


class TestEnvAlwaysSetsFastSynch:
    def test_ring_env(self):
        assert ring_env(0, "/tmp/hostfile.json")["MLX_METAL_FAST_SYNCH"] == "1"

    def test_jaccl_env_without_ring(self):
        env = jaccl_env(0, "10.0.0.1:41200", "/tmp/ibv.json")
        assert env["MLX_METAL_FAST_SYNCH"] == "1"

    def test_jaccl_env_with_ring(self):
        env = jaccl_env(0, "10.0.0.1:41200", "/tmp/ibv.json", ring=True)
        assert env["MLX_METAL_FAST_SYNCH"] == "1"


class TestScrubbedParentEnv:
    def test_drops_client_facing_settings(self, monkeypatch):
        monkeypatch.setenv("OMLX_API_KEY", "secret")
        monkeypatch.setenv("OMLX_BASE_URL", "http://parent:8888")
        env = scrubbed_parent_env()
        assert "OMLX_API_KEY" not in env
        assert "OMLX_BASE_URL" not in env

    @pytest.mark.parametrize("name", _DISTRIBUTED_VARS)
    def test_drops_stale_distributed_var(self, monkeypatch, name):
        monkeypatch.setenv(name, "stale")
        assert name not in scrubbed_parent_env()

    def test_keeps_unrelated_vars(self, monkeypatch):
        monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")
        env = scrubbed_parent_env()
        assert env["SOME_UNRELATED_VAR"] == "keep-me"
