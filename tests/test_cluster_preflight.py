# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.cluster.preflight."""

from omlx.cluster import preflight
from omlx.cluster.preflight import Preflight

# Two header lines only - what `ibv_devices` prints when RDMA is disarmed.
_IBV_HEADER_ONLY = (
    "    device                 node GUID\n"
    "    ------              ----------------\n"
)

_IBV_WITH_DEVICES = (
    "    device                 node GUID\n"
    "    ------              ----------------\n"
    "    rdma_en1            801978065ebeac05\n"
    "    rdma_en7            801978065ebeac0f\n"
)

_BRIDGE_WITH_MEMBERS = (
    "bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500\n"
    "\tmember: en2 flags=3<LEARNING,DISCOVER>\n"
    "\tmember: en3 flags=3<LEARNING,DISCOVER>\n"
)


class TestRdmaStatus:
    def test_enabled(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, "enabled\n"))
        assert preflight.rdma_status() is True

    def test_disabled(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, "disabled\n"))
        assert preflight.rdma_status() is False

    def test_recovery_os_message_reads_as_disabled(self, monkeypatch):
        # rdma_ctl exits 0 even when run outside Recovery OS, so the exit
        # status can't be trusted - only the message body matters.
        monkeypatch.setattr(
            preflight,
            "_run",
            lambda *a: (
                0,
                "rdma_ctl: This tool needs to be executed from Recovery OS.\n",
            ),
        )
        assert preflight.rdma_status() is False

    def test_nonzero_exit_is_disabled(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (127, "rdma_ctl: not found"))
        assert preflight.rdma_status() is False

    def test_empty_output_is_disabled(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, ""))
        assert preflight.rdma_status() is False


class TestIbvDevices:
    def test_disarmed_machine_returns_empty(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, _IBV_HEADER_ONLY))
        assert preflight.ibv_devices() == []

    def test_lists_devices_sorted(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, _IBV_WITH_DEVICES))
        assert preflight.ibv_devices() == ["rdma_en1", "rdma_en7"]

    def test_command_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (127, "not found"))
        assert preflight.ibv_devices() == []


class TestThunderboltBridgeMembers:
    def test_extracts_member_interfaces(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, _BRIDGE_WITH_MEMBERS))
        assert preflight.thunderbolt_bridge_members() == ["en2", "en3"]

    def test_no_such_interface_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            preflight,
            "_run",
            lambda *a: (1, "ifconfig: interface bridge0 does not exist"),
        )
        assert preflight.thunderbolt_bridge_members() == []


class TestThunderboltMaxGbps:
    def test_parses_fixed_speed(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, "Speed: 80 Gb/s\n"))
        assert preflight.thunderbolt_max_gbps() == 80

    def test_parses_up_to_speed(self, monkeypatch):
        monkeypatch.setattr(
            preflight, "_run", lambda *a: (0, "Speed: Up to 120 Gb/s\n")
        )
        assert preflight.thunderbolt_max_gbps() == 120

    def test_takes_max_across_ports(self, monkeypatch):
        out = "Speed: 80 Gb/s\nSpeed: Up to 120 Gb/s\nSpeed: 40 Gb/s\n"
        monkeypatch.setattr(preflight, "_run", lambda *a: (0, out))
        assert preflight.thunderbolt_max_gbps() == 120

    def test_command_failure_returns_zero(self, monkeypatch):
        monkeypatch.setattr(preflight, "_run", lambda *a: (1, ""))
        assert preflight.thunderbolt_max_gbps() == 0


def _ready_preflight(**overrides) -> Preflight:
    fields = dict(
        macos=(26, 2),
        chip="Apple M5 Max",
        rdma_enabled=True,
        rdma_devices=["rdma_en1"],
        bridged_interfaces=[],
        tb_max_gbps=80,
    )
    fields.update(overrides)
    return Preflight(**fields)


class TestRdmaReady:
    """Every one of these conditions must hold, independently."""

    def test_all_conditions_met(self):
        assert _ready_preflight().rdma_ready is True

    def test_macos_too_old(self):
        assert _ready_preflight(macos=(26, 1)).rdma_ready is False

    def test_rdma_not_enabled(self):
        assert _ready_preflight(rdma_enabled=False).rdma_ready is False

    def test_no_rdma_devices(self):
        assert _ready_preflight(rdma_devices=[]).rdma_ready is False

    def test_bridged_interfaces_present(self):
        assert _ready_preflight(bridged_interfaces=["en2"]).rdma_ready is False

    def test_thunderbolt_below_tb5(self):
        assert _ready_preflight(tb_max_gbps=40).rdma_ready is False


class TestBestBackend:
    def test_ready_and_mesh_is_jaccl(self):
        assert _ready_preflight().best_backend(mesh_complete=True) == "jaccl"

    def test_ready_and_not_mesh_is_jaccl_ring(self):
        assert _ready_preflight().best_backend(mesh_complete=False) == "jaccl-ring"

    def test_not_ready_is_ring_regardless_of_mesh(self):
        not_ready = _ready_preflight(rdma_enabled=False)
        assert not_ready.best_backend(mesh_complete=True) == "ring"
        assert not_ready.best_backend(mesh_complete=False) == "ring"


class TestRunHelper:
    def test_missing_binary_returns_127(self):
        rc, out = preflight._run("omlx-test-definitely-not-a-real-binary")
        assert rc == 127
        assert "not found" in out
