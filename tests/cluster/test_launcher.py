# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the launcher's spawn bound, pipe I/O classes, and sweep.

mlx-free: no rank process is spawned here (that is the integration test).
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import pytest

from omlx.cluster import launcher
from omlx.cluster.launcher import (
    CommandReader,
    ControlChannel,
    LocalCluster,
    ReplyReader,
    SpawnBoundError,
    _register_formation,
    _release_formation,
    sweep_orphaned_ranks,
)


@pytest.fixture(autouse=True)
def _reset_formation_slot():
    yield
    launcher._active_cluster = None


def _fake_cluster(alive: bool):
    return types.SimpleNamespace(any_alive=lambda: alive)


# -- CL2-09 spawn bound ------------------------------------------------------


def test_second_live_formation_is_refused():
    first = _fake_cluster(alive=True)
    _register_formation(first)
    with pytest.raises(SpawnBoundError):
        _register_formation(_fake_cluster(alive=True))


def test_formation_slot_frees_after_release():
    first = _fake_cluster(alive=True)
    _register_formation(first)
    _release_formation(first)
    # Slot is free again.
    _register_formation(_fake_cluster(alive=True))


def test_dead_formation_does_not_block():
    _register_formation(_fake_cluster(alive=False))
    # The prior formation is not alive, so a new one may claim the slot.
    _register_formation(_fake_cluster(alive=True))


# -- pipe I/O classes --------------------------------------------------------


def test_command_reader_blocking_then_drain():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b'{"op":"ping"}\n{"op":"generate"}\n')
        reader = CommandReader(read_fd)
        assert reader.readline() == '{"op":"ping"}'
        assert reader.drain_lines() == ['{"op":"generate"}']
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_command_reader_eof_returns_empty():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        assert CommandReader(read_fd).readline() == ""
    finally:
        os.close(read_fd)


def test_reply_reader_line_timeout_and_eof():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b'{"ok":true}\n')
        reader = ReplyReader(read_fd)
        assert reader.readline(1.0) == '{"ok":true}'
        # Nothing more to read: an idle timeout returns None.
        assert reader.readline(0.05) is None
        os.close(write_fd)  # closing the write end signals EOF
        assert reader.readline(1.0) == ""
    finally:
        os.close(read_fd)


def test_control_channel_parses_abort_and_closes_to_abort_all():
    read_fd, write_fd = os.pipe()
    channel = ControlChannel(read_fd)
    os.write(write_fd, b'{"op":"abort","request_id":"r1"}\n')
    events = channel.take_events()
    assert {"op": "abort", "request_id": "r1"} in events
    # A closed pipe means the daemon is gone: abort everything.
    os.close(write_fd)
    assert {"op": "abort"} in channel.take_events()
    os.close(read_fd)


def test_control_channel_none_fd_is_inert():
    assert ControlChannel(None).take_events() == []


# -- sweep -------------------------------------------------------------------


def test_sweep_returns_int_with_no_orphans():
    # No orphaned rank processes on the box: a no-op that returns a count.
    assert isinstance(sweep_orphaned_ranks(), int)


# -- backend spawn wiring (subprocess boundary, no rank process spawned) ------
#
# There is deliberately NO jaccl integration test: forming a real jaccl group
# needs RDMA hardware and a second node, which this dev machine has not. The
# real jaccl acceptance is the P3 live-rig run, not pytest. These tests mock at
# the subprocess boundary and assert the constructed argv + env + ibv matrix,
# proving the launch contract without spawning anything (the ring launch is
# exercised for real by the cluster+integration test_rank_group.py).


class _FakeStdin:
    def write(self, *_args):
        return None

    def flush(self):
        return None


class _FakePopen:
    """Records argv+env; never execs. Alive until killed."""

    def __init__(self, argv, env=None, **_kwargs):
        self.argv = argv
        self.env = env
        self.pid = 4242
        self.stdin = _FakeStdin()
        self.stdout = None
        self._rc: int | None = None

    def poll(self):
        return self._rc

    def kill(self):
        self._rc = -9

    def wait(self, timeout=None):
        self._rc = 0
        return 0


def _spawn_and_capture(monkeypatch, cluster, **start_kwargs):
    captured: list[tuple[list[str], dict[str, str]]] = []

    def fake_popen(argv, env=None, **kwargs):
        captured.append((argv, dict(env or {})))
        return _FakePopen(argv, env, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    # Skip the psutil readiness probe between rank 0 and its peers.
    monkeypatch.setattr(LocalCluster, "wait_until_ready", lambda self, **_k: True)
    try:
        cluster.start(**start_kwargs)
    finally:
        # Tear down without touching a real process; close the rank-0 control fd.
        for entry in cluster.ranks:
            if entry.control_w is not None:
                os.close(entry.control_w)
                entry.control_w = None
        cluster.ranks.clear()
        launcher._active_cluster = None
    return captured


def test_jaccl_spawn_builds_coordinator_env_and_matrix(monkeypatch, tmp_path):
    matrix = [[None, "rdma_en2"], ["rdma_en4", None]]
    cluster = LocalCluster(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        world_size=2,
        backend="jaccl",
        enforce_spawn_bound=False,
    )
    captured = _spawn_and_capture(
        monkeypatch,
        cluster,
        ranks=[0, 1],
        ips=["127.0.0.1", "127.0.0.1"],
        ibv_devices=matrix,
        data_plane_subnet="127.0.0.0/8",
        allow_loopback=True,
    )
    assert len(captured) == 2
    (argv0, env0), (argv1, env1) = captured
    # Fixed argv: interpreter -m worker --model <id> --seed N. No shell, no path.
    assert argv0[1:4] == ["-m", "omlx.cluster.rank_worker", "--model"]
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in argv0
    # rank 0 (the coordinator) and rank 1 share one coordinator and one matrix.
    assert env0["MLX_RANK"] == "0"
    assert env1["MLX_RANK"] == "1"
    assert env0["OMLX_CLUSTER_BACKEND"] == "jaccl"
    assert env0["MLX_JACCL_COORDINATOR"] == "127.0.0.1:41200"
    assert env1["MLX_JACCL_COORDINATOR"] == "127.0.0.1:41200"
    assert env0["MLX_IBV_DEVICES"] == env1["MLX_IBV_DEVICES"]
    assert "MLX_HOSTFILE" not in env0
    # The written matrix is exactly the S0-recipe matrix.
    assert json.loads(pathlib.Path(env0["MLX_IBV_DEVICES"]).read_text()) == matrix


def test_ring_spawn_still_builds_hostfile_env(monkeypatch, tmp_path):
    # Regression: the refactored _spawn_one leaves the ring launch contract
    # unchanged — a hostfile, no coordinator, no ibv matrix.
    cluster = LocalCluster(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        world_size=2,
        backend="ring",
        enforce_spawn_bound=False,
    )
    captured = _spawn_and_capture(
        monkeypatch,
        cluster,
        ranks=[0, 1],
        ips=["127.0.0.1", "127.0.0.1"],
        data_plane_subnet="127.0.0.0/8",
        allow_loopback=True,
    )
    env0 = captured[0][1]
    assert env0["OMLX_CLUSTER_BACKEND"] == "ring"
    assert env0["MLX_HOSTFILE"]
    assert "MLX_JACCL_COORDINATOR" not in env0
    assert "MLX_IBV_DEVICES" not in env0
