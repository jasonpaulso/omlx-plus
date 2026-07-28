# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the launcher's spawn bound, pipe I/O classes, and sweep.

mlx-free: no rank process is spawned here (that is the integration test).
"""

from __future__ import annotations

import os
import types

import pytest

from omlx.cluster import launcher
from omlx.cluster.launcher import (
    CommandReader,
    ControlChannel,
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
