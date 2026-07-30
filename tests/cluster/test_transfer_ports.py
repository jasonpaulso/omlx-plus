# SPDX-License-Identifier: Apache-2.0
"""CL5-17: the derived transfer port range must never collide with the
formation ring range or the jaccl coordinator port.

``settings.py`` used to claim this was "exercised by the unit gate" with no
actual test anywhere -- this file is that test.
"""

from __future__ import annotations

import pytest

from omlx.settings import (
    TRANSFER_PORT_RANGE_OFFSET,
    ClusterSettings,
    assert_transfer_ports_non_overlapping,
    transfer_base_port,
)


def test_transfer_base_port_is_offset_from_the_ring_base():
    cluster = ClusterSettings(data_plane_base_port=41100)
    assert transfer_base_port(cluster) == 41100 + TRANSFER_PORT_RANGE_OFFSET


def test_disjoint_ranges_pass():
    cluster = ClusterSettings(data_plane_base_port=41100)
    assert_transfer_ports_non_overlapping(cluster)  # does not raise


def test_transfer_range_overlapping_the_ring_range_is_refused(monkeypatch):
    # `xfer_lo` is always `ring_hi + 1` when it is derived from the same
    # `TRANSFER_PORT_RANGE_OFFSET` the ring range itself uses -- disjoint by
    # construction, which is exactly what `test_disjoint_ranges_pass` pins.
    # This row exercises the overlap BRANCH itself, standing in for the
    # failure mode CL5-17 actually guards against: the offset and the ring
    # range drifting out of sync (e.g. a future edit bumps
    # `cluster.manager.MAX_WORLD_SIZE` without updating this module's
    # duplicated copy) -- simulated here by making `transfer_base_port`
    # return a port still inside the ring range.
    import omlx.settings as settings_mod

    cluster = ClusterSettings(data_plane_base_port=1000)
    monkeypatch.setattr(settings_mod, "transfer_base_port", lambda _c: 1000)
    with pytest.raises(ValueError, match="overlaps the formation ring range"):
        assert_transfer_ports_non_overlapping(cluster)


def test_transfer_range_overlapping_the_jaccl_coordinator_port_is_refused():
    from omlx.cluster.hostfile import DEFAULT_JACCL_COORDINATOR_PORT

    # Pick a base port so the derived transfer range straddles the jaccl
    # coordinator port.
    base = DEFAULT_JACCL_COORDINATOR_PORT - TRANSFER_PORT_RANGE_OFFSET
    cluster = ClusterSettings(data_plane_base_port=base)
    with pytest.raises(ValueError, match="jaccl coordinator port"):
        assert_transfer_ports_non_overlapping(cluster)


def test_transfer_range_out_of_valid_port_bounds_is_refused():
    cluster = ClusterSettings(data_plane_base_port=65535)
    with pytest.raises(ValueError, match="out of the valid port range"):
        assert_transfer_ports_non_overlapping(cluster)
