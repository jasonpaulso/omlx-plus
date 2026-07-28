# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.cluster.topology.

`_MACBOOK_PROFILER_JSON` and `_STUDIO_PROFILER_JSON` are real
`system_profiler -json SPThunderboltDataType` captures from an M5 Max
MacBook Pro cabled to an M3 Ultra Mac Studio over Thunderbolt 5. The
MacBook's `thunderboltusb4_bus_0` (domain 817DCFA4-B481-4100-80FC-91C7AA969F0B)
is the cabled bus; it is plugged into the Studio's
`thunderboltusb4_bus_1` (domain 28CA4C30-A263-4C55-B302-953416D0B209). The
Studio also has an "Envoy Ultra" drive on `bus_3` with no peer domain, which
must not be mistaken for a Mac-to-Mac link.
"""

from omlx.cluster.topology import (
    Bus,
    NodeReport,
    connectivity,
    find_ring,
    ibv_matrix,
    is_full_mesh,
    missing_cables,
    parse_buses,
    plan,
    unattributed_device,
)

_MACBOOK_PROFILER_JSON = """
{
  "SPThunderboltDataType" : [
    {
      "_name" : "thunderboltusb4_bus_2",
      "device_name_key" : "MacBook Pro",
      "domain_uuid_key" : "AD089FC0-E653-4FFE-B00A-DFF110B2F9A1",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "micro_version_key" : "0.0.0",
        "receptacle_id_key" : "3",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05ACBE5E06781982",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_name" : "thunderboltusb4_bus_1",
      "device_name_key" : "MacBook Pro",
      "domain_uuid_key" : "200E23D3-C921-44E0-95F7-55144AC251CC",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "receptacle_id_key" : "2",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05ACBE5E06781981",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_items" : [
        {
          "_name" : "Mac Studio",
          "device_id_key" : "0xA",
          "device_name_key" : "Mac15,14",
          "domain_uuid_key" : "28CA4C30-A263-4C55-B302-953416D0B209",
          "services_title" : [
            {
              "_name" : "service_ip",
              "protocol_id_key" : 1,
              "protocol_revision_key" : 1,
              "protocol_version_key" : 1,
              "service_uuid_key" : "066216C9-46DB-48B5-A472-768F7EA51471"
            }
          ],
          "vendor_id_key" : "0x0A27",
          "vendor_name_key" : "Apple Inc."
        }
      ],
      "_name" : "thunderboltusb4_bus_0",
      "device_name_key" : "MacBook Pro",
      "domain_uuid_key" : "817DCFA4-B481-4100-80FC-91C7AA969F0B",
      "receptacle_1_tag" : {
        "current_speed_key" : "80 Gb/s",
        "link_status_key" : "0x2",
        "micro_version_key" : "0.0.0",
        "receptacle_id_key" : "1",
        "receptacle_status_key" : "receptacle_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05ACBE5E06781980",
      "vendor_name_key" : "Apple Inc."
    }
  ]
}
"""

_STUDIO_PROFILER_JSON = """
{
  "SPThunderboltDataType" : [
    {
      "_name" : "thunderboltusb4_bus_5",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "1CF1083E-9B3C-4D21-8467-769814784F4D",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "receptacle_id_key" : "6",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9815",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_name" : "thunderboltusb4_bus_4",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "54AA5FD1-F8F9-438A-8825-7268E164186C",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "receptacle_id_key" : "5",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9814",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_items" : [
        {
          "_name" : "Envoy Ultra",
          "device_id_key" : "0xDE7B",
          "device_name_key" : "Envoy Ultra",
          "device_revision_key" : "0x6",
          "mode_key" : "usb_four_v2",
          "receptacle_upstream_ambiguous_tag" : {
            "current_speed_key" : "80 Gb/s",
            "link_status_key" : "0x2",
            "micro_version_key" : "1.3.0",
            "receptacle_status_key" : "receptacle_connected"
          },
          "route_string_key" : "1",
          "switch_uid_key" : "0x80878DFABA68F800",
          "switch_version_key" : "56.56",
          "vendor_id_key" : "0x1E91",
          "vendor_name_key" : "Other World Computing"
        }
      ],
      "_name" : "thunderboltusb4_bus_3",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "5F771049-70BD-4C30-A4AB-4E50482D82E0",
      "receptacle_1_tag" : {
        "current_speed_key" : "80 Gb/s",
        "link_status_key" : "0x2",
        "micro_version_key" : "0.0.0",
        "receptacle_id_key" : "4",
        "receptacle_status_key" : "receptacle_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9813",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_name" : "thunderboltusb4_bus_2",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "C9B32174-396A-4D89-9601-E8E83285C21A",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "receptacle_id_key" : "3",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9812",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_items" : [
        {
          "_name" : "MacBook Pro",
          "device_id_key" : "0xA",
          "device_name_key" : "Mac17,7",
          "domain_uuid_key" : "817DCFA4-B481-4100-80FC-91C7AA969F0B",
          "services_title" : [
            {
              "_name" : "service_ip",
              "protocol_id_key" : 1,
              "protocol_revision_key" : 1,
              "protocol_version_key" : 1,
              "service_uuid_key" : "F013163E-D041-44F6-B752-5236E1FEC7CD"
            },
            {
              "_name" : "unknown_xd_service",
              "protocol_id_key" : 64087,
              "protocol_revision_key" : 0,
              "protocol_version_key" : 1,
              "service_key_key" : "\\u02c7\\u02c7\\u02c7\\u02c7\\u02c7\\u02c7AD",
              "service_uuid_key" : "270E648C-7F7E-46CD-B5C6-D5063D93530F"
            }
          ],
          "vendor_id_key" : "0x0A27",
          "vendor_name_key" : "Apple Inc."
        }
      ],
      "_name" : "thunderboltusb4_bus_1",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "28CA4C30-A263-4C55-B302-953416D0B209",
      "receptacle_1_tag" : {
        "current_speed_key" : "80 Gb/s",
        "link_status_key" : "0x2",
        "micro_version_key" : "0.0.0",
        "receptacle_id_key" : "2",
        "receptacle_status_key" : "receptacle_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9811",
      "vendor_name_key" : "Apple Inc."
    },
    {
      "_name" : "thunderboltusb4_bus_0",
      "device_name_key" : "Mac Studio",
      "domain_uuid_key" : "57AE1A4C-B830-48D1-90DD-A3EDB8AC8DFC",
      "receptacle_1_tag" : {
        "current_speed_key" : "Up to 120 Gb/s",
        "link_status_key" : "0x100",
        "receptacle_id_key" : "1",
        "receptacle_status_key" : "receptacle_no_devices_connected"
      },
      "route_string_key" : "0",
      "switch_uid_key" : "0x05AC714E4ABE9810",
      "vendor_name_key" : "Apple Inc."
    }
  ]
}
"""

_MACBOOK_DOMAIN = "817DCFA4-B481-4100-80FC-91C7AA969F0B"
_STUDIO_DOMAIN = "28CA4C30-A263-4C55-B302-953416D0B209"


def _macbook_report(rdma_ready: bool = True) -> NodeReport:
    return NodeReport(
        node_id="macbook",
        buses=parse_buses(_MACBOOK_PROFILER_JSON),
        rdma_devices=["rdma_en1"],
        rdma_ready=rdma_ready,
    )


def _studio_report(rdma_ready: bool = True) -> NodeReport:
    return NodeReport(
        node_id="studio",
        buses=parse_buses(_STUDIO_PROFILER_JSON),
        rdma_devices=["rdma_en1"],
        rdma_ready=rdma_ready,
    )


class TestParseBuses:
    def test_macbook_bus_count(self):
        buses = parse_buses(_MACBOOK_PROFILER_JSON)
        assert len(buses) == 3

    def test_studio_bus_count(self):
        buses = parse_buses(_STUDIO_PROFILER_JSON)
        assert len(buses) == 6

    def test_macbook_has_exactly_one_peer(self):
        buses = parse_buses(_MACBOOK_PROFILER_JSON)
        peers = [b for b in buses if b.peer_domain_uuid]
        assert len(peers) == 1
        assert peers[0].domain_uuid == _MACBOOK_DOMAIN
        assert peers[0].peer_domain_uuid == _STUDIO_DOMAIN

    def test_studio_has_exactly_one_peer(self):
        # The Envoy Ultra drive on bus_3 must not count as a peer.
        buses = parse_buses(_STUDIO_PROFILER_JSON)
        peers = [b for b in buses if b.peer_domain_uuid]
        assert len(peers) == 1
        assert peers[0].domain_uuid == _STUDIO_DOMAIN
        assert peers[0].peer_domain_uuid == _MACBOOK_DOMAIN
        assert peers[0].peer_model != "Envoy Ultra"

    def test_unparseable_json_returns_empty(self):
        assert parse_buses("not json") == []

    def test_empty_payload_returns_empty(self):
        assert parse_buses('{"SPThunderboltDataType": []}') == []


class TestConnectivity:
    def test_exactly_one_edge_between_the_two_nodes(self):
        edges = connectivity([_macbook_report(), _studio_report()])
        assert edges == {frozenset({"macbook", "studio"})}

    def test_envoy_ultra_does_not_create_an_edge(self):
        # If the Envoy Ultra's bus were mistaken for a Mac peer it would
        # introduce a node named after the drive (or a bogus edge). Neither
        # happens - the graph is still just the one macbook<->studio edge.
        edges = connectivity([_macbook_report(), _studio_report()])
        for edge in edges:
            assert "Envoy Ultra" not in edge


class TestIsFullMeshAndMissingCables:
    _NODES = ["a", "b", "c"]
    _EDGES_MISSING_AC = {frozenset({"a", "b"}), frozenset({"b", "c"})}

    def test_missing_one_edge_is_not_full_mesh(self):
        assert is_full_mesh(self._NODES, self._EDGES_MISSING_AC) is False

    def test_missing_cables_names_the_gap(self):
        assert missing_cables(self._NODES, self._EDGES_MISSING_AC) == [("a", "c")]

    def test_complete_triangle_is_full_mesh(self):
        full = self._EDGES_MISSING_AC | {frozenset({"a", "c"})}
        assert is_full_mesh(self._NODES, full) is True
        assert missing_cables(self._NODES, full) == []

    def test_fewer_than_two_nodes_is_trivially_full_mesh(self):
        assert is_full_mesh(["a"], set()) is True
        assert is_full_mesh([], set()) is True


class TestFindRing:
    def test_fewer_than_three_nodes_returns_none(self):
        assert find_ring(["a", "b"], {frozenset({"a", "b"})}) is None
        assert find_ring(["a"], set()) is None
        assert find_ring([], set()) is None

    def test_finds_three_node_cycle(self):
        nodes = ["a", "b", "c"]
        edges = {
            frozenset({"a", "b"}),
            frozenset({"b", "c"}),
            frozenset({"c", "a"}),
        }
        ring = find_ring(nodes, edges)
        assert ring is not None
        assert set(ring) == set(nodes)
        assert len(ring) == 3
        for i in range(len(ring)):
            assert frozenset({ring[i], ring[(i + 1) % len(ring)]}) in edges

    def test_finds_four_node_cycle(self):
        nodes = ["a", "b", "c", "d"]
        edges = {
            frozenset({"a", "b"}),
            frozenset({"b", "c"}),
            frozenset({"c", "d"}),
            frozenset({"d", "a"}),
        }
        ring = find_ring(nodes, edges)
        assert ring is not None
        assert set(ring) == set(nodes)
        assert len(ring) == 4
        for i in range(len(ring)):
            assert frozenset({ring[i], ring[(i + 1) % len(ring)]}) in edges

    def test_path_graph_has_no_cycle(self):
        nodes = ["a", "b", "c", "d"]
        edges = {
            frozenset({"a", "b"}),
            frozenset({"b", "c"}),
            frozenset({"c", "d"}),
        }
        assert find_ring(nodes, edges) is None


class TestIbvMatrix:
    def _reports(self):
        # a <-> b cabled directly; c has no cable to anyone.
        a = NodeReport(
            node_id="a",
            buses=[Bus(name="bus_0", domain_uuid="A0", peer_domain_uuid="B0")],
            rdma_devices=["rdma_en1"],
        )
        b = NodeReport(
            node_id="b",
            buses=[Bus(name="bus_0", domain_uuid="B0", peer_domain_uuid="A0")],
            rdma_devices=["rdma_en1"],
        )
        c = NodeReport(
            node_id="c",
            buses=[Bus(name="bus_0", domain_uuid="C0", peer_domain_uuid=None)],
            rdma_devices=["rdma_en2"],
        )
        return [a, b, c]

    def test_diagonal_is_none(self):
        matrix = ibv_matrix(self._reports(), order=["a", "b", "c"])
        assert matrix[0][0] is None
        assert matrix[1][1] is None
        assert matrix[2][2] is None

    def test_cabled_pair_has_device_both_directions(self):
        matrix = ibv_matrix(self._reports(), order=["a", "b", "c"])
        assert matrix[0][1] == "rdma_en1"
        assert matrix[1][0] == "rdma_en1"

    def test_uncabled_pair_is_none(self):
        matrix = ibv_matrix(self._reports(), order=["a", "b", "c"])
        assert matrix[0][2] is None
        assert matrix[2][0] is None
        assert matrix[1][2] is None
        assert matrix[2][1] is None

    def test_single_active_device_wins_over_position(self):
        # The 2026-07-27 JACCL failure: three devices enumerate (one per
        # Thunderbolt port), the cable is in the port behind rdma_en2, and
        # the positional pick chose rdma_en1 - a PORT_DOWN device that fails
        # protection-domain allocation. Link state must beat position.
        a = NodeReport(
            node_id="a",
            buses=[Bus(name="bus_0", domain_uuid="A0", peer_domain_uuid="B0")],
            rdma_devices=["rdma_en1", "rdma_en2", "rdma_en7"],
            active_rdma_devices=["rdma_en2"],
        )
        b = NodeReport(
            node_id="b",
            buses=[Bus(name="bus_1", domain_uuid="B0", peer_domain_uuid="A0")],
            rdma_devices=["rdma_en2", "rdma_en3", "rdma_en4"],
            active_rdma_devices=["rdma_en3"],
        )
        matrix = ibv_matrix([a, b], order=["a", "b"])
        assert matrix[0][1] == "rdma_en2"
        assert matrix[1][0] == "rdma_en3"

    def test_multiple_active_devices_map_positionally_between_cables(self):
        # Two cables out of node a: the Nth cabled bus takes the Nth active
        # device. Down devices never enter the pool.
        a = NodeReport(
            node_id="a",
            buses=[
                Bus(name="bus_0", domain_uuid="A0", peer_domain_uuid="B0"),
                Bus(name="bus_1", domain_uuid="A1", peer_domain_uuid="C0"),
            ],
            rdma_devices=["rdma_en1", "rdma_en2", "rdma_en7"],
            active_rdma_devices=["rdma_en1", "rdma_en2"],
        )
        b = NodeReport(
            node_id="b",
            buses=[Bus(name="bus_0", domain_uuid="B0", peer_domain_uuid="A0")],
            rdma_devices=["rdma_en1"],
            active_rdma_devices=["rdma_en1"],
        )
        c = NodeReport(
            node_id="c",
            buses=[Bus(name="bus_0", domain_uuid="C0", peer_domain_uuid="A1")],
            rdma_devices=["rdma_en1"],
            active_rdma_devices=["rdma_en1"],
        )
        matrix = ibv_matrix([a, b, c], order=["a", "b", "c"])
        assert matrix[0][1] == "rdma_en1"
        assert matrix[0][2] == "rdma_en2"

    def test_active_device_unknown_to_devices_list_is_ignored(self):
        # A stale or inconsistent report must not smuggle in a device name
        # the node did not enumerate.
        a = NodeReport(
            node_id="a",
            buses=[Bus(name="bus_0", domain_uuid="A0", peer_domain_uuid="B0")],
            rdma_devices=["rdma_en1", "rdma_en2"],
            active_rdma_devices=["rdma_en9"],
        )
        b = NodeReport(
            node_id="b",
            buses=[Bus(name="bus_0", domain_uuid="B0", peer_domain_uuid="A0")],
            rdma_devices=["rdma_en1"],
        )
        matrix = ibv_matrix([a, b], order=["a", "b"])
        # Falls back to the positional map over all enumerated devices.
        assert matrix[0][1] == "rdma_en1"


class TestHubMediatedCabling:
    """Two Macs either side of a Thunderbolt hub, e.g. a Studio Display.

    Measured 2026-07-27 with a MacBook on the display's upstream port and a
    Studio on its downstream port: the Studio names the far Mac as its bus
    peer, the MacBook names nobody. RDMA is live on both ends regardless, and
    a hand-written symmetric matrix formed a jaccl world over that cabling.
    """

    def _reports(self):
        downstream = NodeReport(
            node_id="studio",
            buses=[Bus(name="bus_2", domain_uuid="S2", peer_domain_uuid="M2")],
            rdma_devices=["rdma_en4", "rdma_en5"],
            active_rdma_devices=["rdma_en4"],
            rdma_ready=True,
        )
        upstream = NodeReport(
            node_id="macbook",
            # Sees the hub, not the Mac behind it: no peer on any bus.
            buses=[Bus(name="bus_2", domain_uuid="M2", peer_domain_uuid=None)],
            rdma_devices=["rdma_en1", "rdma_en2", "rdma_en7"],
            active_rdma_devices=["rdma_en2"],
            rdma_ready=True,
        )
        return [downstream, upstream]

    def test_matrix_is_symmetric_across_the_hub(self):
        matrix = ibv_matrix(self._reports(), order=["studio", "macbook"])
        assert matrix == [[None, "rdma_en4"], ["rdma_en2", None]]

    def test_several_active_devices_stays_null(self):
        """No bus to index by means position cannot break the tie."""
        reports = self._reports()
        reports[1].active_rdma_devices = ["rdma_en1", "rdma_en2"]
        matrix = ibv_matrix(reports, order=["studio", "macbook"])
        assert matrix[1][0] is None

    def test_attributed_node_does_not_fall_back(self):
        """A node that names its peers keeps per-bus selection."""
        reports = self._reports()
        assert unattributed_device(reports[0]) == "rdma_en4"
        matrix = ibv_matrix(reports, order=["studio", "macbook"])
        # studio's entry came from its bus, not from the fallback
        assert matrix[0][1] == "rdma_en4"

    def test_no_active_device_stays_null(self):
        reports = self._reports()
        reports[1].active_rdma_devices = []
        matrix = ibv_matrix(reports, order=["studio", "macbook"])
        assert matrix[1][0] is None


class TestPlan:
    def test_ring_when_a_node_is_not_rdma_ready(self):
        result = plan([_macbook_report(rdma_ready=True), _studio_report(rdma_ready=False)])
        assert result.backend == "ring"
        assert "studio" in result.reason

    def test_ring_names_every_not_ready_node(self):
        result = plan(
            [_macbook_report(rdma_ready=False), _studio_report(rdma_ready=False)]
        )
        assert result.backend == "ring"
        assert "macbook" in result.reason
        assert "studio" in result.reason

    def test_jaccl_when_both_ready_and_full_mesh(self):
        result = plan([_macbook_report(rdma_ready=True), _studio_report(rdma_ready=True)])
        assert result.backend == "jaccl"
        assert result.order == ["macbook", "studio"]
        assert result.missing == []

    def test_single_node_is_ring(self):
        result = plan([_macbook_report(rdma_ready=True)])
        assert result.backend == "ring"
        assert result.reason == "single node"
