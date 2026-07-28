# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hostfile builders, the D7 link-scope predicate, and the
CL2-01 local env build."""

from __future__ import annotations

import json

import pytest

from omlx.cluster.hostfile import (
    LinkScopeError,
    link_scope_verdict,
    local_worker_env,
    require_link_scope,
    ring_addresses,
    write_ring_hostfile,
)

SUBNET = "10.0.2.0/24"
LOOPBACK_SUBNET = "127.0.0.0/8"


# -- D7 refusal matrix -------------------------------------------------------


@pytest.mark.parametrize(
    "address,subnet,routable,loopback,allowed",
    [
        # 169.254/16 is rejected ALWAYS — even with every override on.
        ("169.254.72.202", SUBNET, True, True, False),
        ("169.254.72.202", SUBNET, False, False, False),
        # A management-LAN address (RFC1918 but not the link subnet) is rejected
        # with the override off, accepted with it on.
        ("192.168.5.99", SUBNET, False, False, False),
        ("192.168.5.99", SUBNET, True, False, True),
        # An in-subnet address is accepted with no override.
        ("10.0.2.2", SUBNET, False, False, True),
        # Loopback needs allow_loopback, even inside the loopback subnet.
        ("127.0.0.1", LOOPBACK_SUBNET, False, False, False),
        ("127.0.0.1", LOOPBACK_SUBNET, False, True, True),
        # Unset subnet refuses formation regardless of overrides.
        ("10.0.2.2", None, True, True, False),
        # Multicast is never a link address.
        ("224.0.0.1", SUBNET, True, True, False),
    ],
)
def test_link_scope_refusal_matrix(address, subnet, routable, loopback, allowed):
    verdict = link_scope_verdict(
        address,
        data_plane_subnet=subnet,
        allow_routable_data_plane=routable,
        allow_loopback=loopback,
    )
    assert verdict.allowed is allowed
    assert verdict.reason  # every verdict carries a reason


def test_unset_subnet_reason_names_the_setting():
    verdict = link_scope_verdict(
        "10.0.2.2", data_plane_subnet=None, allow_routable_data_plane=True
    )
    assert not verdict.allowed
    assert "data_plane_subnet" in verdict.reason


def test_require_link_scope_raises_on_refusal():
    with pytest.raises(LinkScopeError):
        require_link_scope(
            "169.254.1.1", data_plane_subnet=SUBNET, allow_routable_data_plane=True
        )


def test_require_link_scope_returns_parsed_address():
    address = require_link_scope("10.0.2.2", data_plane_subnet=SUBNET)
    assert str(address) == "10.0.2.2"


# -- ring hostfile builders --------------------------------------------------


def test_ring_addresses_assigns_one_port_per_rank():
    assert ring_addresses(["10.0.2.1", "10.0.2.2"], 41100) == [
        ["10.0.2.1:41100"],
        ["10.0.2.2:41101"],
    ]


def test_write_ring_hostfile(tmp_path):
    path = write_ring_hostfile(tmp_path / "hosts.json", [["127.0.0.1:41100"]])
    assert json.loads(path.read_text()) == [["127.0.0.1:41100"]]


# -- CL2-01: local env built from an allowlist, never from the wire ----------


def test_local_worker_env_keeps_allowlisted_drops_rest():
    base = {
        "PATH": "/usr/bin",
        "HF_HOME": "/hf",
        "HOME": "/Users/x",
        "PYTHONPATH": "/evil",
        "PYTHONSTARTUP": "/evil.py",
        "DYLD_INSERT_LIBRARIES": "/evil.dylib",
        "OMLX_API_KEY": "secret",
        "OMLX_BASE_URL": "http://attacker",
    }
    env = local_worker_env(base, rank=1, backend="ring", hostfile="/tmp/h.json")

    assert env["PATH"] == "/usr/bin"
    assert env["HF_HOME"] == "/hf"
    assert env["HOME"] == "/Users/x"
    # The RCE-carrying keys never survive.
    for banned in (
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "DYLD_INSERT_LIBRARIES",
        "OMLX_API_KEY",
        "OMLX_BASE_URL",
    ):
        assert banned not in env
    # Topology variables are computed locally.
    assert env["MLX_RANK"] == "1"
    assert env["MLX_HOSTFILE"] == "/tmp/h.json"
    assert env["OMLX_CLUSTER_BACKEND"] == "ring"


def test_local_worker_env_identical_whatever_env_shaped_keys_passed():
    # CL2-01 acceptance: the spawned rank's env is identical whatever
    # env-shaped keys arrive — injected keys and a stale MLX_RANK are dropped.
    clean = {"PATH": "/usr/bin", "HF_HOME": "/hf"}
    polluted = {
        **clean,
        "PYTHONPATH": "/evil",
        "DYLD_INSERT_LIBRARIES": "/evil.dylib",
        "OMLX_API_KEY": "secret",
        "MLX_RANK": "99",
        "OMLX_CLUSTER_BACKEND": "jaccl",
    }
    from_clean = local_worker_env(clean, rank=1, backend="ring", hostfile="h")
    from_polluted = local_worker_env(polluted, rank=1, backend="ring", hostfile="h")
    assert from_clean == from_polluted


def test_local_worker_env_ring_requires_hostfile():
    with pytest.raises(ValueError):
        local_worker_env({}, rank=0, backend="ring")


def test_local_worker_env_rejects_unknown_backend():
    with pytest.raises(ValueError):
        local_worker_env({}, rank=0, backend="nccl")
