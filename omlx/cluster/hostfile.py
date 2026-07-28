# SPDX-License-Identifier: Apache-2.0
"""The launch contract: what a rank process needs in its environment, and the
data-plane address predicate that decides which addresses may enter it.

`mlx.launch` is only SSH plus environment variables plus exec. oMLX already
runs a daemon on every node, so it sets those variables itself and skips SSH,
the shared filesystem, and the identical-script-path assumption entirely.

Two different files are both called a "hostfile" in the wild; they are not the
same thing and mixing them up produces a JSON parse error at init:

- What `mlx.distributed_config` *writes* is a cluster description:
  `{"backend": ..., "hosts": [{"ssh": ..., "ips": [...], "rdma": [...]}]}`.
  oMLX does not use it — the control plane already knows the cluster.
- What `MLX_HOSTFILE` must *contain* for the ring backend is a flat list of
  per-rank link addresses: `[["127.0.0.1:41100"], ["127.0.0.1:41101"]]`.
  That is the format `write_ring_hostfile` produces.

Backend limits (verified on mlx 0.32.0, salvage `mlx_adapter.py`):

- Neither `ring` nor `jaccl` supports `Group.split()`.
- `jaccl` additionally has no `sum_scatter`.
- `ring` supports `recv` only from direct neighbours.
- A long-blocking `send`/`recv` must run on `stream=mx.cpu` (Metal's ~5 s
  command-buffer timeout otherwise kills the waiting rank).

This module is mlx-free on purpose: the unit gate exercises the address
predicate and env builder without touching MLX.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

# Recommended by Apple for distributed runs; harmless single-node.
METAL_FAST_SYNCH = "1"

# The ring backend opens one listening socket per rank on these ports.
DEFAULT_RING_BASE_PORT = 41100
# TCP bootstrap for jaccl: GID/queue-pair exchange, after which RDMA takes over.
DEFAULT_JACCL_COORDINATOR_PORT = 41200

# Names the mlx backend a rank must join, so it can ask for that one by name
# instead of `any`. `mx.distributed.init()` defaults to trying every backend
# and, worse, to returning a single-process group when none come up — which
# reads downstream as a formed cluster serving on one machine.
BACKEND_VAR = "OMLX_CLUSTER_BACKEND"


# -- CL2-01: local env built from an allowlist, never from the wire ----------

# The only parent-environment keys a rank subprocess inherits. An allowlist,
# not the salvage denylist: a network-arriving key the denylist never heard of
# (PYTHONPATH, PYTHONSTARTUP, DYLD_INSERT_LIBRARIES) is code execution in the
# rank process as the daemon user (CL2-01). The topology variables (MLX_RANK,
# MLX_HOSTFILE, ...) are computed locally and overlaid; they are deliberately
# absent from this set so a stale inherited value can never survive.
ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        # Hugging Face + XDG cache locations: a hub-cache model (D10 Qwen)
        # will not resolve without these.
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_TOKEN",
        "XDG_CACHE_HOME",
        # TLS trust roots, in case the loader must reach the hub.
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
)


def _allowlisted(base_env: dict[str, str]) -> dict[str, str]:
    """Keep only allowlisted keys (plus locale ``LC_*``) from a parent env."""
    return {
        key: value
        for key, value in base_env.items()
        if key in ENV_ALLOWLIST or key.startswith("LC_")
    }


# -- hostfile / env builders (salvaged, adapted to typed inputs) -------------


@dataclass(frozen=True)
class RankLaunch:
    """Everything a single rank process needs to join the collective."""

    rank: int
    world_size: int
    backend: str
    env: dict[str, str]


def write_ring_hostfile(path: str | Path, addresses: list[list[str]]) -> Path:
    """Write the ring backend's ``MLX_HOSTFILE``.

    ``addresses[rank]`` is the list of ``"ip:port"`` links that rank listens
    on. A single entry per rank is the common case.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(addresses))
    return path


def ring_addresses(
    ips: list[str], base_port: int = DEFAULT_RING_BASE_PORT
) -> list[list[str]]:
    """Assign one ``ip:port`` link per rank, in rank order.

    Ranks colocated on one machine share an IP and are separated by port, which
    is what makes a whole cluster testable on a single box.
    """
    return [[f"{ip}:{base_port + i}"] for i, ip in enumerate(ips)]


def write_ibv_devices(path: str | Path, matrix: list[list[str | None]]) -> Path:
    """Write the jaccl ``MLX_IBV_DEVICES`` matrix.

    ``matrix[i][j]`` is the RDMA device name on node ``i`` that reaches node
    ``j``, and is null on the diagonal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix))
    return path


def ring_env(rank: int, hostfile: str | Path) -> dict[str, str]:
    """Environment for one rank of a TCP ring run."""
    return {
        "MLX_RANK": str(rank),
        "MLX_HOSTFILE": str(hostfile),
        "MLX_METAL_FAST_SYNCH": METAL_FAST_SYNCH,
        BACKEND_VAR: "ring",
    }


def jaccl_env(
    rank: int,
    coordinator: str,
    ibv_devices: str | Path,
    *,
    ring: bool = False,
) -> dict[str, str]:
    """Environment for one rank of a JACCL run.

    ``coordinator`` is ``"<rank0_ip>:<port>"``.
    """
    env = {
        "MLX_RANK": str(rank),
        "MLX_JACCL_COORDINATOR": coordinator,
        "MLX_IBV_DEVICES": str(ibv_devices),
        "MLX_METAL_FAST_SYNCH": METAL_FAST_SYNCH,
        BACKEND_VAR: "jaccl",
    }
    if ring:
        env["MLX_JACCL_RING"] = "1"
    return env


def local_worker_env(
    base_env: dict[str, str],
    *,
    rank: int,
    backend: str,
    hostfile: str | Path | None = None,
    coordinator: str | None = None,
    ibv_devices: str | Path | None = None,
) -> dict[str, str]:
    """Build a rank's environment locally, from ``base_env`` and typed inputs.

    ``base_env`` (the daemon's own environment) is reduced to the allowlist,
    then the locally-computed topology variables are overlaid. Nothing here
    reads a command or the network: no environment ever crosses the wire
    (CL2-01), so the result is a function of ``base_env`` and these typed
    arguments only.
    """
    env = _allowlisted(base_env)
    if backend == "ring":
        if hostfile is None:
            raise ValueError("the ring backend requires a hostfile")
        env.update(ring_env(rank, hostfile))
    elif backend in ("jaccl", "jaccl-ring"):
        if coordinator is None or ibv_devices is None:
            raise ValueError(
                "the jaccl backends require a coordinator and an ibv device matrix"
            )
        env.update(
            jaccl_env(rank, coordinator, ibv_devices, ring=(backend == "jaccl-ring"))
        )
    else:
        raise ValueError(
            f"unsupported backend {backend!r}; expected ring, jaccl or jaccl-ring"
        )
    return env


# -- D7: data-plane address link-scope predicate -----------------------------


class LinkScopeError(ValueError):
    """A data-plane address was refused by the D7 predicate."""


@dataclass(frozen=True)
class LinkScopeVerdict:
    """Whether one address may enter a hostfile, and why."""

    allowed: bool
    reason: str


def _as_network(subnet: str | IPNetwork | None) -> IPNetwork | None:
    if subnet is None:
        return None
    if isinstance(subnet, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        return subnet
    return ipaddress.ip_network(subnet, strict=False)


def link_scope_verdict(
    address: str | IPAddress,
    *,
    data_plane_subnet: str | IPNetwork | None,
    allow_routable_data_plane: bool = False,
    allow_loopback: bool = False,
) -> LinkScopeVerdict:
    """Decide whether ``address`` may be used as a data-plane link (D7/CL-09).

    Default-deny, evaluated in this order so no override can reach a rule that
    precedes it:

    1. Link-local (``169.254.0.0/16``, IPv6 ``fe80::/10``), multicast, and
       unspecified addresses are rejected ALWAYS — no override reaches them
       (169.254 breaks ring connects, salvage pitfall 2).
    2. With ``data_plane_subnet`` unset, formation refuses, naming the setting
       (CL2-12: an unreviewed node must not degrade to trusting the head).
    3. A loopback address requires ``allow_loopback`` (single-host test mode),
       always — it never bypasses the subnet requirement.
    4. An address inside ``data_plane_subnet`` is accepted.
    5. Anything else — including any management-LAN address, private or not —
       is rejected unless ``allow_routable_data_plane`` (the operator override).
    """
    ip = ipaddress.ip_address(address) if isinstance(address, str) else address

    if ip.is_link_local:
        return LinkScopeVerdict(
            False, f"link-local address {ip} rejected always (breaks ring connects)"
        )
    if ip.is_multicast:
        return LinkScopeVerdict(False, f"multicast address {ip} is not a link address")
    if ip.is_unspecified:
        return LinkScopeVerdict(
            False, f"unspecified address {ip} is not a link address"
        )

    network = _as_network(data_plane_subnet)
    if network is None:
        return LinkScopeVerdict(
            False,
            "cluster.data_plane_subnet is unset; formation refuses "
            "(set it to the Thunderbolt link subnet, e.g. 10.0.2.0/24)",
        )

    if ip.is_loopback and not allow_loopback:
        return LinkScopeVerdict(
            False,
            f"loopback address {ip} rejected "
            "(set cluster.allow_loopback for single-host testing)",
        )

    if ip in network:
        return LinkScopeVerdict(True, f"{ip} is inside data_plane_subnet {network}")

    if allow_routable_data_plane:
        return LinkScopeVerdict(
            True, f"{ip} accepted by cluster.allow_routable_data_plane override"
        )

    return LinkScopeVerdict(
        False,
        f"{ip} is not inside data_plane_subnet {network}; "
        "set cluster.allow_routable_data_plane to override",
    )


def require_link_scope(
    address: str | IPAddress,
    *,
    data_plane_subnet: str | IPNetwork | None,
    allow_routable_data_plane: bool = False,
    allow_loopback: bool = False,
) -> IPAddress:
    """Return the parsed address if it passes the predicate, else raise.

    The one place a data-plane address becomes trusted enough to enter a
    hostfile. Callers feed it operator-configured or peer-supplied addresses;
    a refusal is a ``LinkScopeError`` naming the reason.
    """
    verdict = link_scope_verdict(
        address,
        data_plane_subnet=data_plane_subnet,
        allow_routable_data_plane=allow_routable_data_plane,
        allow_loopback=allow_loopback,
    )
    if not verdict.allowed:
        raise LinkScopeError(verdict.reason)
    return ipaddress.ip_address(address) if isinstance(address, str) else address
