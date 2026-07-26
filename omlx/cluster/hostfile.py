# SPDX-License-Identifier: Apache-2.0
"""The launch contract: what a rank process needs in its environment.

`mlx.launch` is only SSH plus environment variables plus exec. oMLX already
runs a daemon on every node, so it can set those variables itself and skip SSH,
the shared filesystem, and the identical-script-path assumption entirely. This
module is the whole contract, so there is exactly one place to fix when the
upstream launcher moves.

Everything here was read out of the shipped binaries
(`libmlx.dylib`, `libjaccl.dylib` in mlx 0.32.0) and confirmed by launching
ranks, rather than taken from documentation.

Two different files are both called a "hostfile" in the wild; they are not the
same thing and mixing them up produces a JSON parse error at init:

- What `mlx.distributed_config` *writes* is a cluster description:
  `{"backend": ..., "hosts": [{"ssh": ..., "ips": [...], "rdma": [...]}]}`.
  oMLX does not use it - the control plane already knows the cluster.
- What `MLX_HOSTFILE` must *contain* for the ring backend is a flat list of
  per-rank link addresses: `[["127.0.0.1:41100"], ["127.0.0.1:41101"]]`.
  That is the format `write_ring_hostfile` produces.

Backend limits, all verified rather than assumed:

- **Neither `ring` nor `jaccl` supports `Group.split()`.** Both raise
  "Group split not supported". So `sharded_load`'s `pipeline_group` and
  `tensor_group` can never both be real subgroups: a run is tensor-parallel
  across the whole world *or* pipeline-parallel across the whole world, never
  a 2D mesh of the two. Only MPI could do that, and MPI is not a supported
  transport here.
- `jaccl` additionally has no `sum_scatter`.
- `ring` supports `recv` only from direct neighbours, which is fine for
  pipeline parallel (adjacent ranks) and rules out arbitrary rank-to-rank.
- A long-blocking `send`/`recv` must run on `stream=mx.cpu`. On the default
  GPU stream the ~5 s Metal command-buffer timeout kills the waiting rank.
  Confirmed: a 7 s idle `recv` on the cpu stream completes; the same wait on
  the GPU stream does not.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Recommended by Apple for distributed runs; harmless single-node.
METAL_FAST_SYNCH = "1"

# The ring backend opens one listening socket per rank on these ports.
DEFAULT_RING_BASE_PORT = 41100
# TCP bootstrap for jaccl: GID/queue-pair exchange, after which RDMA takes over.
DEFAULT_JACCL_COORDINATOR_PORT = 41200


@dataclass(frozen=True)
class RankLaunch:
    """Everything a single rank process needs to join the collective."""

    rank: int
    world_size: int
    backend: str
    env: dict[str, str]


def write_ring_hostfile(path: str | Path, addresses: list[list[str]]) -> Path:
    """Write the ring backend's `MLX_HOSTFILE`.

    `addresses[rank]` is the list of `"ip:port"` links that rank listens on.
    A single entry per rank is the common case; multiple entries describe a
    rank reachable over several physical links.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(addresses))
    return path


def ring_addresses(
    ips: list[str], base_port: int = DEFAULT_RING_BASE_PORT
) -> list[list[str]]:
    """Assign one `ip:port` link per rank, in rank order.

    Ranks colocated on one machine share an IP and are separated by port, which
    is what makes a whole cluster testable on a single box.
    """
    return [[f"{ip}:{base_port + i}"] for i, ip in enumerate(ips)]


def write_ibv_devices(path: str | Path, matrix: list[list[str | None]]) -> Path:
    """Write the jaccl `MLX_IBV_DEVICES` matrix.

    `matrix[i][j]` is the RDMA device name on node `i` that reaches node `j`,
    and is null on the diagonal. RDMA over Thunderbolt cannot route, so a null
    off the diagonal means those two nodes have no cable between them and the
    mesh backend cannot be used.
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
    }


def jaccl_env(
    rank: int,
    coordinator: str,
    ibv_devices: str | Path,
    *,
    ring: bool = False,
) -> dict[str, str]:
    """Environment for one rank of a JACCL run.

    `coordinator` is `"<rank0_ip>:<port>"`. The binaries accept both the
    `MLX_`- and `JACCL_`-prefixed spellings of every one of these; we emit the
    `MLX_` forms.
    """
    env = {
        "MLX_RANK": str(rank),
        "MLX_JACCL_COORDINATOR": coordinator,
        "MLX_IBV_DEVICES": str(ibv_devices),
        "MLX_METAL_FAST_SYNCH": METAL_FAST_SYNCH,
    }
    if ring:
        # Ring cabling rather than a full mesh.
        env["MLX_JACCL_RING"] = "1"
    return env


def build(
    *,
    backend: str,
    rank: int,
    world_size: int,
    hostfile: str | Path | None = None,
    coordinator: str | None = None,
    ibv_devices: str | Path | None = None,
) -> RankLaunch:
    """Assemble the launch spec for one rank.

    Raises `ValueError` rather than letting a missing variable surface later as
    an opaque init failure inside the C++ backend.
    """
    if backend == "ring":
        if hostfile is None:
            raise ValueError("the ring backend requires a hostfile")
        env = ring_env(rank, hostfile)
    elif backend in ("jaccl", "jaccl-ring"):
        if coordinator is None or ibv_devices is None:
            raise ValueError(
                "the jaccl backends require a coordinator and an ibv device matrix"
            )
        env = jaccl_env(
            rank, coordinator, ibv_devices, ring=(backend == "jaccl-ring")
        )
    else:
        raise ValueError(
            f"unsupported backend {backend!r}; expected ring, jaccl or jaccl-ring"
        )
    return RankLaunch(rank=rank, world_size=world_size, backend=backend, env=env)


def scrubbed_parent_env() -> dict[str, str]:
    """A copy of this process's environment safe to hand a rank subprocess.

    The daemon's own client-facing settings must not leak into a worker: a
    worker that inherits `OMLX_BASE_URL` can end up pointed back at the server
    that spawned it. Any stale distributed variables are dropped too, so a
    worker never half-inherits a previous run's topology.
    """
    drop = {
        "OMLX_BASE_URL",
        "OMLX_API_KEY",
        "MLX_RANK",
        "MLX_HOSTFILE",
        "MLX_JACCL_COORDINATOR",
        "MLX_IBV_DEVICES",
        "MLX_JACCL_RING",
        "JACCL_RANK",
        "JACCL_COORDINATOR",
        "JACCL_IBV_DEVICES",
        "JACCL_RING",
    }
    return {k: v for k, v in os.environ.items() if k not in drop}
