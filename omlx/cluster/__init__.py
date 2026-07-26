# SPDX-License-Identifier: Apache-2.0
"""Multi-machine ("cluster") serving for oMLX.

oMLX already runs as a daemon on every Mac in a workgroup. This package turns
that fleet into a single serving surface: peers discover each other over
Bonjour, the leader probes the Thunderbolt topology, and a model too large for
one machine is sharded across ranks via `mlx.distributed`.

The feature is off by default (`cluster.enabled = false`). The daemon's only
contact with this package is `bootstrap.install()` at startup, which returns
immediately when the setting is off; nothing else here is imported, no socket
is opened and no process is spawned. Single-node behaviour is unchanged.

Module map:

- `bootstrap`  - the only seam the server calls: install() / shutdown()
- `preflight`  - per-node capability checks (RDMA, Thunderbolt, macOS version)
- `topology`   - Thunderbolt connectivity matrix, mesh/ring analysis, backend choice
- `discovery`  - Bonjour peer advertisement and browsing
- `hostfile`   - the on-wire launch contract (env vars + hostfile JSON)
- `mlx_adapter`- every mlx.distributed call, isolated behind one interface
- `worker`     - the rank process and its lockstep loop
- `launcher`   - spawning and supervising rank processes from the daemon
"""

from __future__ import annotations

__all__ = [
    "bootstrap",
    "discovery",
    "hostfile",
    "launcher",
    "mlx_adapter",
    "preflight",
    "topology",
    "worker",
]
