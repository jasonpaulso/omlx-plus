# SPDX-License-Identifier: Apache-2.0
"""Multi-machine ("cluster") serving for oMLX.

oMLX already runs as a daemon on every Mac in a workgroup. This package turns
that fleet into a single serving surface: peers discover each other over
Bonjour, the leader probes the Thunderbolt topology, and a model too large for
one machine is sharded across ranks via `mlx.distributed`.

The feature is off by default. With `cluster.enabled = false` (the default)
nothing in this package is imported at request time and single-node behaviour
is byte-identical to a build without it.

Module map:

- `preflight`  - per-node capability checks (RDMA, Thunderbolt, macOS version)
- `topology`   - Thunderbolt connectivity matrix, mesh/ring analysis, backend choice
- `hostfile`   - the on-wire launch contract (env vars + hostfile JSON)
"""

from __future__ import annotations

__all__ = ["preflight", "topology", "hostfile"]
