"""Shared distributed-session helper for the S0 spike scripts.

Adapted (not copied verbatim) from discovery/omlx/omlx/cluster/mlx_adapter.py
(salvaged utility, not the branch's architecture). One session per process
lifetime -- callers must never init/teardown repeatedly (jaccl PD leak).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

BACKEND_VAR = "OMLX_CLUSTER_BACKEND"


@dataclass(frozen=True)
class WorldInfo:
    rank: int
    size: int

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


class DistributedSession:
    def __init__(self) -> None:
        backend = os.environ.get(BACKEND_VAR) or "any"
        self._group = mx.distributed.init(strict=True, backend=backend)
        self.world = WorldInfo(rank=self._group.rank(), size=self._group.size())
        self.barrier()

    @property
    def group(self):
        return self._group

    def barrier(self) -> None:
        mx.eval(mx.distributed.all_sum(mx.ones(10), group=self._group, stream=mx.cpu))

    def broadcast(self, obj: Any) -> Any:
        if self.world.size == 1:
            return obj
        if self.world.is_leader:
            payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            size = mx.array([len(payload)], dtype=mx.int64)
        else:
            payload = b""
            size = mx.array([0], dtype=mx.int64)

        size = mx.distributed.all_sum(size, group=self._group, stream=mx.cpu)
        mx.eval(size)
        length = int(size.item())
        if length == 0:
            return None

        if self.world.is_leader:
            buf = mx.array(list(payload), dtype=mx.uint32)
        else:
            buf = mx.zeros(length, dtype=mx.uint32)
        buf = mx.distributed.all_sum(buf, group=self._group, stream=mx.cpu)
        mx.eval(buf)
        return pickle.loads(bytes(bytearray(buf.tolist())))

    def all_sum_latency(self, nbytes: int) -> mx.array:
        """One timed all_sum over a real array of nbytes (uint8), stream=cpu."""
        n = max(1, nbytes)
        arr = mx.ones(n, dtype=mx.uint8) if self.world.is_leader else mx.zeros(n, dtype=mx.uint8)
        return mx.distributed.all_sum(arr, group=self._group, stream=mx.cpu)
