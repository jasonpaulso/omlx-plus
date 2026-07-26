# SPDX-License-Identifier: Apache-2.0
"""The single place oMLX touches mlx's distributed API.

mlx-lm's distributed serving surface is Apple's current headline feature and is
moving fast, so `sharded_load` and the server-loop internals are treated as
unstable API. Everything that could change upstream is funnelled through this
module: when mlx-lm moves, this file is the blast radius.

Everything asserted here was measured on mlx 0.32.0 / mlx-lm 0.31.3, not read
from documentation.

Hard constraints
----------------
**No group splitting.** Both usable backends raise "Group split not supported"
from `Group.split()`. Verified on `ring` with four ranks, and present as a
literal error string in `libmlx.dylib` for `jaccl`. Consequently a run is
tensor-parallel across the entire world *or* pipeline-parallel across the
entire world - `sharded_load`'s two group parameters can never both be real
subgroups, and 2D TP x PP is impossible on this transport. Only MPI supports
splitting, and MPI is not a transport oMLX offers.

**One distributed session per process.** Repeated init/teardown cycles exhaust
kernel protection domains and recovery is a reboot, so this module deliberately
offers no teardown-and-reinit path. Changing model or world size means killing
the worker process and starting a new one. That is why workers are cheap
subprocesses rather than something clever in-process.

**Long waits belong on the CPU stream.** A `send`/`recv` that blocks longer
than the ~5 s Metal command-buffer timeout kills the waiting rank when issued
on the default GPU stream. Confirmed: a 7 s idle `recv` completes on
`stream=mx.cpu` and does not on the GPU stream. Every collective here that can
block for an unbounded time is pinned to the CPU stream.

**Init needs a real barrier.** RDMA resources are set up lazily, so an
`all_sum` over a non-trivial array is issued immediately after `init()` to
force that work to happen before any real traffic.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Model families that implement `shard(group)` in mlx-lm 0.31.3. This is only a
# fast pre-flight answer for the admin UI; the authoritative check is
# `hasattr(model, "shard")` after loading, since the list moves upstream.
KNOWN_TENSOR_PARALLEL_ARCHITECTURES = frozenset(
    {
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v3_2",
        "glm4_moe",
        "glm4_moe_lite",
        "gpt_oss",
        "kimi_k25",
        "minimax",
        "step3p5",
        "longcat_flash",
        "longcat_flash_ngram",
        "exaone_moe",
        "ministral3",
        "iquestloopcoder",
    }
)


@dataclass(frozen=True)
class WorldInfo:
    """This process's place in the collective."""

    rank: int
    size: int

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


class DistributedSession:
    """A joined collective, and the operations oMLX performs on it.

    Constructing this calls `mx.distributed.init()`, which reads the
    environment prepared by `omlx.cluster.hostfile`. There is intentionally no
    `close()`.
    """

    def __init__(self) -> None:
        self._group = mx.distributed.init()
        self.world = WorldInfo(rank=self._group.rank(), size=self._group.size())
        self.barrier()
        logger.info(
            "cluster: joined collective as rank %d of %d",
            self.world.rank,
            self.world.size,
        )

    @property
    def group(self):  # noqa: ANN201 - mx.distributed.Group is not exported
        return self._group

    def barrier(self) -> None:
        """Synchronise every rank, and force lazy RDMA setup on first call."""
        mx.eval(mx.distributed.all_sum(mx.ones(10), group=self._group, stream=mx.cpu))

    def seed_everyone(self, seed: int) -> int:
        """Agree on one RNG seed so sampling is bit-identical on every rank.

        Under tensor parallelism the final logits are all-reduced, so every
        rank holds the same distribution and will draw the same token *only*
        if it draws from the same generator state. Rank 0's seed wins; the
        others discard theirs.
        """
        chosen = mx.array([seed if self.world.is_leader else 0], dtype=mx.int64)
        chosen = mx.distributed.all_sum(chosen, group=self._group, stream=mx.cpu)
        mx.eval(chosen)
        agreed = int(chosen.item())
        mx.random.seed(agreed)
        return agreed

    def broadcast(self, obj: Any) -> Any:
        """Send a Python object from rank 0 to every rank.

        Pickle bytes are shipped through `all_sum`: ranks other than 0
        contribute zeros, so the sum is rank 0's payload. This is mlx-lm's own
        technique and it is used here only for control messages that must be
        perfectly ordered with respect to the collective stream.

        It is not free - it spends a collective on control flow - so bulk
        scheduling metadata should travel over the TCP control plane instead.
        Two collectives are needed because receivers must size the buffer
        before filling it.
        """
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

    def agree_int(self, value: int) -> int:
        """Take rank 0's value for something every rank must decide alike.

        Cheaper than `broadcast` for a single number, and used for the things
        that would otherwise diverge on local state - batch size, step counts,
        whether to stop.
        """
        arr = mx.array([value if self.world.is_leader else 0], dtype=mx.int64)
        arr = mx.distributed.all_sum(arr, group=self._group, stream=mx.cpu)
        mx.eval(arr)
        return int(arr.item())

    def all_agree(self, flag: bool) -> bool:
        """True only when *every* rank passes True.

        Used before anything that must not proceed half-way across the cluster.
        """
        arr = mx.array([1 if flag else 0], dtype=mx.int64)
        arr = mx.distributed.all_sum(arr, group=self._group, stream=mx.cpu)
        mx.eval(arr)
        return int(arr.item()) == self.world.size


def load_sharded(model_path: str, session: DistributedSession, *, pipeline: bool = False):
    """Load only the weights this rank needs.

    `sharded_load` lazily reads the config, decides shardability, and downloads
    or memory-maps only this rank's slice, which is the whole point: four
    machines each pull a quarter of the weights.

    Because neither backend supports group splitting, exactly one of the two
    group arguments is ever a real group. `pipeline=True` splits by layer depth
    and requires the model to expose `pipeline(group)`; the default splits by
    tensor and requires `shard(group)`.
    """
    from mlx_lm.utils import sharded_load

    group = session.group
    if pipeline:
        model, tokenizer = sharded_load(model_path, pipeline_group=group)
    else:
        model, tokenizer = sharded_load(model_path, tensor_group=group)

    logger.info(
        "cluster: rank %d loaded %s (%s parallel)",
        session.world.rank,
        model_path,
        "pipeline" if pipeline else "tensor",
    )
    return model, tokenizer


def parallelism_support(model: Any) -> tuple[bool, bool]:
    """`(tensor_parallel, pipeline_parallel)` support for a loaded model.

    This is the authoritative check. The architecture list above is only a
    guess made before loading.
    """
    return hasattr(model, "shard"), hasattr(model, "pipeline")
