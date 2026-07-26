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
from functools import lru_cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def tensor_parallel_architectures() -> frozenset[str]:
    """Model families whose `Model` class implements `shard(group)`.

    Derived by inspecting the installed mlx-lm rather than hard-coded. The set
    moves with every mlx-lm release, and a stale allow-list is worse than none:
    it either refuses a model that would have worked or promises one that will
    not. A hand-written version of this list was wrong in three places against
    mlx-lm 0.31.3.

    This is only a fast answer for the admin UI and preflight. The
    authoritative check is `parallelism_support()` on the loaded model.
    """
    import importlib
    import io
    import pkgutil
    from contextlib import redirect_stderr, redirect_stdout

    import mlx_lm.models

    found: set[str] = set()
    # Some model modules print installation hints on import; keep that noise
    # out of the daemon's own stdout.
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        for module in pkgutil.iter_modules(mlx_lm.models.__path__):
            try:
                loaded = importlib.import_module(f"mlx_lm.models.{module.name}")
            except BaseException:  # noqa: BLE001 - optional deps, bad imports
                continue
            model_cls = getattr(loaded, "Model", None)
            if model_cls is not None and hasattr(model_cls, "shard"):
                found.add(module.name)
    return frozenset(found)


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

    Note that **no model shipped in mlx-lm 0.31.3 defines `pipeline`** - all 21
    shardable families implement `shard` only. The pipeline path is therefore
    unreachable with the current mlx-lm and is kept because upstream is
    actively adding pipeline support for the largest models. `load` refuses
    rather than proceeding if the loaded model cannot do what was asked.
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
