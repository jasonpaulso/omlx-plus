# SPDX-License-Identifier: Apache-2.0
"""S4 D4 superseded the E1 fast-path touchpoint (`_resolve_cluster_engine`,
resolved before the pool lock) with cluster entries as first-class
`EngineEntry` rows: a formed model is reached through the normal
already-loaded path in `get_engine`, gated by `entry.kind == "cluster"`.

Same coverage intent as the old touchpoint test: a formed model is
reachable through the pool, and `cluster.role=off` takes the identical path
it always did (no cluster entry can exist).
"""

from __future__ import annotations

from omlx.engine_pool import EngineEntry, EnginePool


def _make_pool() -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    return pool


async def test_role_off_has_no_cluster_entries(tmp_path):
    """With no cluster manager installed (role=off), a plain unloaded entry
    behaves exactly as it always did -- no `kind="cluster"` branch is ever
    reachable.
    """
    from omlx.cluster.manager import set_cluster_manager

    set_cluster_manager(None)
    pool = _make_pool()
    pool._entries["local-model"] = EngineEntry(
        model_id="local-model",
        model_path=str(tmp_path),
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )
    entry = pool.get_entry("local-model")
    assert entry is not None
    assert entry.kind == "local"


async def test_formed_cluster_entry_is_reachable_via_get_engine():
    """A `kind="cluster"` entry with an already-bound engine is reachable
    through `get_engine`'s normal already-loaded fast path -- zero I/O, no
    placement call, the engine object returned as-is.
    """
    pool = _make_pool()
    sentinel = object()
    entry = EngineEntry(
        model_id="formed",
        model_path="/does/not/matter",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )
    entry.kind = "cluster"
    entry.engine = sentinel
    pool._entries["formed"] = entry

    result = await pool.get_engine("formed")
    assert result is sentinel
    # The fast path never touches runtime_settings_signature for cluster
    # entries (D4's variant guard skips the whole signature block).
    assert entry.runtime_settings_signature is None


async def test_formed_cluster_entry_leases_and_updates_last_access():
    pool = _make_pool()
    sentinel = object()
    entry = EngineEntry(
        model_id="formed",
        model_path="/does/not/matter",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )
    entry.kind = "cluster"
    entry.engine = sentinel
    pool._entries["formed"] = entry

    result = await pool.get_engine("formed", _lease=True)
    assert result is sentinel
    assert entry.in_use == 1
    assert entry.last_access > 0


async def test_unformed_model_id_falls_through_normally():
    pool = _make_pool()
    entry = EngineEntry(
        model_id="other",
        model_path="/does/not/matter",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
    )
    pool._entries["other"] = entry
    assert pool.get_entry("other").engine is None
    assert pool.get_entry("other").kind == "local"
