# SPDX-License-Identifier: Apache-2.0
"""Tests for S4 D2 placement: the pure `plan_placement` function plus the
I/O-bearing input helper that reads a model's `config.json`.
"""

from __future__ import annotations

import json

from omlx.cluster.placement import (
    NodeCapacity,
    NodeFit,
    PlacementDecision,
    plan_placement,
    resolve_placement_inputs,
    worker_node_capacity,
)
from omlx.cluster.state import MemberNodeState

HEAD = NodeCapacity(
    node_id="head",
    memory_ceiling=100,
    current_model_memory=0,
    models_present={"m": 100},
)
WORKER = NodeCapacity(
    node_id="w1", memory_ceiling=100, current_model_memory=0, models_present={"m": 100}
)
DIVISIBLE_CONFIG = {"num_attention_heads": 8, "num_key_value_heads": 2}
INDIVISIBLE_CONFIG = {"num_attention_heads": 7}


def _plan(**overrides):
    kwargs = dict(
        model_id="m",
        model_type="llm",
        est_size=50,
        model_config=DIVISIBLE_CONFIG,
        head=HEAD,
        workers=[WORKER],
        prefer="auto",
    )
    kwargs.update(overrides)
    return plan_placement(**kwargs)


class TestLocalRule:
    def test_fits_local(self):
        decision = _plan(est_size=50)
        assert decision.mode == "local"
        assert decision.requires_eviction is False
        assert decision.world_size == 1
        assert decision.fits["head"].ok is True

    def test_fits_local_with_eviction(self):
        head = NodeCapacity(
            node_id="head",
            memory_ceiling=100,
            current_model_memory=60,
            models_present={},
        )
        decision = _plan(est_size=50, head=head)
        assert decision.mode == "local"
        assert decision.requires_eviction is True
        # Nothing evictable on this head (e.g. the resident memory is
        # pinned) -- the fit itself stays not-ok even though eviction
        # would be needed to fix it.
        assert decision.fits["head"].ok is False

    def test_fits_local_with_eviction_credits_evictable_memory(self):
        """S4 P3 rig defect #3: an evictable (non-pinned, idle, local)
        resident on the head makes the fit `ok` once credited, even though
        the raw projected total still exceeds the ceiling."""
        head = NodeCapacity(
            node_id="head",
            memory_ceiling=100,
            current_model_memory=60,
            evictable_memory=60,
            models_present={},
        )
        decision = _plan(est_size=50, head=head, prefer="local")
        assert decision.mode == "local"
        assert decision.requires_eviction is True
        assert decision.fits["head"].ok is True

    def test_prefer_local_short_circuits_even_when_it_does_not_fit(self):
        decision = _plan(est_size=1000, prefer="local")
        assert decision.mode == "local"


class TestDistributedRule:
    def test_too_big_local_divisible_fits_per_rank_goes_distributed(self):
        decision = _plan(est_size=150)
        assert decision.mode == "distributed"
        assert decision.world_size == 2
        assert decision.divisible is True
        assert decision.per_rank_estimate == int(150 / 2 * 1.15)
        assert all(fit.ok for fit in decision.fits.values())

    def test_indivisible_rejects_with_reason(self):
        decision = _plan(est_size=150, model_config=INDIVISIBLE_CONFIG)
        assert decision.mode == "reject"
        assert decision.divisible is False
        assert any("divisible" in r for r in decision.reasons)

    def test_too_big_everywhere_rejects(self):
        """A known, insufficient distributed fit always rejects — even
        under prefer=auto — rather than silently falling back to local."""
        decision = _plan(est_size=1000)
        assert decision.mode == "reject"
        assert decision.divisible is True
        assert any("fit" in r for r in decision.reasons)

    def test_prefer_distributed_on_a_locally_fitting_model(self):
        decision = _plan(est_size=50, prefer="distributed")
        assert decision.mode == "distributed"

    def test_absent_on_worker_still_distributes_with_transfer_reason(self):
        worker = NodeCapacity(
            node_id="w1", memory_ceiling=100, current_model_memory=0, models_present={}
        )
        decision = _plan(est_size=150, workers=[worker])
        assert decision.mode == "distributed"
        assert decision.presence["w1"] is False
        assert any("absent on w1" in r for r in decision.reasons)

    def test_distributed_fit_with_evictable_local_resident_needs_eviction(self):
        """S4 P3 rig defect #3: the head-side per-rank fit in the
        distributed branch gets the same eviction credit as the local
        rule-1 fit. Without it, a distributed reload that only needs the
        pool to LRU-evict a non-pinned local model reads as a flat reject.
        """
        head = NodeCapacity(
            node_id="head",
            memory_ceiling=100,
            current_model_memory=90,
            evictable_memory=90,
            models_present={"m": 100},
        )
        decision = _plan(est_size=150, head=head)
        assert decision.mode == "distributed"
        assert decision.requires_eviction is True
        head_fit = decision.fits["head"]
        assert head_fit.ok is True
        assert head_fit.requires_eviction is True
        # Uncredited: the worker side is untouched by the head's eviction.
        assert decision.fits["w1"].requires_eviction is False

    def test_distributed_rejects_when_head_resident_is_pinned_not_evictable(self):
        """Same shape as above, but the head's resident memory isn't
        evictable (e.g. pinned) -- still a genuine reject, not silently
        admitted."""
        head = NodeCapacity(
            node_id="head",
            memory_ceiling=100,
            current_model_memory=90,
            evictable_memory=0,
            models_present={"m": 100},
        )
        decision = _plan(est_size=150, head=head)
        assert decision.mode == "reject"
        head_fit = decision.fits["head"]
        assert head_fit.ok is False
        assert head_fit.requires_eviction is True
        assert any("fit" in r for r in decision.reasons)


class TestNoMembers:
    def test_no_members_under_auto_is_local_not_reject(self):
        decision = _plan(est_size=150, workers=[])
        assert decision.mode == "local"
        assert any("no cluster members" in r for r in decision.reasons)

    def test_no_members_under_distributed_rejects(self):
        decision = _plan(est_size=150, workers=[], prefer="distributed")
        assert decision.mode == "reject"
        assert any("no cluster members" in r for r in decision.reasons)


class TestCapacityUnknown:
    """D2's binding rule: a ceiling of 0 (or no node_state) is unknown
    capacity, and placement never auto-distributes on unknown."""

    UNKNOWN_WORKER = NodeCapacity(
        node_id="w1", memory_ceiling=0, current_model_memory=0, models_present={}
    )

    def test_unknown_worker_capacity_under_auto_is_local_not_reject(self):
        decision = _plan(est_size=150, workers=[self.UNKNOWN_WORKER])
        assert decision.mode == "local"
        assert any("worker capacity unknown" in r for r in decision.reasons)

    def test_unknown_worker_capacity_under_distributed_rejects(self):
        decision = _plan(
            est_size=150, workers=[self.UNKNOWN_WORKER], prefer="distributed"
        )
        assert decision.mode == "reject"
        assert any("worker capacity unknown" in r for r in decision.reasons)


class TestEligibility:
    """rev6: only model_type=="llm" is distributable."""

    def test_too_big_vlm_is_local_under_auto(self):
        decision = _plan(est_size=150, model_type="vlm")
        assert decision.mode == "local"
        assert any("not eligible" in r for r in decision.reasons)

    def test_too_big_vlm_rejects_under_distributed(self):
        decision = _plan(est_size=150, model_type="vlm", prefer="distributed")
        assert decision.mode == "reject"

    def test_too_big_embedding_is_local_under_auto(self):
        decision = _plan(est_size=150, model_type="embedding")
        assert decision.mode == "local"

    def test_too_big_embedding_rejects_under_distributed(self):
        decision = _plan(est_size=150, model_type="embedding", prefer="distributed")
        assert decision.mode == "reject"


class TestDecisionRoundTrip:
    def test_to_dict_from_dict_round_trips(self):
        decision = _plan(est_size=150)
        assert decision.mode == "distributed"  # sanity: exercise a rich shape
        restored = PlacementDecision.from_dict(
            json.loads(json.dumps(decision.to_dict()))
        )
        assert restored == decision

    def test_node_fit_and_decision_to_dict_shape(self):
        fit = NodeFit(ceiling=10, projected=5, ok=True)
        assert fit.to_dict() == {
            "ceiling": 10,
            "projected": 5,
            "ok": True,
            "requires_eviction": False,
        }


class TestResolvePlacementInputs:
    """The I/O-bearing helper, exercised against a fixture dir (no live model)."""

    def test_reads_size_and_config(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)
        config = {"model_type": "qwen3", "num_attention_heads": 8}
        (tmp_path / "config.json").write_text(json.dumps(config))

        est_size, read_config = resolve_placement_inputs(str(tmp_path))

        assert est_size > 0
        assert read_config == config

    def test_missing_config_returns_none_not_raise(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)

        est_size, read_config = resolve_placement_inputs(str(tmp_path))

        assert est_size > 0
        assert read_config is None


class TestWorkerNodeCapacity:
    def test_converts_stored_node_state(self):
        state = MemberNodeState(
            total_memory=1000,
            memory_ceiling=100,
            models_present={"m": 50},
            received_at=1.0,
        )
        capacity = worker_node_capacity("w1", state)
        assert capacity == NodeCapacity(
            node_id="w1",
            memory_ceiling=100,
            current_model_memory=0,
            models_present={"m": 50},
        )
