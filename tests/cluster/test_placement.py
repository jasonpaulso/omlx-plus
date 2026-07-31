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


class TestLanguageModelOnlyEligibility:
    """S6 D0: a `language_model_only: true` checkpoint (vision tower shipped
    off) is text-eligible for distributed placement even though discovery
    classifies it "vlm" -- placement-layer only, single-node classification
    untouched (this module never touches model_discovery.py).
    """

    LANGUAGE_MODEL_ONLY_CONFIG = {
        "vision_config": {"some": "tower"},
        "language_model_only": True,
        "text_config": {"num_attention_heads": 24, "num_key_value_heads": 4},
    }
    PLAIN_VLM_CONFIG = {
        "vision_config": {"some": "tower"},
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
    }

    def test_language_model_only_vlm_is_distributed_eligible(self):
        decision = _plan(
            est_size=150,
            model_type="vlm",
            model_config=self.LANGUAGE_MODEL_ONLY_CONFIG,
        )
        assert decision.mode == "distributed"

    def test_plain_vlm_without_language_model_only_stays_refused(self):
        decision = _plan(
            est_size=150, model_type="vlm", model_config=self.PLAIN_VLM_CONFIG
        )
        assert decision.mode == "local"
        assert any("not eligible" in r for r in decision.reasons)

    def test_plain_vlm_rejects_under_distributed(self):
        decision = _plan(
            est_size=150,
            model_type="vlm",
            model_config=self.PLAIN_VLM_CONFIG,
            prefer="distributed",
        )
        assert decision.mode == "reject"

    def test_single_node_prefer_local_is_untouched_by_the_eligibility_change(self):
        # prefer=local short-circuits before eligibility is even consulted --
        # single-node serving of a language_model_only OR a plain vlm config
        # is unaffected by this change either way.
        for config in (self.LANGUAGE_MODEL_ONLY_CONFIG, self.PLAIN_VLM_CONFIG):
            decision = _plan(
                est_size=1000, model_type="vlm", model_config=config, prefer="local"
            )
            assert decision.mode == "local"


class TestD0EvidenceCrossCheck:
    """S6 P1c item 5 (P2-exec catch): a MISLABELED checkpoint -- declares
    `language_model_only: true` while its own weight index still ships a
    vision tower -- must not be silently trusted on the DECLARED boolean
    alone. `resolve_placement_inputs` folds the real evidence
    (`has_vision_tower_weights`) into the config it returns; `plan_placement`
    reads that evidence, never a file itself (it stays pure).
    """

    TEXT_CONFIG = {"num_attention_heads": 24, "num_key_value_heads": 4}

    def test_declared_true_no_vision_weights_present_is_eligible(self):
        config = {
            "language_model_only": True,
            "text_config": self.TEXT_CONFIG,
            "_language_model_only_vision_weights_present": False,
        }
        decision = _plan(est_size=150, model_type="vlm", model_config=config)
        assert decision.mode == "distributed"

    def test_declared_true_vision_weights_present_is_refused_by_default(self):
        config = {
            "language_model_only": True,
            "text_config": self.TEXT_CONFIG,
            "_language_model_only_vision_weights_present": True,
        }
        decision = _plan(est_size=150, model_type="vlm", model_config=config)
        assert decision.mode == "local"
        assert any("vision-tower parameters" in r for r in decision.reasons)

    def test_declared_true_vision_weights_present_rejects_under_distributed(self):
        config = {
            "language_model_only": True,
            "text_config": self.TEXT_CONFIG,
            "_language_model_only_vision_weights_present": True,
        }
        decision = _plan(
            est_size=150, model_type="vlm", model_config=config, prefer="distributed"
        )
        assert decision.mode == "reject"

    def test_declared_false_is_unchanged(self):
        # No cross-check evidence key at all is consulted when the flag
        # itself is false -- exactly today's plain-vlm-refused behaviour.
        config = {
            "language_model_only": False,
            "vision_config": {"some": "tower"},
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
        }
        decision = _plan(est_size=150, model_type="vlm", model_config=config)
        assert decision.mode == "local"
        assert any("not eligible" in r for r in decision.reasons)

    def test_resolve_placement_inputs_folds_in_the_real_evidence(self, tmp_path):
        """The actual I/O boundary, not a hand-built config dict: a
        `language_model_only: true` config.json paired with a real
        vision-tower-carrying safetensors index must come back MISLABELED,
        and `plan_placement` must refuse it on the default path."""
        config = {"language_model_only": True, "text_config": self.TEXT_CONFIG}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"0" * 500)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"0" * 500)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "language_model.layers.0.weight": (
                            "model-00001-of-00002.safetensors"
                        ),
                        "vision_tower.blocks.0.weight": (
                            "model-00002-of-00002.safetensors"
                        ),
                    }
                }
            )
        )

        _est, read_config = resolve_placement_inputs(str(tmp_path))

        assert read_config is not None
        assert read_config["_language_model_only_vision_weights_present"] is True
        decision = _plan(est_size=150, model_type="vlm", model_config=read_config)
        assert decision.mode == "local"
        assert any("vision-tower parameters" in r for r in decision.reasons)

    def test_resolve_placement_inputs_leaves_declared_false_configs_untouched(
        self, tmp_path
    ):
        config = {"model_type": "qwen3", "num_attention_heads": 8}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)

        _est, read_config = resolve_placement_inputs(str(tmp_path))

        assert read_config == config  # byte for byte -- no evidence key added


class TestTextOnlyDistributionOptIn:
    """S6 P1c item 6 (Qwen-anchor fork, user-decided 2026-07-30):
    `allow_text_only_distribution=True` -- eligibility for a "vlm"-classified
    model no longer hinges on `language_model_only` at all. Default (False,
    the value every OTHER test in this module implicitly exercises via
    `_plan`'s default) is byte-for-byte item 5's behaviour.
    """

    def test_off_by_default_leaves_a_plain_vlm_refused(self):
        config = {"num_attention_heads": 24, "num_key_value_heads": 4}
        decision = _plan(est_size=150, model_type="vlm", model_config=config)
        assert decision.mode == "local"

    def test_on_makes_a_plain_vlm_eligible(self):
        # The real Qwen3.6-27B-bf16 shape: `language_model_only: FALSE`,
        # genuinely multimodal, divisible via text_config.
        config = {
            "language_model_only": False,
            "vision_config": {"some": "tower"},
            "text_config": {"num_attention_heads": 24, "num_key_value_heads": 4},
        }
        decision = _plan(
            est_size=150,
            model_type="vlm",
            model_config=config,
            allow_text_only_distribution=True,
        )
        assert decision.mode == "distributed"

    def test_on_still_enforces_divisibility_via_text_config(self):
        config = {
            "language_model_only": False,
            "vision_config": {"some": "tower"},
            "text_config": {"num_attention_heads": 7},
        }
        decision = _plan(
            est_size=150,
            model_type="vlm",
            model_config=config,
            allow_text_only_distribution=True,
            prefer="distributed",
        )
        assert decision.mode == "reject"
        assert any("not divisible" in r for r in decision.reasons)

    def test_on_leaves_llm_eligibility_untouched(self):
        decision = _plan(
            est_size=150, model_type="llm", allow_text_only_distribution=True
        )
        assert decision.mode == "distributed"

    def test_on_does_not_make_non_vlm_non_llm_types_eligible(self):
        config = {"num_attention_heads": 24, "num_key_value_heads": 4}
        decision = _plan(
            est_size=150,
            model_type="embedding",
            model_config=config,
            allow_text_only_distribution=True,
        )
        assert decision.mode == "local"

    def test_single_node_prefer_local_is_untouched_by_the_opt_in(self):
        config = {"vision_config": {"some": "tower"}}
        decision = _plan(
            est_size=1000,
            model_type="vlm",
            model_config=config,
            prefer="local",
            allow_text_only_distribution=True,
        )
        assert decision.mode == "local"


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
