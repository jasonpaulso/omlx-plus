# SPDX-License-Identifier: Apache-2.0
"""S4 D2: placement — deciding local vs distributed for one model.

:func:`plan_placement` is the one pure function: no I/O, no clock, no
globals. Callers assemble every input and the returned
:class:`PlacementDecision` is JSON-serializable, carrying its full
reasoning so the same object can be previewed (``GET
/v1/cluster/placement``), recorded on a formation job, and compared for
the S4 acceptance row that checks preview against actual.

The I/O-bearing pieces that assemble those inputs — reading a model's
``config.json`` for the divisibility check, converting a worker's D1
``MemberNodeState`` into a :class:`NodeCapacity` — live in this module too,
but are deliberately separate functions so the pure decision logic never
touches a filesystem or a clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..model_discovery import estimate_model_size
from .state import MemberNodeState
from .tp import TPDivisibilityError, check_divisibility

PlacementMode = Literal["local", "distributed", "reject"]
Prefer = Literal["auto", "local", "distributed"]

# S4 D2 rule 2: only a plain LLM is distributable in v1. The ClusterEngine
# rejects VLM/SpecPrefill at request time with no load-time gate, so without
# this predicate a too-big VLM would form a resident formation that can
# serve nothing.
_ELIGIBLE_MODEL_TYPES = frozenset({"llm"})

# S2-measured 0.538x params/rank for dense bf16; MoE-3-bit measured only
# operationally. Deliberately conservative, revisited only with a
# measurement in hand — a code constant, not a setting.
_PER_RANK_HEADROOM = 1.15


@dataclass(frozen=True)
class NodeCapacity:
    """One node's placement-relevant capacity: head's own, or a worker's.

    ``models_present`` maps model_id -> size_bytes for models physically on
    that node (not necessarily loaded) — the presence-aware scoring input.
    """

    node_id: str
    memory_ceiling: int
    current_model_memory: int
    models_present: dict[str, int] = field(default_factory=dict)

    @property
    def capacity_known(self) -> bool:
        """D2's binding capacity-unknown rule: a ceiling of 0 is unknown."""
        return self.memory_ceiling > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "memory_ceiling": self.memory_ceiling,
            "current_model_memory": self.current_model_memory,
            "models_present": dict(self.models_present),
        }


@dataclass(frozen=True)
class NodeFit:
    """Whether one node can hold its share of a candidate placement."""

    ceiling: int
    projected: int
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {"ceiling": self.ceiling, "projected": self.projected, "ok": self.ok}


@dataclass(frozen=True)
class PlacementDecision:
    """The full output of :func:`plan_placement` — previewable, recordable."""

    mode: PlacementMode
    world_size: int
    per_rank_estimate: int
    reasons: tuple[str, ...]
    fits: dict[str, NodeFit]
    presence: dict[str, bool]
    divisible: bool
    requires_eviction: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "world_size": self.world_size,
            "per_rank_estimate": self.per_rank_estimate,
            "reasons": list(self.reasons),
            "fits": {node_id: fit.to_dict() for node_id, fit in self.fits.items()},
            "presence": dict(self.presence),
            "divisible": self.divisible,
            "requires_eviction": self.requires_eviction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlacementDecision:
        return cls(
            mode=data["mode"],
            world_size=int(data["world_size"]),
            per_rank_estimate=int(data["per_rank_estimate"]),
            reasons=tuple(str(r) for r in data.get("reasons") or []),
            fits={
                str(node_id): NodeFit(
                    ceiling=int(fit["ceiling"]),
                    projected=int(fit["projected"]),
                    ok=bool(fit["ok"]),
                )
                for node_id, fit in (data.get("fits") or {}).items()
            },
            presence={
                str(node_id): bool(present)
                for node_id, present in (data.get("presence") or {}).items()
            },
            divisible=bool(data["divisible"]),
            requires_eviction=bool(data["requires_eviction"]),
        )


def _local_fit(head: NodeCapacity, est_size: int) -> dict[str, NodeFit]:
    projected = head.current_model_memory + est_size
    ok = head.capacity_known and projected <= head.memory_ceiling
    return {
        head.node_id: NodeFit(ceiling=head.memory_ceiling, projected=projected, ok=ok)
    }


def plan_placement(
    model_id: str,
    model_type: str,
    est_size: int,
    model_config: dict[str, Any] | None,
    head: NodeCapacity,
    workers: list[NodeCapacity],
    prefer: Prefer,
) -> PlacementDecision:
    """Decide local vs distributed vs reject for one model. Pure, no I/O.

    Rules (S4 D2, v1, 2-node):

    1. ``prefer=local``, or ``prefer=auto`` and it fits the head, -> local.
    2. ``prefer=distributed``, or ``prefer=auto`` and it doesn't fit
       locally, -> distributed iff the model is eligible (``model_type ==
       "llm"``) AND divisibility passes AND the per-rank estimate fits
       every rank's *known* ceiling.
    3. Otherwise -> reject, naming every failed predicate.

    Capacity-unknown (a ceiling of 0, or no worker capacity at all) never
    auto-distributes: ``prefer=auto`` resolves to local with the reason
    recorded; ``prefer=distributed`` rejects with the same reason. This is
    distinct from a *known* insufficient/indivisible outcome, which always
    rejects (even under ``auto``) rather than silently falling back to a
    local placement that will not fit.
    """
    presence = {
        node.node_id: model_id in node.models_present for node in [head, *workers]
    }
    eligible = model_type in _ELIGIBLE_MODEL_TYPES

    def local_decision(reasons: list[str]) -> PlacementDecision:
        requires_eviction = head.current_model_memory + est_size > head.memory_ceiling
        return PlacementDecision(
            mode="local",
            world_size=1,
            per_rank_estimate=est_size,
            reasons=tuple(reasons),
            fits=_local_fit(head, est_size),
            presence=presence,
            divisible=True,
            requires_eviction=requires_eviction,
        )

    if prefer == "local":
        return local_decision([])

    if not eligible:
        reason = f"model_type={model_type!r} is not eligible for distributed placement"
        if prefer == "distributed":
            return PlacementDecision(
                mode="reject",
                world_size=1,
                per_rank_estimate=est_size,
                reasons=(reason,),
                fits=_local_fit(head, est_size),
                presence=presence,
                divisible=True,
                requires_eviction=False,
            )
        return local_decision([reason])

    if prefer == "auto" and head.capacity_known and est_size <= head.memory_ceiling:
        return local_decision([])

    # Attempt distributed: prefer=distributed, or prefer=auto that doesn't
    # fit (or can't be verified to fit) locally.
    if not workers:
        reason = "no cluster members available for distributed placement"
        if prefer == "auto":
            return local_decision([reason])
        return PlacementDecision(
            mode="reject",
            world_size=1,
            per_rank_estimate=est_size,
            reasons=(reason,),
            fits=_local_fit(head, est_size),
            presence=presence,
            divisible=True,
            requires_eviction=False,
        )

    unknown_workers = [w for w in workers if not w.capacity_known]
    if unknown_workers:
        reason = "worker capacity unknown"
        if prefer == "auto":
            return local_decision([reason])
        return PlacementDecision(
            mode="reject",
            world_size=1 + len(workers),
            per_rank_estimate=est_size,
            reasons=(reason,),
            fits=_local_fit(head, est_size),
            presence=presence,
            divisible=True,
            requires_eviction=False,
        )

    # Both head and every worker report known capacity: this is a "known"
    # outcome from here on, so it always rejects rather than falling back
    # to local, even under prefer=auto (rule 3).
    world_size = 1 + len(workers)
    reasons: list[str] = []
    divisible = True
    if model_config is None:
        divisible = False
        reasons.append("model config unavailable; cannot verify TP divisibility")
    else:
        try:
            check_divisibility(model_config, world_size)
        except TPDivisibilityError as exc:
            divisible = False
            reasons.append(str(exc))

    per_rank_estimate = int(est_size / world_size * _PER_RANK_HEADROOM)
    nodes = [head, *workers]
    fits: dict[str, NodeFit] = {}
    all_fit = True
    for node in nodes:
        projected = node.current_model_memory + per_rank_estimate
        ok = node.capacity_known and projected <= node.memory_ceiling
        fits[node.node_id] = NodeFit(
            ceiling=node.memory_ceiling, projected=projected, ok=ok
        )
        if not ok:
            all_fit = False
    if not all_fit:
        reasons.append("model does not fit under the per-rank ceiling on every node")

    for node in nodes:
        if not presence[node.node_id]:
            reasons.append(f"model absent on {node.node_id}; transfer required (S5)")

    if divisible and all_fit:
        return PlacementDecision(
            mode="distributed",
            world_size=world_size,
            per_rank_estimate=per_rank_estimate,
            reasons=tuple(reasons),
            fits=fits,
            presence=presence,
            divisible=True,
            requires_eviction=False,
        )

    return PlacementDecision(
        mode="reject",
        world_size=world_size,
        per_rank_estimate=per_rank_estimate,
        reasons=tuple(reasons),
        fits=fits,
        presence=presence,
        divisible=divisible,
        requires_eviction=False,
    )


def resolve_placement_inputs(model_path: str) -> tuple[int, dict[str, Any] | None]:
    """I/O-bearing: fresh ``est_size`` + raw ``config.json`` for a model dir.

    Separate from :func:`plan_placement` on purpose (D2). Nothing in
    discovery retains a model's head counts, so this does its own
    ``json.load`` of ``config.json`` for the two fields
    :func:`~omlx.cluster.tp.check_divisibility` needs — cheap, no model
    instantiation. A missing/unreadable config returns ``None`` rather than
    raising; the caller then sees ``divisible=False`` with a clear reason.
    """
    path = Path(model_path)
    est_size = estimate_model_size(path)
    config: dict[str, Any] | None = None
    try:
        with open(path / "config.json", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        config = None
    return est_size, config


def worker_node_capacity(member_id: str, state: MemberNodeState) -> NodeCapacity:
    """Convert a stored D1 :class:`MemberNodeState` into a placement input.

    ``current_model_memory`` is not part of the D1 wire shape (only
    ``memory_ceiling`` and inventory are self-reported), so it is always 0
    here — a recorded limitation, not an oversight: the per-rank fit check
    is conservative against a worker's total ceiling rather than its live
    occupancy.
    """
    return NodeCapacity(
        node_id=member_id,
        memory_ceiling=state.memory_ceiling,
        current_model_memory=0,
        models_present=dict(state.models_present),
    )
