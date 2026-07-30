# SPDX-License-Identifier: Apache-2.0
"""The wire format between a daemon and its rank processes.

Three parties depend on this module and none of them import the others: the
daemon-side engine writes commands and reads reply frames, the rank-0 worker
reads commands and writes replies, and the tests exercise both without
spawning mlx. It is therefore free of any mlx import.

Two wire shapes live here:

* **Pipe frames** (`GenerationSpec`, reply frames) — newline-delimited JSON
  between a daemon and its own rank-0 child, over stdin/stdout plus one
  out-of-band control pipe.
* **The per-step collective message** (`StepMessage`) — the E4 unified
  broadcast rank 0 issues to every rank once per decode step.
* **The head->worker command schema** (`Command`) — a closed, Pydantic-typed,
  versioned envelope. It is defined here so both the head that mints it and
  the worker that confines it share one definition; P1 owns the schema and its
  rejection tests, P2 wires it onto the heartbeat.

Everything is JSON, never pickle: CL-09 accepts an on-link attacker who can
inject into the ring, and a pickle frame would upgrade "inject wrong bytes"
to arbitrary code execution on every rank (D4). A frame that is not valid
JSON of the expected shape is rejected, not coerced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Bumped when the pipe/collective/command wire shape changes. E10 already
# rejects stack skew at join (`versions.compare_versions`), so head and worker
# always run identical code and this integer is a second, explicit guard
# rather than the primary skew defence (CL2-04).
# S3 bump (was 1): adds RankOp, the forward-replay message the TP batch
# generator broadcasts once per model invocation.
# S5 bump (was 2): adds TRANSFER_START/TRANSFER_ROUND/TRANSFER_ABORT.
PROTOCOL_VERSION = 3


class ProtocolError(ValueError):
    """A frame could not be parsed as the protocol shape it claimed to be."""


# -- generation request (daemon -> rank 0, then broadcast to every rank) -----


@dataclass
class GenerationSpec:
    """One generation request, as it crosses the pipe.

    Prompt tokens are sent already encoded: the daemon holds the tokenizer it
    used to apply the chat template, and re-encoding the rendered string in the
    worker would be a second chance to disagree with it for nothing.
    """

    prompt_ids: list[int]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float | None = None
    repetition_context_size: int = 20
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_ids": list(self.prompt_ids),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "repetition_context_size": self.repetition_context_size,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "seed": self.seed,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationSpec:
        known = {
            "prompt_ids",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "stop_token_ids",
            "seed",
            "request_id",
        }
        return cls(**{k: v for k, v in data.items() if k in known})


# -- the E4 per-step collective message --------------------------------------

# Composition-delta operations carried inside a StepMessage. S3 extends this
# set; S2 uses only these three, all originated by rank 0.
DELTA_FINISH = "finish"
DELTA_ABORT = "abort"
DELTA_ADMIT = "admit"


@dataclass
class StepMessage:
    """The single message rank 0 broadcasts to every rank each decode step.

    ``tokens`` maps request id to the token id sampled this step; ``deltas``
    carry composition changes (finish/abort/admit) that only rank 0 can see;
    ``done`` tells every rank the generation ended this step. Serialised as
    JSON bytes and shipped through two ``all_sum`` collectives (size then
    payload), never pickle (D4).
    """

    step: int
    tokens: dict[str, int]
    deltas: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tokens": {str(k): int(v) for k, v in self.tokens.items()},
            "deltas": [dict(d) for d in self.deltas],
            "done": bool(self.done),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepMessage:
        if not isinstance(data, dict) or "step" not in data or "tokens" not in data:
            raise ProtocolError("StepMessage payload missing required fields")
        tokens = data.get("tokens") or {}
        if not isinstance(tokens, dict):
            raise ProtocolError("StepMessage 'tokens' must be an object")
        deltas = data.get("deltas") or []
        if not isinstance(deltas, list):
            raise ProtocolError("StepMessage 'deltas' must be an array")
        return cls(
            step=int(data["step"]),
            tokens={str(k): int(v) for k, v in tokens.items()},
            deltas=[dict(d) for d in deltas],
            done=bool(data.get("done", False)),
        )

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> StepMessage:
        """Parse a broadcast payload. A non-JSON frame (e.g. pickle) is rejected.

        The collective ships bytes; anything an on-link attacker could inject
        that is not our JSON shape must fail here rather than be interpreted.
        """
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"StepMessage payload is not JSON: {exc}") from exc
        return cls.from_dict(data)


# -- the S3 forward-replay collective message --------------------------------

# The only op kind defined so far. A closed set, per D3/CL2-04: any other
# value is rejected loudly, never ignored.
RANK_OP_FORWARD = "forward"

# ``phase`` discriminates three shapes of forward that the follower must
# reconcile differently — inferring it from ``len(tags)`` is not reliable
# (a single newly-admitted row's delta forward is a length-1 *batched* op),
# so it rides the wire explicitly instead:
#
# * ``standalone`` — a per-request prefill forward (``scheduler.py:3191``,
#   ``:4419``). Always exactly one tag, run over that request's own
#   unwrapped per-layer cache; the cache lives independently of any batch
#   until (if ever) it is admitted.
# * ``admit`` — ``TPBatchGenerator.insert()``'s delta-batch forward: the one
#   decode step mlx-lm's ``GenerationBatch.__init__`` runs synchronously on
#   construction, over a *freshly batch-wrapped* (``BatchKVCache``, one row)
#   copy of that same cache — never the running batch. Always exactly one
#   tag. This must not be folded into ``batched`` framing: a follower
#   reconciling a later genuinely-batched op's tag list against this op's
#   single tag would read it as "the batch is now just this row" and drop
#   every other running row. It also must not be folded into ``standalone``
#   framing: unlike prefill, the leader's forward here runs over a wrapped
#   batch-of-one cache, and replaying it against an unwrapped cache would
#   size-mismatch the shard's collectives against the leader's. The
#   follower stores the wrapped, now-stepped result so a later ``batched``
#   op admitting this tag can reuse it (mirroring ``GenerationBatch.extend``
#   folding an already-stepped delta into the persistent batch).
# * ``batched`` — a TPBatchGenerator decode forward over the persistent
#   running batch. ``tags`` is the full ordered row set of that batch as of
#   this call.
PHASE_STANDALONE = "standalone"
PHASE_ADMIT = "admit"
PHASE_BATCHED = "batched"
_KNOWN_PHASES = {PHASE_STANDALONE, PHASE_ADMIT, PHASE_BATCHED}


@dataclass
class RankOp:
    """One model invocation, broadcast from the leader's model proxy to every
    follower so it can replay the identical forward on its own shard (D2/D3).

    ``tags`` identifies which per-request cache(s) this call touches — a
    monotonic, never-reused id minted by the leader's tag registry on first
    sight of a cache. For a ``batched`` op this is the **full ordered row
    set** of the batch (not just newly introduced tags): the follower diffs
    its own locally-held order against this list to derive extends/filters
    and must end up with an identical order, which turns a desync into a
    loud error instead of a hung collective. ``token_ids`` is parallel to
    ``tags`` — one token sequence per row (length 1 for decode, >1 for a
    prefill chunk). ``release`` carries tags to free *after* this op's
    forward is replayed (a sweep-detected stranded cache, mainly; ordinary
    per-row batch removal is already implied by a row's absence from a later
    ``batched`` op's ``tags``) — composition changes ride the same message
    as the tokens, per D3.
    """

    tags: list[int]
    token_ids: list[list[int]]
    release: list[int] = field(default_factory=list)
    done: bool = False
    op: str = RANK_OP_FORWARD
    phase: str = PHASE_BATCHED

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "phase": self.phase,
            "tags": [int(t) for t in self.tags],
            "token_ids": [[int(t) for t in row] for row in self.token_ids],
            "release": [int(t) for t in self.release],
            "done": bool(self.done),
        }

    @classmethod
    def from_dict(cls, data: Any) -> RankOp:
        """Parse one broadcast op, rejecting anything unexpected (CL2-04).

        Rejected — with a named error, never silently ignored or coerced:
        a non-object payload, an unrecognised ``op`` kind or ``phase``, a
        missing required field, a field of the wrong shape, or any extra
        key the schema does not declare.
        """
        if not isinstance(data, dict):
            raise ProtocolError("RankOp payload must be a JSON object")
        known = {"op", "phase", "tags", "token_ids", "release", "done"}
        extra = set(data.keys()) - known
        if extra:
            raise ProtocolError(f"RankOp payload has unknown fields: {sorted(extra)}")
        op = data.get("op")
        if op != RANK_OP_FORWARD:
            raise ProtocolError(f"unknown RankOp kind {op!r}")
        phase = data.get("phase", PHASE_BATCHED)
        if phase not in _KNOWN_PHASES:
            raise ProtocolError(f"unknown RankOp phase {phase!r}")
        if "tags" not in data or "token_ids" not in data:
            raise ProtocolError("RankOp payload missing required fields")
        tags = data["tags"]
        token_ids = data["token_ids"]
        if not isinstance(tags, list) or not all(isinstance(t, int) for t in tags):
            raise ProtocolError("RankOp 'tags' must be a list of integers")
        if not isinstance(token_ids, list) or not all(
            isinstance(row, list) and all(isinstance(t, int) for t in row)
            for row in token_ids
        ):
            raise ProtocolError("RankOp 'token_ids' must be a list of int lists")
        if len(tags) != len(token_ids):
            raise ProtocolError("RankOp 'tags' and 'token_ids' length mismatch")
        if phase in (PHASE_STANDALONE, PHASE_ADMIT) and len(tags) != 1:
            raise ProtocolError(f"RankOp phase {phase!r} must carry exactly one tag")
        release = data.get("release", [])
        if not isinstance(release, list) or not all(
            isinstance(t, int) for t in release
        ):
            raise ProtocolError("RankOp 'release' must be a list of integers")
        return cls(
            tags=list(tags),
            token_ids=[list(row) for row in token_ids],
            release=list(release),
            done=bool(data.get("done", False)),
            op=op,
            phase=phase,
        )


# -- reply frames (rank 0 -> daemon over stdout) -----------------------------


def chunk_frame(request_id: str, text: str, tokens: int) -> dict[str, Any]:
    """A streamed text chunk for one request."""
    return {"ok": True, "request_id": request_id, "chunk": text, "tokens": tokens}


def done_frame(
    request_id: str,
    *,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    tax: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The terminal frame for one request, carrying usage and the D9 tax."""
    frame: dict[str, Any] = {
        "ok": True,
        "request_id": request_id,
        "done": True,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }
    if tax is not None:
        frame["tax"] = tax
    return frame


def error_frame(
    request_id: str, error: str, *, code: str | None = None, **extra: Any
) -> dict[str, Any]:
    """A terminal failure frame for one request.

    ``code`` is a closed, machine-readable classification (S3 D5): the only
    value defined so far is ``"queue_full"``, which the daemon-side engine
    maps back to ``SchedulerQueueFullError`` so the existing HTTP 503 path
    fires. ``extra`` carries whatever that code needs reconstructed on the
    other end (``current_depth``/``max_depth`` for ``queue_full``).
    """
    frame: dict[str, Any] = {"ok": False, "request_id": request_id, "error": error}
    if code is not None:
        frame["code"] = code
    frame.update(extra)
    return frame


# -- the head->worker command schema (CL2-04) --------------------------------


# The `str` mixin (not `enum.StrEnum`) keeps JSON round-tripping trivial while
# staying importable under mypy's 3.10 target, where StrEnum does not exist.
class CommandKind(str, Enum):  # noqa: UP042
    """The closed set of commands a head may deliver to a worker."""

    SPAWN_RANK = "spawn_rank"
    SWEEP = "sweep"
    PRESENCE = "presence"
    TEARDOWN = "teardown"
    TRANSFER_START = "transfer_start"
    TRANSFER_ROUND = "transfer_round"
    TRANSFER_ABORT = "transfer_abort"


class TransferSource(str, Enum):  # noqa: UP042
    """Where a transfer's bytes come from (D6)."""

    PEER = "peer"
    HF = "hf"


class Backend(str, Enum):  # noqa: UP042
    """The transports a rank may be told to form on."""

    RING = "ring"
    JACCL = "jaccl"


class _CommandBase(BaseModel):
    """Fields every command carries. Unknown fields are rejected, not ignored.

    ``schema_version`` is checked against ``PROTOCOL_VERSION``; ``job_id`` and
    ``step`` are the head-minted replay key (CL2-06) — the worker keeps the
    last applied ``(job_id, step)`` and treats re-delivery as a no-op ack.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    job_id: str
    step: int


class SpawnRankCommand(_CommandBase):
    """Spawn one rank. Carries typed scalars only — never env, path, or cwd.

    ``model_id`` is an identifier the worker resolves against its OWN model
    dirs (CL2-02); ``peers`` are per-rank data-plane addresses the worker
    re-validates against its own settings and never accepts for its own rank
    (CL2-03). No environment crosses the wire (CL2-01).

    ``ibv_devices`` is the jaccl ``MLX_IBV_DEVICES`` matrix (null for ring):
    device names for peers this node cannot itself observe. The worker keeps the
    head's peer rows but OVERWRITES its own rank's row from its OWN
    ``cluster.rdma_device`` — a head-supplied device name for its own rank is
    never trusted (CL2-03), the exact discipline ``peers`` gets for addresses.
    """

    kind: Literal[CommandKind.SPAWN_RANK] = CommandKind.SPAWN_RANK
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    backend: Backend
    model_id: str
    peers: list[str]
    base_port: int = Field(ge=1, le=65535)
    seed: int = 0
    ibv_devices: list[list[str | None]] | None = None


class SweepCommand(_CommandBase):
    """Sweep orphaned ranks whose parent daemon is gone (CL2-08 idempotence)."""

    kind: Literal[CommandKind.SWEEP] = CommandKind.SWEEP


class PresenceCommand(_CommandBase):
    """Ask whether a model id resolves against the worker's own model dirs."""

    kind: Literal[CommandKind.PRESENCE] = CommandKind.PRESENCE
    model_id: str


class TeardownCommand(_CommandBase):
    """Tear the local formation down."""

    kind: Literal[CommandKind.TEARDOWN] = CommandKind.TEARDOWN


class TransferStartCommand(_CommandBase):
    """Open a transfer job for one model (D2/D4).

    ``manifest`` carries raw ``{relative_path, size, sha256}`` dicts -- shape
    and business-rule validation both happen worker-side through
    :func:`omlx.cluster.manifest.validate_received_manifest`, the single
    path every entry must clear before it can reach ``FileManifestEntry``
    (CL5-06's "single validating entry type"); this command layer stays a
    plain container, matching every other command's typed-scalar style.

    Deliberately never carries ``hf_token`` or ``endpoint`` (D7 low): the
    worker uses its own settings for both, and ``extra="forbid"`` makes an
    attempt to add either a rejected command, not a silently-accepted one.
    ``epoch`` binds the job to the worker's CURRENT liveness epoch (CL5-10)
    -- a job replayed against a worker that has since restarted (and so
    minted a new epoch) is refused, never resumed against stale local state.
    ``repair`` opts into re-transferring a destination the worker already
    finds digest-complete (CL5-10 default-refuses that case).
    """

    kind: Literal[CommandKind.TRANSFER_START] = CommandKind.TRANSFER_START
    model_id: str
    manifest: list[dict[str, Any]]
    source: TransferSource
    epoch: str
    repair: bool = False
    # HF-source only (D6). None on a peer transfer.
    hf_repo_id: str | None = None
    hf_revision: str | None = None


class TransferRoundCommand(_CommandBase):
    """Run one round: transfer exactly ``subset`` over a fresh 2-rank ring
    session addressed by ``peers``/``base_port`` (D1/D2). ``peers`` is
    ``[head_address, worker_address]`` in rank order, re-validated by the
    worker against its own link-scope settings exactly like formation's
    ``SpawnRankCommand.peers`` (CL2-03).
    """

    kind: Literal[CommandKind.TRANSFER_ROUND] = CommandKind.TRANSFER_ROUND
    subset: list[str]
    peers: list[str]
    base_port: int = Field(ge=1, le=65535)


class TransferAbortCommand(_CommandBase):
    """Cancel the owned transfer task and discard its staging (D4)."""

    kind: Literal[CommandKind.TRANSFER_ABORT] = CommandKind.TRANSFER_ABORT


Command = (
    SpawnRankCommand
    | SweepCommand
    | PresenceCommand
    | TeardownCommand
    | TransferStartCommand
    | TransferRoundCommand
    | TransferAbortCommand
)

_COMMAND_BY_KIND: dict[str, type[_CommandBase]] = {
    CommandKind.SPAWN_RANK.value: SpawnRankCommand,
    CommandKind.SWEEP.value: SweepCommand,
    CommandKind.PRESENCE.value: PresenceCommand,
    CommandKind.TEARDOWN.value: TeardownCommand,
    CommandKind.TRANSFER_START.value: TransferStartCommand,
    CommandKind.TRANSFER_ROUND.value: TransferRoundCommand,
    CommandKind.TRANSFER_ABORT.value: TransferAbortCommand,
}


def parse_command(data: Any) -> Command:
    """Parse one head->worker command, failing closed on anything unexpected.

    Rejected — with a named error, never silently ignored (CL2-04):

    * a non-object payload,
    * an unknown ``kind``,
    * a ``schema_version`` other than ``PROTOCOL_VERSION`` (E10-bound),
    * any field the per-kind model does not declare (``extra="forbid"``).
    """
    if not isinstance(data, dict):
        raise ProtocolError("command must be a JSON object")
    kind = data.get("kind")
    model_cls = _COMMAND_BY_KIND.get(kind if isinstance(kind, str) else "")
    if model_cls is None:
        raise ProtocolError(f"unknown command kind {kind!r}")
    version = data.get("schema_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"command schema_version {version!r} != protocol {PROTOCOL_VERSION}"
        )
    try:
        return model_cls.model_validate(data)  # type: ignore[return-value]
    except ValidationError as exc:
        raise ProtocolError(f"invalid {kind} command: {exc}") from exc


def command_to_wire(command: Command) -> dict[str, Any]:
    """Serialise a command to the JSON dict the heartbeat response carries."""
    return command.model_dump(mode="json")


def make_job_update(
    job_id: str, step: int, *, status: str, **extra: Any
) -> dict[str, Any]:
    """Build a worker->head job update.

    The head attributes every update to the AUTHENTICATED member and ignores
    any member/rank id carried here (CL2-07), so this shape deliberately does
    not name a member. ``status`` is one of ``accepted``, ``spawned``,
    ``present``, ``absent``, ``swept``, ``torn_down``, ``error``, ``rejected``.
    """
    update: dict[str, Any] = {"job_id": job_id, "step": step, "status": status}
    update.update(extra)
    return update


def make_transfer_update(
    job_id: str, step: int, *, status: str, **extra: Any
) -> dict[str, Any]:
    """Build a worker->head transfer update (D1b/D2).

    Rides the heartbeat's ``transfer_updates`` field, a sibling of
    ``job_updates`` -- NOT an ack: a TRANSFER_ROUND command's immediate ack
    only confirms the round started, and its actual outcome (arriving
    possibly minutes later) is one of these instead, so nothing is dropped
    for being unmatched to a single-shot ack future. ``status`` is one of
    ``accepted``, ``rejected``, ``have`` (a bare presence report), ``round_done``,
    ``round_error``, ``aborted``, ``hf_progress``, ``done``, ``error``.
    """
    update: dict[str, Any] = {"job_id": job_id, "step": step, "status": status}
    update.update(extra)
    return update


# -- stop-sequence straddle handling -----------------------------------------


class StopTextBuffer:
    """Streams text out while holding back anything that could still become a
    stop sequence.

    A stop sequence can straddle two tokens, so text cannot be released the
    moment it is decoded: ``"<|im_"`` looks harmless until the next token turns
    it into ``"<|im_end|>"``. This holds back the longest suffix that is a
    prefix of any stop string, releases everything before it, and on a real hit
    reports the text truncated at the match.

    With no stop strings configured this is a pass-through and costs one branch.
    """

    def __init__(self, stops: list[str]) -> None:
        self._stops = [s for s in stops if s]
        self._pending = ""
        self._released = ""
        self.hit: str | None = None

    @property
    def text(self) -> str:
        """Everything released so far, excluding held-back text."""
        return self._released

    def push(self, chunk: str) -> str:
        """Add decoded text; return the part that is safe to emit now."""
        if not self._stops:
            self._released += chunk
            return chunk

        self._pending += chunk
        for stop in self._stops:
            index = self._pending.find(stop)
            if index != -1:
                self.hit = stop
                emit = self._pending[:index]
                self._pending = ""
                self._released += emit
                return emit

        keep = self._suffix_prefix_length()
        emit = self._pending[: len(self._pending) - keep] if keep else self._pending
        self._pending = self._pending[len(self._pending) - keep :] if keep else ""
        self._released += emit
        return emit

    def flush(self) -> str:
        """Release the held-back tail. Only correct once no more tokens come."""
        if self.hit is not None:
            self._pending = ""
            return ""
        emit, self._pending = self._pending, ""
        self._released += emit
        return emit

    def _suffix_prefix_length(self) -> int:
        """Longest suffix of the buffer that could still grow into a stop."""
        longest = max(len(s) for s in self._stops)
        window = min(len(self._pending), longest - 1)
        for size in range(window, 0, -1):
            tail = self._pending[-size:]
            if any(stop.startswith(tail) for stop in self._stops):
                return size
        return 0
