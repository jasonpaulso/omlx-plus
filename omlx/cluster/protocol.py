# SPDX-License-Identifier: Apache-2.0
"""The wire format between a daemon and its rank-0 worker.

Kept in its own module because three parties depend on it and none of them
should import the others: the daemon-side engine writes commands, the worker
reads them, and the tests exercise both without spawning mlx.

Two channels, not one. Commands and their replies flow over the worker's
stdin/stdout, which are busy for the whole of a generation - rank 0 is inside
the decode loop and cannot go back and read another line. Anything that must
reach a *running* generation therefore travels over a second pipe, which the
worker polls between tokens. Today that is only ``abort``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Commands, sent over stdin and broadcast verbatim by rank 0.
CMD_LOAD = "load"
CMD_GENERATE = "generate"
CMD_PING = "ping"
CMD_SHUTDOWN = "shutdown"

# Out-of-band, sent over the control pipe while a generation is running.
SIGNAL_ABORT = "abort"

# Why a decode loop ended. Rank 0 decides, and the value is agreed through the
# collective, so every rank leaves the loop on the same iteration.
STEP_CONTINUE = 0
STEP_EOS = 1
STEP_STOP_TEXT = 2
STEP_ABORT = 3

FINISH_REASON = {
    STEP_EOS: "stop",
    STEP_STOP_TEXT: "stop",
    STEP_ABORT: "abort",
}


@dataclass
class GenerationSpec:
    """One generation request, as it crosses the pipe.

    Prompt tokens are sent already encoded. The daemon holds the tokenizer it
    used to apply the chat template, and re-encoding the rendered string inside
    the worker would be a second chance to disagree with it for nothing.
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


class StopTextBuffer:
    """Streams text out while holding back anything that could still become a
    stop sequence.

    A stop sequence can straddle two tokens, so text cannot be released the
    moment it is decoded: `"<|im_"` looks harmless until the next token turns it
    into `"<|im_end|>"`. This holds back the longest suffix that is a prefix of
    any stop string, releases everything before it, and on a real hit reports
    the text truncated at the match.

    With no stop strings configured this is a pass-through and costs one
    branch.
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
