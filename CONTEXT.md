# Session Context — 2026-07-29 (evening) — row 4 fixed; it was never a cluster defect

**Status:** the S3 P3 row-4 carry-forward (#10) is fixed and committed at `87fb0e2e` on
`feat/cluster-v1`. **Row 4 itself is still open** — it was measured on the live 2-Mac rig on
both backends, and only a rig re-run closes it. S3 (#7) stays open behind it.

## The finding that changed the shape of the fix

The recorded diagnosis said the cluster route returns a `StreamingResponse` before the
generate command reaches rank 0, so `server.py:708`'s 503 handler can't fire. That fact is
true and was not the cause.

**Single-node `stream: true` degrades identically, and always has.** `Scheduler.add_request`
raises from inside the route's response generator (`batched.py:817`); starlette emits
`http.response.start` before it iterates; so the 503 handler could only ever answer
`stream: false`. Nothing anywhere checked queue depth in preflight — `preflight_or_raise` is
the prefill *memory* guard, and it short-circuits entirely when that guard is off (the
default). The cluster path was matching single-node semantics exactly, which is what spec-S3
asked of it.

Settled by reading, not inference: `batched.py:817` sits inside an async generator;
`engine_core.py:624` re-raises after cleanup; `preflight_or_raise` (`scheduler.py:8074`) opens
with `if not self._prefill_memory_guard: return`. A TestClient probe was started and abandoned
— it was measuring the stub's surface, not the code.

## What landed (`87fb0e2e`)

Fix goes at the preflight seam all four LLM routes already await (`server.py:3042, 3546, 5386,
5873`), so **no `server.py` change was needed**.

| Piece | Where |
| --- | --- |
| `waiting_queue_capacity(max_num_seqs)` — one cap definition, three callers | `scheduler.py` |
| `Scheduler.preflight_queue_or_raise()` | `scheduler.py` |
| `BaseEngine._preflight_queue()` — shared, silent when no scheduler | `engine/base.py` |
| Called first + unconditionally in both preflights | `engine/batched.py`, `engine/vlm.py` |
| Head-side gate on rank-0 in-flight count | `cluster/engine.py` |
| `rank_max_num_seqs()` / `rank_inflight_capacity()` | `cluster/scheduler_config.py` |
| `logger.warning` on the backstop `queue_full` frame | `cluster/engine.py` |

Deliberate choices worth not re-litigating:

- The queue check is **ahead of tokenization and separate from the memory path**. Folding it
  into `preflight_or_raise` would make it dead whenever the memory guard is off; putting it
  after the tokenize `try/except` would make it dead on tokenizer errors. There is a test
  pinning each.
- `ClusterEngine` gates on `len(self._pending)` rather than asking rank 0. It is rank 0's only
  submitter, so that count *is* what rank 0 holds, modulo frames in flight. Approximate by
  construction — the preflight→submit race stays open, `add_request` stays the authority, and
  the in-stream backstop is now logged.
- **No reservation protocol.** The cap is backpressure, not a correctness invariant, and the
  acceptance assertion is index-free. Over-admitting by a few under a burst is immaterial.
- DFlash needs no change: primary mode bypasses the scheduler, fallback mode delegates to the
  fallback engine's preflight.

## Verification

| Gate | Result |
| --- | --- |
| `pytest -m cluster` | **21 passed** (3:18) — matches baseline exactly |
| Default unit gate | **7527 passed, 2 failed** — both the known GLM numerical ones; zero delta |
| black / ruff / mypy | **zero delta** vs `402f3b1d` (mypy 1068/136 both sides) |
| Falsifiability | 20 new tests run against stashed source: **17 fail without the fix** |

On those 17: the 3 that pass are `test_no_scheduler_is_not_an_error` and **both**
`TestStreamingChatReaches503` cases. Those two mock `preflight_chat` to raise, so they prove
the route can carry a 503 on `stream: true` — they do **not** prove the streaming path was
broken before. The engine-level tests in `test_engine_preflight.py` carry that weight. Don't
over-credit the two test names.

**Fresh-context independence was not obtained, third session running.** Every gate here was
run by the main session. The session-level "do not call the AgentTool unless the user
requested it" is more specific than the CLAUDE.md orchestration policy, so no `verifier` was
dispatched — but that means the evidence above is self-reported, as it was in the two prior
sessions where the dispatched verifier went idle. Worth solving as its own thread.

## Next

1. **Rig re-measure of row 4** — the named next action, deliberately not run unsupervised.
   Same `flood` recipe (41 concurrent at `max_num_seqs=8`, `max_tokens` large enough that
   nothing completes), both backends, asserting a real HTTP 503. Watch the traps in
   [[gotcha-omlx-cluster-rig-operation]]: use `.venv/bin/omlx` (the PATH `omlx` is a uv tool
   that silently wipes cluster settings), scrub ghost members before forming, and **sync the
   Studio checkout to `87fb0e2e` by hand** — E10 does not check it, so a stale worker joins
   cleanly and runs mismatched TP code.
2. **#7 — S3 completion**, once row 4 closes.
3. **Separate finding, not fixed:** `server.py:3528` and `:3532` dereference `ms` behind a
   `_server_state.settings_manager` truthiness check —
   `elif _server_state.settings_manager and ms.specprefill_keep_pct is not None`. `ms` is
   `None` whenever the model has no per-model settings entry, so that is a live 500 on
   `/v1/chat/completions`. Found because it broke a new test under suite ordering; left alone
   per surgical-changes rule. Worth its own small fix.
4. Optional: dedicated ring-vs-jaccl comparison, driven by the TTFT gap (4.0–12.4s vs
   1.3–5.8s), not the throughput ratio.

**Rig untouched this session** — no daemons started, nothing ssh'd. Daily-driver `:8899`
(PID 87617) and Studio `:8888/:8889` were never approached.

---
