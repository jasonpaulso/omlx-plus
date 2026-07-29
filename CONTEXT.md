# Session Context — 2026-07-29 (evening) — row 4 re-measured on the rig: STILL FAILS

**Status:** the defect was diagnosed correctly (it was never cluster-specific), a fix landed
at `87fb0e2e`, and a second unrelated route-guard defect was fixed at `6a64e7c0`. Then the rig
run showed **the fix does not close row 4**: the gate counts requests that have not entered
the engine yet, so it never engages on a cold burst. S3 (#7) stays open. Closing row 4 needs a
design change (slot reservation at preflight), not a patch — details in "Rig re-measure" and
"Next" below.

Half of row 4 *is* closed: the rejection is now logged head-side.

Branch tip: see `git log`. **Rig left clean** — daemons down, 8910/8911 free, `backend=ring`
and roles intact on both nodes, member scrubbed, model unloaded; daily-driver `:8899`
(PID 87617) and Studio `:8888`/`:8889` untouched throughout.

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
  submitter, so that count *is* what rank 0 holds, modulo frames in flight. **Disproven by the
  rig run below** — `_pending` is filled inside the response generator, so at preflight time
  on a cold burst it is empty and the gate never engages.
- ~~**No reservation protocol.** The cap is backpressure, not a correctness invariant, and the
  acceptance assertion is index-free. Over-admitting by a few under a burst is immaterial.~~
  **Wrong, and it is the reason row 4 still fails.** On a cold burst the gate does not
  over-admit by a few — it does not fire at all. Reservation is the fix.
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

## Rig re-measure — ran it, and row 4 still FAILS

Ring, at `b3d39ad6`, both checkouts SHA-matched and both venvs confirmed importing the new
symbols. `negotiated_backend: ring` == requested, smoke generation served before the burst.

```
status codes {200: 41} | HTTP 503: 0 | streamed: 40 | in-stream queue_full: 1  -> FAIL
```

Identical to pre-fix. **The reason is in the fix, not the rig.**
`ClusterEngine._preflight_queue` gates on `len(self._pending)`, but `_pending` is filled at
`cluster/engine.py:350` — inside `stream_generate`, which runs only *after* the route commits
to the `StreamingResponse`. On a cold burst all 41 preflight before any generator starts, so
the counter is empty and the gate never fires. Head log confirms: zero preflight 503s, exactly
one backstop warning.

Single-node has the same shape — `Scheduler.waiting` is filled by `add_request`, also inside
the generator. So `preflight_queue_or_raise` engages only when the queue is **already** at cap
when a later request preflights (sustained backpressure, the production shape), never on a
cold burst — which is exactly row 4's recipe. The unit tests pass because they pre-fill the
queue first: a real behaviour, just not the measured one.

**What did close:** the rejection is now logged head-side, the second half of row 4's
complaint. Fired once, with request id and `32/32`.

jaccl deliberately not run: the defect is head-side and cannot vary with the collective
backend, so ring is decisive.

## Next

1. **Decide how to close row 4.** It needs the head to *reserve* a slot at preflight instead
   of counting generators that haven't started. Reservation was consciously rejected during
   design ("over-admitting by a few is immaterial") — wrong for a cold burst, where the gate
   doesn't over-admit, it doesn't engage. Needs a preflight→submit identity; the route's
   `request_id` comes from an `x-request-id` header that is usually absent. Design change,
   your call.
2. **#7 — S3 completion**, still blocked behind it.
3. Optional: dedicated ring-vs-jaccl comparison, driven by the TTFT gap (4.0–12.4s vs
   1.3–5.8s), not the throughput ratio.

**Rig operation, hard-won:** start the worker **attached** —
`ssh -o ServerAliveInterval=15 ... 'cd ... && ulimit -n 65536 && exec .venv/bin/omlx serve ...'`
from a background call. A `nohup`-orphaned worker fails every control-plane call with
"All connection attempts failed" while curl/httpx/ClusterClient from the same host all
succeed; cause still unknown (fd limits, uvloop, TCC, fork, Metal init, firewalls and
interface all ruled out; a Tailscale network extension is the untested lead). Also: the
api_key is **not** shared between nodes, there is no `omlx cluster join` verb (the worker
calls its own `/v1/cluster/local/join`), and head LAN is `192.168.4.68`.

## Also fixed (`6a64e7c0`) — and a correction to how I first reported it

`/v1/chat/completions` guarded its SpecPrefill fallbacks on the wrong object:
`elif _server_state.settings_manager and ms.specprefill_keep_pct is not None` dereferences
`ms`, not the manager. Now `elif ms and ...`, matching `if ms:` eight lines above.

**I first reported this as "a live 500". That was wrong**, and wrong in a specific way worth
remembering: I inferred it from a downstream symptom (it broke one of my tests under suite
ordering) instead of checking the claim. Reading it afterwards: `ModelSettingsManager
.get_settings` returns a default `ModelSettings()` rather than `None` for an unknown model,
and an empty model id is rejected by engine resolution long before this line — so no
production request is known to reach it. It is a real defect, masked by two invariants that
live in other files and that this code has no business relying on. Fixed with a regression
test that fails pre-fix; severity stated honestly in the commit.

**Rig untouched this session** — no daemons started, nothing ssh'd. Daily-driver `:8899`
(PID 87617) and Studio `:8888/:8889` were never approached.

---
