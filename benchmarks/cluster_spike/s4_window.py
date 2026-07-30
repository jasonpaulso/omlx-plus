#!/usr/bin/env python3
"""Row-1 concurrent window: 3 streams across 3 surfaces, raw SSE arrival capture.

Counts BOTH delta.content and delta.reasoning_content tokens (MiniMax streams
reasoning_content; a content-only counter records one arrival for 128 tokens).
Also captures in-stream error payloads (no `choices` key) as errors.
"""
import asyncio
import json
import sys
import time

import httpx

SP = "/private/tmp/claude-501/-Users-jasonschulz-Developer-Runners-omlx/1879156f-8130-4401-8c13-ce2b9b199c07/scratchpad"


async def stream_one(client, surface, base, key, model, max_tokens=96):
    rec = {"surface": surface, "started_at": None, "first_token_at": None,
           "completed_at": None, "completion_tokens": 0, "content_chars": 0, "usage_tokens": None, "error": None}
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Count from 1 to 30, one number per line."}]}
    rec["started_at"] = time.monotonic()
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"},
                                 json=body, timeout=600) as r:
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}"
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "choices" not in d:
                    rec["error"] = f"in-stream error: {json.dumps(d)[:200]}"
                    return rec
                if d.get("usage"):
                    rec["usage_tokens"] = (d["usage"].get("completion_tokens")
                                           or d["usage"].get("output_tokens"))
                if not d["choices"]:
                    continue
                delta = d["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.monotonic()
                    if rec["first_token_at"] is None:
                        rec["first_token_at"] = now
                    rec["completion_tokens"] += 1
                    rec["content_chars"] += len(delta.get("content") or "") + len(delta.get("reasoning_content") or "")
        rec["completed_at"] = time.monotonic()
        # SSE events under-count when the server coalesces tokens; prefer the
        # stream's own usage chunk, else a chars/4 floor, else the event count.
        rec["completion_tokens"] = max(rec["completion_tokens"],
                                       rec["usage_tokens"] or 0,
                                       rec["content_chars"] // 4)
    except Exception as e:  # noqa: BLE001 — raw capture, record everything
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


async def main():
    head_key, worker_key = sys.argv[1], sys.argv[2]
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            stream_one(client, "head-minimax", "http://127.0.0.1:8910", head_key, "MiniMax-M2.7-3bit"),
            stream_one(client, "head-llama", "http://127.0.0.1:8910", head_key, "Llama-3.2-1B-Instruct-4bit"),
            stream_one(client, "worker-llama", "http://192.168.5.28:8911", worker_key, "Llama-3.2-1B-Instruct-4bit"),
        )
    out = {"streams": list(results)}
    with open(f"{SP}/s4b_row1_window.json", "w") as f:
        json.dump(out, f, indent=2)
    for s in results:
        dur = (s["completed_at"] - s["started_at"]) if s["completed_at"] else None
        print(s["surface"], "tokens:", s["completion_tokens"], "err:", s["error"],
              "dur:", round(dur, 2) if dur else None)


asyncio.run(main())
