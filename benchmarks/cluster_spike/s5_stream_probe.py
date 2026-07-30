#!/usr/bin/env python3
"""Stream N tokens from an omlx endpoint; print usage-based token count.

Usage: s5_stream_probe.py <base_url> <api_key> <model_id> [max_tokens]
Counts via the final usage chunk (stream_options.include_usage) with a
chars/4 floor fallback — SSE event counting under-counts coalesced tokens.
Also checks the `error` key before the `choices` guard (reasoning_content
models stream there).
"""
import json
import sys
import urllib.request

base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 48

body = {
    "model": model,
    "messages": [{"role": "user", "content": "Count from 1 to 30, digits only."}],
    "max_tokens": max_tok,
    "stream": True,
    "stream_options": {"include_usage": True},
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
usage = None
chars = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("error"):
            print(json.dumps({"error": chunk["error"]}))
            sys.exit(1)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            chars += len(d.get("content") or "") + len(d.get("reasoning_content") or "")
completion = (usage or {}).get("completion_tokens") or 0
print(json.dumps({"completion_tokens": max(completion, chars // 4), "usage": usage}))
