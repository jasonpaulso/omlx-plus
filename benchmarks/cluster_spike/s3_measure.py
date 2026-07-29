#!/usr/bin/env python3
"""S3 P3 live-rig measurement driver.

Captures RAW SSE token-arrival timestamps and dumps them to JSON. It does not
compute the D7 gate -- see s3_compute.py, which applies the formula to the raw
dump. Keeping capture and arithmetic apart means the gate can be recomputed
without re-running the rig.

Stdlib only, so it runs from any checkout without a venv.

Subcommands:
  single      one request                          -> raw timestamps
  concurrent  N identical requests at once         -> raw timestamps
  abort       two requests, disconnect one midway  -> raw timestamps + outcome
  flood       many concurrent requests             -> status codes (queue-full)
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

CLOCK = time.perf_counter


def _post_stream(url, api_key, body, timeout=600):
    """Open a streaming POST. Returns (status, response) or (status, None)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context() if url.startswith("https") else None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        return resp.status, resp
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        return exc.code, detail
    except (urllib.error.URLError, OSError) as exc:
        # Connection refused/reset must not kill the run: sibling threads are
        # still capturing timestamps we need.
        return None, f"{type(exc).__name__}: {exc}"


def stream_request(url, api_key, body, label, abort_after=None, timeout=600):
    """Drive one SSE request, recording an arrival timestamp per content token.

    abort_after: close the connection after this many content chunks (client
    disconnect), to exercise abort-mid-batch.
    """
    rec = {
        "label": label,
        "status": None,
        "error": None,
        "submit_t": CLOCK(),
        "arrivals": [],  # perf_counter at each content-bearing chunk
        "text_len": 0,
        "usage": None,
        "aborted": False,
        "finish_reason": None,
    }
    status, resp = _post_stream(url, api_key, body, timeout=timeout)
    rec["status"] = status
    if status != 200:
        rec["error"] = resp if isinstance(resp, str) else "non-200"
        return rec
    try:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                rec["usage"] = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if choices[0].get("finish_reason"):
                rec["finish_reason"] = choices[0]["finish_reason"]
            if piece:
                rec["arrivals"].append(CLOCK())
                rec["text_len"] += len(piece)
                if abort_after is not None and len(rec["arrivals"]) >= abort_after:
                    rec["aborted"] = True
                    break
    except Exception as exc:  # noqa: BLE001 - record whatever the rig throws
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
    return rec


def build_body(args, prompt, max_tokens=None, stream=True):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def run_parallel(fns):
    results = [None] * len(fns)
    threads = []

    def wrap(i, fn):
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001 - never lose sibling captures
            results[i] = {
                "label": f"thread{i}",
                "status": None,
                "error": f"{type(exc).__name__}: {exc}",
                "arrivals": [],
                "usage": None,
                "aborted": False,
                "finish_reason": None,
            }

    for i, fn in enumerate(fns):
        t = threading.Thread(target=wrap, args=(i, fn), daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "concurrent", "abort", "flood"])
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8910")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--abort-after", type=int, default=8)
    ap.add_argument("--backend", default="", help="recorded as metadata only")
    ap.add_argument("--out", required=True, help="raw JSON dump path")
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8") as fh:
        prompt = fh.read()

    url = args.url.rstrip("/") + "/v1/chat/completions"
    meta = {
        "mode": args.mode,
        "backend": args.backend,
        "model": args.model,
        "prompt_file": args.prompt_file,
        "prompt_chars": len(prompt),
        "max_tokens": args.max_tokens,
        "n": args.n,
        "wall_start": time.time(),
    }

    if args.mode == "single":
        records = [stream_request(url, args.api_key, build_body(args, prompt), "single")]

    elif args.mode == "concurrent":
        body = build_body(args, prompt)
        records = run_parallel(
            [
                (lambda i=i: stream_request(url, args.api_key, dict(body), f"c{i}"))
                for i in range(args.n)
            ]
        )

    elif args.mode == "abort":
        body = build_body(args, prompt)
        records = run_parallel(
            [
                lambda: stream_request(
                    url, args.api_key, dict(body), "victim", abort_after=args.abort_after
                ),
                lambda: stream_request(url, args.api_key, dict(body), "survivor"),
            ]
        )

    elif args.mode == "flood":
        # Queue-full: long max_tokens so nothing completes during the burst.
        body = build_body(args, prompt, max_tokens=4096)
        records = run_parallel(
            [
                (
                    lambda i=i: stream_request(
                        url,
                        args.api_key,
                        dict(body),
                        f"f{i}",
                        abort_after=3,  # stop reading once streaming is proven
                        timeout=300,
                    )
                )
                for i in range(args.n)
            ]
        )

    meta["wall_end"] = time.time()
    dump = {"meta": meta, "records": records}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=2)

    # Terse operator-facing summary; the gate lives in s3_compute.py.
    ok = sum(1 for r in records if r["status"] == 200)
    codes = {}
    for r in records:
        codes[r["status"]] = codes.get(r["status"], 0) + 1
    print(f"mode={args.mode} backend={args.backend} n={len(records)} ok={ok}")
    print(f"status codes: {codes}")
    for r in records:
        print(
            f"  {r['label']}: status={r['status']} tokens={len(r['arrivals'])} "
            f"aborted={r['aborted']} finish={r['finish_reason']} "
            f"usage={r['usage']} err={(r['error'] or '')[:120]}"
        )
    print(f"raw dump -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
