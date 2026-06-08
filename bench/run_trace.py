#!/usr/bin/env python3
"""QuillCache trace runner.

Drive the QuillCache gateway with a shared-prefix chat workload and measure real
TTFT / total latency plus QuillCache's routing decision headers (`x-quillcache-*`).
Works against any OpenAI-compatible endpoint behind the gateway, or directly
against vLLM for an A/B baseline. Standard library only — no pip install.

Example:
    python bench/run_trace.py --base-url http://127.0.0.1:8080 \
        --model Qwen/Qwen2.5-0.5B-Instruct --requests 64 --concurrency 8
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# A long, identical system prompt across requests -> a big shared prefix, which
# is exactly what cache-aware routing should exploit.
SHARED_SYSTEM_PROMPT = "You are QuillCache-Bench, a precise assistant. " * 64


def one_request(base_url, model, idx, max_tokens):
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question {idx}: reply in one short sentence."},
        ],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft = None
    headers = {}
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            headers = {
                k.lower(): v
                for k, v in resp.headers.items()
                if k.lower().startswith("x-quillcache-")
            }
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                if ttft is None:
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            ttft = time.perf_counter() - start
                    except json.JSONDecodeError:
                        pass
        return {"ok": True, "ttft": ttft, "total": time.perf_counter() - start, "headers": headers}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return {"ok": False, "error": str(exc)}


def pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.base_url, args.model, i, args.max_tokens)
            for i in range(args.requests)
        ]
        for fut in futures:
            results.append(fut.result())

    ok = [r for r in results if r.get("ok")]
    ttfts = [r["ttft"] * 1000 for r in ok if r.get("ttft") is not None]
    totals = [r["total"] * 1000 for r in ok]
    hits = sum(1 for r in ok if r["headers"].get("x-quillcache-local-hits", "0") not in ("0", ""))

    print(f"requests: {len(results)}  ok: {len(ok)}  failed: {len(results) - len(ok)}")
    if ttfts:
        print(f"TTFT  ms  p50 {pct(ttfts, 0.5):.1f}  p99 {pct(ttfts, 0.99):.1f}  mean {sum(ttfts) / len(ttfts):.1f}")
    if totals:
        print(f"total ms  p50 {pct(totals, 0.5):.1f}  p99 {pct(totals, 0.99):.1f}")
    print(f"responses with a QuillCache local-hit header: {hits}/{len(ok)}")
    engines = Counter(r["headers"].get("x-quillcache-engine-id", "?") for r in ok)
    if engines:
        print("engine distribution: " + "  ".join(f"{e}: {n}" for e, n in sorted(engines.items())))
    for r in ok:
        if r["headers"]:
            print("sample decision headers:", json.dumps(r["headers"]))
            break
    if len(ok) < len(results):
        for r in results:
            if not r.get("ok"):
                print("first error:", r.get("error"))
                break


if __name__ == "__main__":
    main()
