#!/usr/bin/env python3
"""Measure the QuillCache gateway's own hot-path overhead.

A control plane sits in front of every request, so the question an interviewer
asks first is: *what does the control plane itself cost?* This driver answers it
by sending the same shared-prefix streaming workload two ways —

  1. **direct**  : client -> mock engine
  2. **gateway** : client -> QuillCache gateway -> mock engine

against the *same* near-instant mock engine (tools/mock_engine.py with ttft/itl
0). The difference in latency is QuillCache's added cost: request parse, prompt
-> block-hash derivation, the control-plane routing decision + residency
write-back, decision headers, and the streaming proxy. Throughput (req/s) shows
whether the gateway is a bottleneck. Standard library only.

    # term 1: instant engine        python tools/mock_engine.py --port 9000
    # term 2: gateway -> :9000       cargo run -- gateway --config <cfg with base_url :9000>
    # term 3:
    python bench/bench_gateway.py \
        --direct-url http://127.0.0.1:9000 \
        --gateway-url http://127.0.0.1:8080 \
        --model mock --requests 2000 --concurrency 32
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SHARED_SYSTEM_PROMPT = "You are QuillCache-Bench, a precise assistant. " * 64


def one_request(base_url, model, idx, max_tokens):
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question {idx}: reply briefly."},
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
                        if delta.get("content") or chunk["choices"][0].get("text"):
                            ttft = (time.perf_counter() - start) * 1000.0
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
        return {"ok": True, "ttft": ttft, "total": (time.perf_counter() - start) * 1000.0}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def run_phase(name, base_url, args):
    # Warmup (not measured): prime connections and the gateway's residency index.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(
            lambda i: one_request(base_url, args.model, i, args.max_tokens),
            range(args.warmup),
        ))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda i: one_request(base_url, args.model, i, args.max_tokens),
            range(args.requests),
        ))
    wall = time.perf_counter() - start

    ok = [r for r in results if r.get("ok")]
    totals = [r["total"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    summary = {
        "name": name,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "throughput_rps": len(ok) / wall if wall > 0 else 0.0,
        "total": {q: pct(totals, p) for q, p in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999))},
        "ttft": {q: pct(ttfts, p) for q, p in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999))},
        "total_mean": sum(totals) / len(totals) if totals else 0.0,
        "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
    }
    if ok and ok[0].get("error"):
        summary["first_error"] = ok[0]["error"]
    if summary["failed"]:
        summary["first_error"] = next((r.get("error") for r in results if not r.get("ok")), None)
    return summary


def fmt(s):
    return (
        f"  {s['name']:8s}  ok {s['ok']:5d}  fail {s['failed']:4d}  "
        f"{s['throughput_rps']:8.1f} req/s | "
        f"total ms p50 {s['total']['p50']:7.2f}  p99 {s['total']['p99']:8.2f}  "
        f"p999 {s['total']['p999']:8.2f}  mean {s['total_mean']:7.2f} | "
        f"TTFT ms p50 {s['ttft']['p50']:7.2f}  p99 {s['ttft']['p99']:8.2f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direct-url", default="http://127.0.0.1:9000")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    direct = run_phase("direct", args.direct_url, args)
    gateway = run_phase("gateway", args.gateway_url, args)

    overhead = {
        q: gateway["total"][q] - direct["total"][q] for q in ("p50", "p99", "p999")
    }
    overhead_mean = gateway["total_mean"] - direct["total_mean"]

    if args.json:
        print(json.dumps(
            {"direct": direct, "gateway": gateway,
             "added_overhead_ms": {**overhead, "mean": overhead_mean}},
            indent=2,
        ))
        return

    print(f"\nQuillCache gateway overhead  (reqs={args.requests} concurrency={args.concurrency} "
          f"max_tokens={args.max_tokens}, same mock engine both paths)")
    print(fmt(direct))
    print(fmt(gateway))
    print(f"\n  => QuillCache added latency:  p50 {overhead['p50']:+.2f} ms   "
          f"p99 {overhead['p99']:+.2f} ms   p999 {overhead['p999']:+.2f} ms   mean {overhead_mean:+.2f} ms")
    if direct["throughput_rps"] > 0:
        ratio = 100.0 * gateway["throughput_rps"] / direct["throughput_rps"]
        print(f"     gateway sustains {ratio:.1f}% of direct throughput "
              f"({gateway['throughput_rps']:.0f} vs {direct['throughput_rps']:.0f} req/s)")
    for s in (direct, gateway):
        if s.get("first_error"):
            print(f"     {s['name']} first error: {s['first_error']}")


if __name__ == "__main__":
    main()
