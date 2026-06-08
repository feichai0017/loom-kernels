#!/usr/bin/env python3
"""vLLM -> QuillCache KV-events bridge.

vLLM publishes KV cache events (BlockStored / BlockRemoved / AllBlocksCleared)
over ZMQ (msgpack-encoded) when started with `--kv-events-config`. This bridge
subscribes to that stream and forwards each batch to the QuillCache gateway's
`/v1/kv-events` endpoint as vendor-neutral JSON, turning a real vLLM into a
QuillCache-observable engine.

Run it co-located with vLLM (it must reach the ZMQ endpoint). Deps:
    pip install pyzmq msgpack requests

Example:
    python bridge/vllm_kv_bridge.py \
        --zmq tcp://127.0.0.1:5557 --topic "" \
        --gateway http://127.0.0.1:8080 --engine-id vllm-a

NOTE: vLLM's exact KV-event field names vary by version. `translate()` maps the
common shape; adjust it if your vLLM build differs (print --debug to inspect).
"""
import argparse
import time

try:
    import msgpack
    import requests
    import zmq
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(f"missing dependency ({exc}); run: pip install pyzmq msgpack requests")


def translate(engine_id, payload, debug=False):
    """Translate one decoded vLLM KV-event payload into a QuillCache KvEventBatch."""
    if debug:
        print("raw payload:", payload)
    raw_events = payload.get("events", payload) if isinstance(payload, dict) else payload
    if isinstance(raw_events, dict):
        raw_events = [raw_events]

    events = []
    for ev in raw_events or []:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("type") or ev.get("event") or "").lower()
        block_hashes = [str(h) for h in ev.get("block_hashes", [])]
        if "clear" in kind:
            events.append({"type": "all_blocks_cleared"})
        elif "remov" in kind:
            events.append(
                {"type": "block_removed", "block_hashes": block_hashes, "medium": ev.get("medium")}
            )
        elif "stor" in kind or block_hashes:
            parent = ev.get("parent_block_hash")
            events.append(
                {
                    "type": "block_stored",
                    "block_hashes": block_hashes,
                    "parent_block_hash": (str(parent) if parent is not None else None),
                    "token_ids": list(ev.get("token_ids", [])),
                    "block_size": ev.get("block_size") or len(ev.get("token_ids", [])) or 16,
                    "medium": ev.get("medium"),
                    "lora_name": ev.get("lora_id") or ev.get("lora_name"),
                }
            )
    return {"engine_id": engine_id, "events": events}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zmq", default="tcp://127.0.0.1:5557", help="vLLM KV-events ZMQ endpoint")
    ap.add_argument("--topic", default="", help="ZMQ SUB topic filter")
    ap.add_argument("--gateway", default="http://127.0.0.1:8080")
    ap.add_argument("--engine-id", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.zmq)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    url = args.gateway.rstrip("/") + "/v1/kv-events"
    print(f"bridge: {args.zmq} (topic={args.topic!r}) -> {url} as engine {args.engine_id!r}")

    sent = 0
    while True:
        try:
            parts = sock.recv_multipart()
            payload = msgpack.unpackb(parts[-1], raw=False)  # last frame is the payload
            batch = translate(args.engine_id, payload, args.debug)
            if not batch["events"]:
                continue
            requests.post(url, json=batch, timeout=5)
            sent += 1
            if sent % 50 == 0:
                print(f"forwarded {sent} batches")
        except KeyboardInterrupt:
            print(f"\nstopped after {sent} batches")
            break
        except Exception as exc:  # keep the bridge alive across transient errors
            print("bridge error:", exc)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
