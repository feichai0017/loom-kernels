#!/usr/bin/env python3
"""A minimal OpenAI-compatible mock inference engine — stdlib only, no GPU.

It speaks just enough of the vLLM/OpenAI surface for QuillCache to proxy it:
streaming `/v1/chat/completions` and `/v1/completions`, plus `/healthz`. Token
output is synthetic; timing is configurable so you can isolate one variable:

  --ttft-ms N   delay before the first streamed token (default 0 = instant)
  --itl-ms  N   inter-token latency between subsequent tokens (default 0)
  --tokens  N   tokens to emit when the request doesn't cap max_tokens

With the defaults (0/0) the engine is effectively instant, so any latency a
client observes *through the QuillCache gateway* is the control plane's own
overhead — which is exactly what bench/bench_gateway.py measures. Threaded so it
serves concurrent load.

    python tools/mock_engine.py --port 9000                 # instant engine
    python tools/mock_engine.py --port 9000 --ttft-ms 200   # mimic a real TTFT
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/healthz", "/health", "/v1/models"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        is_chat = self.path.rstrip("/").endswith("/chat/completions")
        if not (is_chat or self.path.rstrip("/").endswith("/completions")):
            self._json(404, {"error": "not found"})
            return

        max_tokens = body.get("max_tokens") or ARGS.tokens
        n_tokens = max(1, min(int(max_tokens), 256))
        model = body.get("model", "mock-engine")
        if body.get("stream"):
            self._stream(is_chat, model, n_tokens)
        else:
            self._whole(is_chat, model, n_tokens)

    def _stream(self, is_chat, model, n_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        if ARGS.ttft_ms:
            time.sleep(ARGS.ttft_ms / 1000.0)
        for i in range(n_tokens):
            if i and ARGS.itl_ms:
                time.sleep(ARGS.itl_ms / 1000.0)
            delta = {"content": f"tok{i} "}
            choice = (
                {"index": 0, "delta": delta, "finish_reason": None}
                if is_chat
                else {"index": 0, "text": f"tok{i} ", "finish_reason": None}
            )
            chunk = {
                "id": "mock-1",
                "object": "chat.completion.chunk" if is_chat else "text_completion",
                "model": model,
                "choices": [choice],
            }
            self._sse(chunk)
        self._sse_raw("[DONE]")

    def _whole(self, is_chat, model, n_tokens):
        if ARGS.ttft_ms or ARGS.itl_ms:
            time.sleep((ARGS.ttft_ms + ARGS.itl_ms * (n_tokens - 1)) / 1000.0)
        text = " ".join(f"tok{i}" for i in range(n_tokens))
        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            if is_chat
            else {"index": 0, "text": text, "finish_reason": "stop"}
        )
        self._json(
            200,
            {
                "id": "mock-1",
                "object": "chat.completion" if is_chat else "text_completion",
                "model": model,
                "choices": [choice],
                "usage": {"prompt_tokens": 0, "completion_tokens": n_tokens, "total_tokens": n_tokens},
            },
        )

    def _sse(self, obj):
        self._sse_raw(json.dumps(obj))

    def _sse_raw(self, payload):
        try:
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--ttft-ms", type=float, default=0.0)
    ap.add_argument("--itl-ms", type=float, default=0.0)
    ap.add_argument("--tokens", type=int, default=16)
    ARGS = ap.parse_args()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"mock engine on http://{ARGS.host}:{ARGS.port}  ttft={ARGS.ttft_ms}ms itl={ARGS.itl_ms}ms")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
