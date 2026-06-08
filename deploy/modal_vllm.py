"""Serve vLLM (OpenAI-compatible, with KV cache events) on a cloud GPU via Modal.

This is a deploy recipe you run with YOUR Modal account — it cannot be deployed
on your behalf.

    pip install modal
    modal token new                      # one-time auth
    modal deploy deploy/modal_vllm.py    # prints a public https URL

Point a QuillCache gateway engine `base_url` at that URL (see
docs/m3-real-vllm.md). Modal's API and vLLM flags evolve; pin/adjust versions to
match your setup.

KV events: vLLM publishes them over ZMQ *inside* this container. To capture them
precisely, run bridge/vllm_kv_bridge.py as a sidecar in this container (see the
runbook). For a first run you can skip events and just proxy requests to get real
TTFT from bench/run_trace.py.
"""
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_VERSION = "0.6.6"

image = modal.Image.debian_slim().pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub")
app = modal.App("quillcache-vllm")


@app.function(gpu="L4", image=image, timeout=60 * 60, max_containers=1)
@modal.web_server(8000, startup_timeout=600)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--enable-prefix-caching",
        # Publish KV cache events over ZMQ for bridge/vllm_kv_bridge.py.
        "--kv-events-config",
        '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}',
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
