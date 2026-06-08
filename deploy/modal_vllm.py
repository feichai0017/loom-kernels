"""Serve vLLM (OpenAI-compatible, with KV cache events) on a cloud GPU via Modal.

This is a deploy recipe you run with YOUR Modal account — it cannot be deployed
on your behalf.

    pip install modal
    modal token new                      # one-time auth
    modal deploy deploy/modal_vllm.py    # prints a public https URL

Point a QuillCache gateway engine `base_url` at that URL (see
docs/m3-real-vllm.md). Modal's API and vLLM flags evolve; pin/adjust versions to
match your setup.

KV events (Tier 2 — precise residency): vLLM publishes them over ZMQ *inside*
this container. Set QC_KV_EVENTS=1 and QC_GATEWAY_URL=https://your-gateway to
enable `--kv-events-config` and start bridge/vllm_kv_bridge.py as a co-located
sidecar that forwards events to your QuillCache gateway's /v1/kv-events. The
gateway must be reachable from Modal (a public URL or a tunnel) — a laptop-local
gateway is not. For a first run, leave QC_KV_EVENTS unset and just proxy requests
to get real TTFT from bench/run_trace.py (inferred residency closes the loop on
its own; events upgrade it to ground truth).
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub", "pyzmq", "msgpack", "requests")
    # Co-locate the KV-events bridge so Tier 2 can run as a sidecar in this
    # container (it must reach vLLM's in-container ZMQ endpoint). Modal's
    # add_local_file API is version-sensitive; adjust if your client differs.
    .add_local_file("bridge/vllm_kv_bridge.py", "/root/vllm_kv_bridge.py", copy=True)
)
# App name is parameterizable so one script can deploy a fleet:
#   modal deploy deploy/modal_vllm.py                              # quillcache-vllm
#   QC_VLLM_APP=quillcache-vllm-b modal deploy deploy/modal_vllm.py  # 2nd instance
app = modal.App(os.environ.get("QC_VLLM_APP", "quillcache-vllm"))


@app.function(gpu="L4", image=image, timeout=60 * 60, max_containers=1)
@modal.web_server(8000, startup_timeout=600)
def serve():
    # flashinfer JIT-compiles its sampler kernel at runtime and needs nvcc,
    # which the slim image lacks. Use vLLM's native sampler (no JIT, no nvcc).
    # The model forward (attention) uses a prebuilt backend and is unaffected.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    kv_events = os.environ.get("QC_KV_EVENTS") == "1"
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
    ]
    if kv_events:
        # Tier 2: publish KV cache events over ZMQ inside this container.
        cmd += [
            "--kv-events-config",
            '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}',
        ]
    subprocess.Popen(" ".join(cmd), shell=True)

    if kv_events:
        # Co-located sidecar: subscribe to vLLM's in-container ZMQ stream and
        # forward each batch to the QuillCache gateway as vendor-neutral JSON.
        gateway = os.environ.get("QC_GATEWAY_URL", "http://127.0.0.1:8080")
        engine_id = os.environ.get("QC_ENGINE_ID", os.environ.get("QC_VLLM_APP", "quillcache-vllm"))
        subprocess.Popen(
            [
                "python",
                "/root/vllm_kv_bridge.py",
                "--zmq",
                "tcp://127.0.0.1:5557",
                "--gateway",
                gateway,
                "--engine-id",
                engine_id,
            ]
        )
