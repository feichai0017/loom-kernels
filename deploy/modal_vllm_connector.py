"""Stage B — the REAL connector end-to-end on a GPU (Modal L4).

One container runs the whole faithful data path, all real:
  - `quillcache transfer-node`  — a Transfer Engine RAM segment over TCP   (Rust)
  - `quillcache store-master`   — the MasterService: two-phase Put, the     (Rust)
                                  identity-guarded Get
  - `vllm serve <model>`        — vLLM 0.22.1 with QuillCacheV1Connector      (GPU)
                                  registered via --kv-transfer-config

Prefix caching is turned OFF in vLLM, so the *only* way a repeated prompt can
skip prefill is through the connector loading KV from the QuillCache store.
We send the same long-prefix prompt twice and prove from the vLLM logs that:
  request 1  -> connector SAVED the prefix   ("committed N-token prefix … to the store")
  request 2  -> connector HIT + LOADED it    ("external cache HIT" / "loading N tokens … from the store")
and that the store master reports the stored objects.

    modal run deploy/modal_vllm_connector.py

First run builds the Rust binary into the image (cached afterwards) and downloads
the 0.5B model. GPU + a few minutes.
"""

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    # Rust toolchain, to build the real store binary from this repo.
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .pip_install("vllm", "huggingface_hub")
    # Only the Rust build inputs (NOT the whole repo) so edits elsewhere can't
    # race the build snapshot; then build just the binary.
    .add_local_file("Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir("crates", "/build/crates", copy=True, ignore=["**/target/**"])
    .add_local_dir("src", "/build/src", copy=True, ignore=["**/target/**"])
    .run_commands(
        "cd /build && $HOME/.cargo/bin/cargo build --release --bin quillcache",
        "cp /build/target/release/quillcache /usr/local/bin/quillcache",
    )
    # The connector + store client, importable by vLLM's worker/scheduler processes.
    .add_local_file("bridge/quillcache_v1_connector.py", "/root/quillcache_v1_connector.py", copy=True)
    .add_local_file("bridge/quillcache_store_client.py", "/root/quillcache_store_client.py", copy=True)
)
app = modal.App("quillcache-vllm-connector")


@app.function(gpu="L4", image=image, timeout=60 * 60)
def run_e2e():
    import json
    import os
    import subprocess
    import time
    import urllib.request

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    procs = {}

    def spawn(name, args, logpath):
        f = open(logpath, "w")
        procs[name] = (subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env), f)

    def tail(path, n=60):
        try:
            return "\n".join(open(path, errors="replace").read().splitlines()[-n:])
        except OSError:
            return f"<no {path}>"

    def wait_ready(url, timeout, name, proc=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited early (code {proc.returncode}) before ready:\n{tail('/tmp/vllm.log')}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status < 500:
                        return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(f"{name} not ready at {url} within {timeout}s")

    def shutdown():
        for _, (p, f) in procs.items():
            try:
                p.terminate()
            except Exception:
                pass
            f.close()

    try:
        # 1) the storage node + the master (both Rust, CPU).
        spawn("transfer-node", ["quillcache", "transfer-node", "--addr", "127.0.0.1:8100", "--segment", "seg-0"], "/tmp/node.log")
        spawn("store-master", ["quillcache", "store-master", "--addr", "127.0.0.1:7777"], "/tmp/master.log")
        wait_ready("http://127.0.0.1:7777/v1/state", 30, "store-master", procs["store-master"][0])

        # 2) vLLM with the QuillCache connector. Prefix caching OFF so a repeated
        #    prompt can only skip prefill via the connector.
        kv_cfg = {
            "kv_connector": "QuillCacheV1Connector",
            "kv_connector_module_path": "quillcache_v1_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "master_url": "http://127.0.0.1:7777",
                "segment_endpoints": {"seg-0": "127.0.0.1:8100"},
                "tenant_id": "default",
                "replica_num": 1,
            },
        }
        spawn(
            "vllm",
            [
                "vllm", "serve", MODEL,
                "--port", "8000",
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.6",
                "--no-enable-prefix-caching",
                "--disable-hybrid-kv-cache-manager",
                "--kv-transfer-config", json.dumps(kv_cfg),
            ],
            "/tmp/vllm.log",
        )
        wait_ready("http://127.0.0.1:8000/health", 600, "vllm", procs["vllm"][0])

        # 3) Two identical long-prefix requests. #1 populates the store, #2 must hit it.
        shared_prefix = (
            "You are a meticulous assistant. Follow these standing rules for every "
            "answer. " + " ".join(
                f"Rule {i}: always be precise, cite assumptions, and prefer concrete "
                "examples over abstractions." for i in range(24)
            )
        )

        def chat(tag):
            body = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": shared_prefix},
                    {"role": "user", "content": "In one sentence, what is a KV cache?"},
                ],
                "max_tokens": 16,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"}, method="POST",
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            dt = (time.time() - t0) * 1000
            usage = out.get("usage", {})
            text = out["choices"][0]["message"]["content"]
            print(f"[{tag}] {dt:.0f} ms  prompt_tokens={usage.get('prompt_tokens')}  -> {text!r}")
            return dt

        t1 = chat("request-1 (populate store)")
        time.sleep(2.0)  # let async saves + manifest commit settle
        t2 = chat("request-2 (should hit store)")
        time.sleep(1.0)

        # 4) Evidence: connector log lines + store state.
        vllm_log = open("/tmp/vllm.log", errors="replace").read()
        markers = [
            "QC match-check",
            "QC committed",
            "QC external cache HIT",
            "QC loading",
            "attn_metadata is None but load",
            "identity guard REFUSED",
        ]
        hits = {m: [l for l in vllm_log.splitlines() if m in l] for m in markers}
        state = json.loads(urllib.request.urlopen("http://127.0.0.1:7777/v1/state").read())
        result = {
            "ok": True,
            "ttft_ms": {"request_1": round(t1), "request_2": round(t2)},
            "connector_log_evidence": {m: hits[m][:4] for m in markers},
            "store_state": state,
            "vllm_log_tail": "\n".join(vllm_log.splitlines()[-40:]),
        }
    except Exception as e:
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "vllm_log_tail": tail("/tmp/vllm.log"),
            "master_log_tail": tail("/tmp/master.log", 20),
            "node_log_tail": tail("/tmp/node.log", 20),
        }
    finally:
        shutdown()
    return result


@app.local_entrypoint()
def main():
    import json

    res = run_e2e.remote()
    print("\n" + "=" * 80)
    print("QuillCacheV1Connector — end-to-end on vLLM 0.22.1 + the real store (L4)")
    print("=" * 80)
    if not res.get("ok"):
        print(f"\nFAILED: {res.get('error')}")
        print("\n--- vllm log tail ---\n" + res.get("vllm_log_tail", ""))
        print("\n--- master log tail ---\n" + res.get("master_log_tail", ""))
        print("\n--- node log tail ---\n" + res.get("node_log_tail", ""))
        return
    print("\nlatency:", res["ttft_ms"])
    print("\n--- connector log evidence (from inside vLLM's processes) ---")
    for marker, lines in res["connector_log_evidence"].items():
        print(f"\n[{marker}]  ({len(lines)} line(s))")
        for l in lines:
            print("   ", l.strip()[:160])
    print("\n--- store master /v1/state ---")
    print(json.dumps(res["store_state"], indent=2)[:1500])
    save = res["connector_log_evidence"]["QC committed"]
    load = res["connector_log_evidence"]["QC loading"]
    verdict = "REAL end-to-end: store populated AND reused via the connector" if (save and load) \
        else "partial — see match-check trace + vllm_log_tail"
    print(f"\nVERDICT: {verdict}")
    if not (save and load):
        print("\n--- vllm log tail ---\n" + res["vllm_log_tail"])
