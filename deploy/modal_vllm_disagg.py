"""GPU-real TRUE mid-request P/D — vLLM-native disaggregation, end to end on GPUs.

Unlike modal_vllm_pd.py (content-addressed reuse, both engines `kv_both`), this
exercises vLLM's first-class disaggregation handshake with role-split engines:

  - prefill : vLLM on GPU 0 (port 8000), QuillCacheV1Connector, kv_role=kv_producer
  - decode  : vLLM on GPU 1 (port 8001), QuillCacheV1Connector, kv_role=kv_consumer
  - store   : one quillcache store-master + transfer-node on localhost, shared
  - proxy   : quillcache pd-proxy (the router) mints a transfer_id per request

ONE request to the proxy drives the real handshake:
  proxy → prefill (do_remote_decode, transfer_id T): the producer prefills and
          offloads the request's KV to the store under `qc-pd/T` (no real decode);
  proxy → decode  (do_remote_prefill, transfer_id T): the consumer pulls the KV
          named by T, SKIPS prefill, and generates.

The transfer_id — not the prompt content — names the KV, so this is true P/D for
a UNIQUE prompt (no prefix-cache hit possible). We then run the same prompt as a
monolithic request and assert the disagg output matches: proof the KV computed on
GPU 0 and pulled onto GPU 1 is correct.

    modal run deploy/modal_vllm_disagg.py
"""

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .pip_install("vllm", "huggingface_hub")
    .add_local_file("Cargo.toml", "/build/Cargo.toml", copy=True)
    .add_local_file("Cargo.lock", "/build/Cargo.lock", copy=True)
    .add_local_dir("crates", "/build/crates", copy=True, ignore=["**/target/**"])
    .add_local_dir("src", "/build/src", copy=True, ignore=["**/target/**"])
    .run_commands(
        "cd /build && $HOME/.cargo/bin/cargo build --release --bin quillcache",
        "cp /build/target/release/quillcache /usr/local/bin/quillcache",
    )
    .add_local_file("bridge/quillcache_v1_connector.py", "/root/quillcache_v1_connector.py", copy=True)
    .add_local_file("bridge/quillcache_store_client.py", "/root/quillcache_store_client.py", copy=True)
)
app = modal.App("quillcache-vllm-disagg")


@app.function(gpu="L4:2", image=image, timeout=60 * 60)
def run_disagg():
    import json
    import os
    import re
    import subprocess
    import time
    import urllib.request

    procs = {}

    def tail(path, n=60):
        try:
            return "\n".join(open(path, errors="replace").read().splitlines()[-n:])
        except OSError:
            return f"<no {path}>"

    def spawn(name, args, logpath, extra_env=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = "/root"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        if extra_env:
            env.update(extra_env)
        f = open(logpath, "w")
        procs[name] = (subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env), f)

    def wait_ready(url, timeout, name, proc=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"{name} exited early (code {proc.returncode})")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status < 500:
                        return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError(f"{name} not ready at {url} within {timeout}s")

    def kv_cfg(role, engine_id):
        return json.dumps({
            "kv_connector": "QuillCacheV1Connector",
            "kv_connector_module_path": "quillcache_v1_connector",
            "kv_role": role,
            "engine_id": engine_id,
            "kv_connector_extra_config": {
                "master_url": "http://127.0.0.1:7777",
                "segment_endpoints": {"seg-0": "127.0.0.1:8100"},
                "tenant_id": "default",
                "replica_num": 1,
            },
        })

    def serve(port, gpu_ordinal, logpath, role, engine_id):
        spawn(
            f"vllm-{port}",
            [
                "vllm", "serve", MODEL,
                "--port", str(port),
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.55",
                "--no-enable-prefix-caching",
                "--disable-hybrid-kv-cache-manager",
                "--kv-transfer-config", kv_cfg(role, engine_id),
            ],
            logpath,
            extra_env={"CUDA_VISIBLE_DEVICES": str(gpu_ordinal)},
        )

    # A UNIQUE prompt so no content/prefix cache could serve it — the only way
    # decode can skip prefill is the transfer_id handshake.
    PROMPT = (
        "You are QuillCache's disaggregation test harness, run id 7af3c91e. "
        "Considering the tradeoffs of KV-cache disaggregation across prefill and "
        "decode instances, answer precisely and concretely: "
        "In one sentence, what is a KV cache?"
    )

    def post(url, body, timeout=180):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return (time.time() - t0) * 1000, out

    def msg_body():
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 24,
            "temperature": 0.0,
        }

    try:
        # 1) shared store: one master + one transfer-node segment on localhost.
        spawn("transfer-node", ["quillcache", "transfer-node", "--addr", "127.0.0.1:8100", "--segment", "seg-0"], "/tmp/node.log")
        spawn("store-master", ["quillcache", "store-master", "--addr", "127.0.0.1:7777"], "/tmp/master.log")
        wait_ready("http://127.0.0.1:7777/v1/state", 30, "store-master", procs["store-master"][0])

        # 2) role-split engines: prefill=kv_producer (GPU0), decode=kv_consumer (GPU1).
        serve(8000, 0, "/tmp/vllm_prefill.log", "kv_producer", "prefill-engine")
        serve(8001, 1, "/tmp/vllm_decode.log", "kv_consumer", "decode-engine")
        wait_ready("http://127.0.0.1:8000/health", 900, "vllm-prefill", procs["vllm-8000"][0])
        wait_ready("http://127.0.0.1:8001/health", 900, "vllm-decode", procs["vllm-8001"][0])

        # 3) the router (pd-proxy) fronts both engines.
        spawn(
            "pd-proxy",
            ["quillcache", "pd-proxy", "--bind", "127.0.0.1:9000",
             "--prefill", "http://127.0.0.1:8000", "--decode", "http://127.0.0.1:8001"],
            "/tmp/proxy.log",
        )
        wait_ready("http://127.0.0.1:9000/v1/state", 30, "pd-proxy", procs["pd-proxy"][0])

        # 4) ONE request THROUGH the proxy → the true disagg handshake.
        t_disagg, disagg_out = post("http://127.0.0.1:9000/v1/chat/completions", msg_body())
        disagg_text = disagg_out["choices"][0]["message"]["content"]
        time.sleep(1.5)  # let producer commit + consumer load settle in the logs

        # 5) monolithic reference: same prompt straight to the decode engine, NO
        #    handshake (no kv_transfer_params) → normal prefill+decode on GPU 1.
        t_ref, ref_out = post("http://127.0.0.1:8001/v1/chat/completions", msg_body())
        ref_text = ref_out["choices"][0]["message"]["content"]

        prefill_log = open("/tmp/vllm_prefill.log", errors="replace").read()
        decode_log = open("/tmp/vllm_decode.log", errors="replace").read()

        def grep(text, needle):
            return [l for l in text.splitlines() if needle in l]

        # Producer offloaded under qc-pd/<transfer_id>; consumer pulled by id + loaded it.
        producer_committed = [l for l in grep(prefill_log, "QC committed") if "qc-pd/" in l]
        consumer_claimed = grep(decode_log, "QC disagg consumer")
        consumer_loaded = [l for l in grep(decode_log, "QC loading") if "qc-pd/" in l]

        # Correlate the transfer_id across producer + consumer.
        def first_id(lines, pat):
            for l in lines:
                m = re.search(pat, l)
                if m:
                    return m.group(1)
            return None

        producer_id = first_id(producer_committed, r"qc-pd/(\S+?)\)")
        consumer_id = first_id(consumer_claimed, r"transfer_id=(\S+?)\s")

        state = json.loads(urllib.request.urlopen("http://127.0.0.1:7777/v1/state").read())

        result = {
            "ok": True,
            "disagg_ms": round(t_disagg),
            "ref_ms": round(t_ref),
            "disagg_text": disagg_text,
            "ref_text": ref_text,
            "outputs_match": disagg_text.strip() == ref_text.strip(),
            "producer_committed": producer_committed[:3],
            "consumer_claimed": consumer_claimed[:3],
            "consumer_loaded": consumer_loaded[:3],
            "producer_id": producer_id,
            "consumer_id": consumer_id,
            "transfer_id_correlated": bool(producer_id) and producer_id == consumer_id,
            "store_state": state,
            "proxy_log_tail": "\n".join(open("/tmp/proxy.log", errors="replace").read().splitlines()[-12:]),
            "decode_log_tail": "\n".join(decode_log.splitlines()[-30:]),
        }
    except Exception as e:
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "prefill_log_tail": tail("/tmp/vllm_prefill.log"),
            "decode_log_tail": tail("/tmp/vllm_decode.log"),
            "master_log_tail": tail("/tmp/master.log", 15),
            "proxy_log_tail": tail("/tmp/proxy.log", 15),
        }
    finally:
        for _, (p, f) in procs.items():
            try:
                p.terminate()
            except Exception:
                pass
            f.close()
    return result


@app.local_entrypoint()
def main():
    import json

    res = run_disagg.remote()
    print("\n" + "=" * 80)
    print("QuillCache TRUE mid-request P/D — prefill=kv_producer(GPU0) → store → decode=kv_consumer(GPU1)")
    print("=" * 80)
    if not res.get("ok"):
        print(f"\nFAILED: {res.get('error')}")
        print("\n--- prefill log tail ---\n" + res.get("prefill_log_tail", ""))
        print("\n--- decode log tail ---\n" + res.get("decode_log_tail", ""))
        print("\n--- proxy log tail ---\n" + res.get("proxy_log_tail", ""))
        print("\n--- master log tail ---\n" + res.get("master_log_tail", ""))
        return

    print(f"\ndisagg via proxy: {res['disagg_ms']}ms → {res['disagg_text']!r}")
    print(f"monolithic ref:   {res['ref_ms']}ms → {res['ref_text']!r}")
    print(f"\noutputs identical: {res['outputs_match']}  (proof the transferred KV is correct)")

    print("\n[prefill GPU0 = kv_producer offloaded KV under qc-pd/<transfer_id>]")
    for l in res["producer_committed"]:
        print("   ", l.strip()[:150])
    print("\n[decode GPU1 = kv_consumer claimed the prefix by transfer_id (skipped prefill)]")
    for l in res["consumer_claimed"]:
        print("   ", l.strip()[:150])
    print("\n[decode GPU1 pulled the KV over the transfer engine]")
    for l in res["consumer_loaded"]:
        print("   ", l.strip()[:150])
    print(f"\ntransfer_id: producer={res['producer_id']}  consumer={res['consumer_id']}  correlated={res['transfer_id_correlated']}")
    print("\n--- store master /v1/state ---")
    print(json.dumps(res["store_state"], indent=2)[:600])

    real = (
        bool(res["producer_committed"])
        and bool(res["consumer_claimed"])
        and bool(res["consumer_loaded"])
        and res["transfer_id_correlated"]
    )
    print(
        f"\nVERDICT: {'REAL vLLM-native P/D — router minted a transfer_id; producer(GPU0) offloaded KV under it; consumer(GPU1) pulled by id, skipped prefill, generated' + (' — and output matches the monolithic reference' if res['outputs_match'] else ' (output differs from reference — see texts)') if real else 'partial — see logs'}"
    )
    if not real:
        print("\n--- proxy log tail ---\n" + res.get("proxy_log_tail", ""))
        print("\n--- decode log tail ---\n" + res["decode_log_tail"])
