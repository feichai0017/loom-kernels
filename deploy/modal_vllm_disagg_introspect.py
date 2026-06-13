"""Dump vLLM 0.22.1's native disaggregated-prefilling API (CPU, no GPU).

For true mid-request P/D we need the ground-truth of how vLLM 0.22.1 wires a
kv_producer (prefill) to a kv_consumer (decode): the KVTransferConfig (kv_role),
the request-level `kv_transfer_params` handshake, and how a consumer connector's
get_num_new_matched_tokens / update_state_after_alloc consume the producer's KV.

    modal run deploy/modal_vllm_disagg_introspect.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("quillcache-vllm-disagg-introspect")


@app.function(image=image, timeout=600)
def dump():
    import glob
    import os
    import re
    import sysconfig

    sp = sysconfig.get_paths()["purelib"]
    root = os.path.join(sp, "vllm")
    out = {}

    # 1) The KVTransferConfig source (kv_role, connector, extra_config).
    for f in glob.glob(os.path.join(root, "config/kv_transfer*.py")) + glob.glob(
        os.path.join(root, "**/kv_transfer_config.py"), recursive=True
    ):
        out[f.replace(sp + "/", "")] = open(f, errors="replace").read()

    # 2) Where `kv_transfer_params` is threaded through (the disagg handshake).
    hits = {}
    for f in glob.glob(os.path.join(root, "**/*.py"), recursive=True):
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        for kw in ("kv_transfer_params", "do_remote_prefill", "do_remote_decode", "remote_engine_id"):
            for i, line in enumerate(text.splitlines()):
                if kw in line:
                    hits.setdefault(kw, []).append(
                        f"{f.replace(sp + '/', '')}:{i + 1}: {line.strip()[:140]}"
                    )

    # 3) Any shipped disagg proxy / example entrypoint inside the package.
    proxy_files = [
        f.replace(sp + "/", "")
        for f in glob.glob(os.path.join(root, "**/*.py"), recursive=True)
        if re.search(r"disagg|disaggregat", os.path.basename(f), re.I)
    ]

    return {
        "config_sources": out,
        "kv_transfer_params_hits": {k: v[:25] for k, v in hits.items()},
        "disagg_files": proxy_files,
    }


@app.local_entrypoint()
def main():
    res = dump.remote()
    for name, src in res["config_sources"].items():
        print("=" * 90)
        print(name)
        print("=" * 90)
        print(src[:6000])
    print("\n\n##### kv_transfer_params / disagg handshake call sites #####")
    for kw, lines in res["kv_transfer_params_hits"].items():
        print(f"\n--- {kw} ({len(lines)} shown) ---")
        for l in lines:
            print(l)
    print("\n\n##### files named *disagg* in the package #####")
    for f in res["disagg_files"]:
        print(f)
