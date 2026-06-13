"""A REAL vLLM v1 KV connector backed by the QuillCache store.

This subclasses vLLM's `KVConnectorBase_V1` (verified against the exact API of
the deployed vllm 0.22.1 — see deploy/modal_vllm_introspect.py) and is modelled
on vLLM's own reference `ExampleConnector`. It keeps vLLM's paged-KV slot-mapping
extract/inject logic *verbatim* (so the GPU tensor layout is correct for MLA /
Triton / default attention backends) and swaps the reference's
safetensors-on-local-disk for the **QuillCache distributed store**:

  - save  (offload): extract a layer's KV for the new prefix blocks → serialize
    (safetensors in-memory) → two-phase Put into the store (`put_start` → WRITE
    each replica over the transfer engine → `put_end`).
  - load  (prefix hit): **identity-guarded** `get_replica_list` (refused, HTTP
    403, *before any byte moves*, if the requester's identity doesn't match the
    writer's) → READ a replica over the transfer engine → deserialize → inject
    into vLLM's paged KV buffer.

The store master + transfer wire are the same real, tested QuillCache services
used by docs/real-engine-pool.md. The identity guard is QuillCache's
differentiator over a vanilla shared-storage connector: cross-tenant / cross-model
KV reuse is refused at the store, not merely by convention.

Run it (see deploy/modal_vllm_connector.py for the full Modal recipe):

    vllm serve <model> \
      --no-enable-prefix-caching \           # force prefix hits to come via the store
      --disable-hybrid-kv-cache-manager \    # this connector is not HMA-aware (like ExampleConnector)
      --kv-transfer-config '{
        "kv_connector": "QuillCacheV1Connector",
        "kv_connector_module_path": "quillcache_v1_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "master_url": "http://127.0.0.1:7777",
          "segment_endpoints": {"seg-0": "127.0.0.1:8100"},
          "tenant_id": "default",
          "replica_num": 1
        }
      }'
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

# The QuillCache store clients (stdlib; the master over HTTP + the transfer wire over TCP).
from quillcache_store_client import StoreMasterClient, TransferEngineClient, identity

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Largest block-aligned token count strictly below `num_tokens` (vLLM's rule)."""
    return max(0, (num_tokens - 1) // block_size * block_size)


def _prefix_hash(token_ids: list[int], block_size: int, mm_hashes: list[str]) -> tuple[str, int]:
    """Canonical (hash, aligned_len) for the block-aligned prompt prefix.

    Computed identically scheduler-side (match check + meta build) so save and
    load address the same store keys. Mirrors ExampleConnector's foldername hash.
    """
    aligned = align_to_block_size(len(token_ids), block_size)
    token_bytes = torch.tensor(token_ids[:aligned], dtype=torch.long).numpy().tobytes()
    if mm_hashes:
        token_bytes += ("-".join(mm_hashes)).encode("utf-8")
    digest = safe_hash(token_bytes, usedforsecurity=False).hexdigest()
    return digest, aligned


@dataclass
class ReqMeta:
    # Block-aligned prompt-prefix tokens this op covers.
    token_ids: torch.Tensor
    # Paged-KV slot for each token (same length as token_ids).
    slot_mapping: torch.Tensor
    # True = save (offload) this prefix; False = load it from the store.
    is_store: bool
    # Canonical store-key prefix (identity + prompt-prefix hash), computed scheduler-side.
    prefix_hash: str

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        prefix_hash: str,
    ) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            prefix_hash=prefix_hash,
        )


@dataclass
class QuillCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        prefix_hash: str,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, prefix_hash)
        )


class QuillCacheV1Connector(KVConnectorBase_V1):
    """vLLM v1 KV connector whose external store is the QuillCache pool."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size

        cfg = self._kv_transfer_config
        model = vllm_config.model_config.model
        tokenizer = getattr(vllm_config.model_config, "tokenizer", None) or model
        # Identity = QuillCache IdentityScope. The store enforces it on every Get.
        self._identity = identity(
            model_id=cfg.get_from_extra_config("model_id", model),
            tokenizer_id=cfg.get_from_extra_config("tokenizer_id", tokenizer),
            tenant_id=cfg.get_from_extra_config("tenant_id", "default"),
            adapter_id=cfg.get_from_extra_config("adapter_id", None),
        )
        self._master = StoreMasterClient(
            cfg.get_from_extra_config("master_url", "http://127.0.0.1:7777")
        )
        self._transfer = TransferEngineClient()
        # {segment_name: "host:port"} — the transfer node serving each segment.
        endpoints = cfg.get_from_extra_config("segment_endpoints", {"seg-0": "127.0.0.1:8100"})
        if isinstance(endpoints, str):
            endpoints = json.loads(endpoints)
        self._segment_endpoints = dict(endpoints)
        self._replica_num = int(cfg.get_from_extra_config("replica_num", 1))

        # Scheduler-side: requests for which a prefix hit was found and blocks
        # were allocated, so the worker must load them next forward pass.
        self._requests_need_load: dict[str, Request] = {}
        # Worker-side: prefixes saved this step -> aligned token count (for manifest).
        self._saved_this_step: dict[str, int] = {}

        # The worker owns byte movement; it registers the storage segments on the
        # master so put_start can allocate on them. Idempotent-tolerant.
        if role == KVConnectorRole.WORKER:
            for name in self._segment_endpoints:
                try:
                    self._master.mount(name, 1 << 30)
                except Exception as e:  # already mounted, or master not up yet
                    logger.info("segment %s mount skipped: %s", name, e)

        logger.info(
            "QuillCacheV1Connector role=%s master=%s segments=%s identity=%s",
            role,
            self._master.base,
            list(self._segment_endpoints),
            self._identity,
        )

    # ==============================
    # Store primitives (the real QuillCache data path)
    # ==============================

    def _layer_key(self, prefix_hash: str, layer_name: str) -> str:
        return f"qc/{prefix_hash}/{layer_name}"

    def _manifest_key(self, prefix_hash: str) -> str:
        return f"qc/{prefix_hash}/__manifest__"

    def _put_bytes(self, key: str, data: bytes) -> None:
        """Two-phase Put: allocate replica buffers, WRITE each, commit."""
        buffers = self._master.put_start(key, self._identity, len(data), self._replica_num)
        for buffer in buffers:
            endpoint = self._segment_endpoints[buffer["segment_name"]]
            self._transfer.write(endpoint, buffer["offset"], data)
        self._master.put_end(key)

    def _get_bytes(self, key: str) -> bytes | None:
        """Identity-guarded Get: locate a replica, READ its bytes. None on miss.

        The store refuses (HTTP 403) before any byte moves if this connector's
        identity doesn't match the writer's — surfaced here as a logged miss."""
        try:
            replicas = self._master.get_replica_list(key, self._identity)
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 403:
                logger.warning("QuillCache identity guard REFUSED reuse of %s", key)
            elif code not in (404,):
                logger.warning("get_replica_list(%s) failed: %s", key, e)
            return None
        for replica in replicas:
            memory = (replica.get("data") or {}).get("Memory")
            if memory:
                endpoint = self._segment_endpoints[memory["segment_name"]]
                return self._transfer.read(endpoint, memory["offset"], memory["size"])
        return None

    def _exists(self, key: str) -> bool:
        """Does this prefix's commit manifest exist & is it ours? (no byte move)."""
        try:
            return bool(self._master.get_replica_list(key, self._identity))
        except Exception as e:
            if getattr(e, "code", None) == 403:
                logger.warning("QuillCache identity guard REFUSED match on %s", key)
            return False

    @staticmethod
    def _serialize(t: torch.Tensor) -> bytes:
        return safetensors.torch.save({"kv": t.detach().cpu().contiguous()})

    @staticmethod
    def _deserialize(buf: bytes, device: str) -> torch.Tensor:
        return safetensors.torch.load(buf)["kv"].to(device)

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Inject store-resident KV for each load request into the paged buffer."""

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ) -> None:
            # Verbatim from vLLM's ExampleConnector — layout-correct per backend.
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            elif isinstance(attn_metadata, TritonAttentionMetadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QuillCacheConnectorMetadata)
        n_load = sum(1 for r in metadata.requests if not r.is_store)
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            if n_load:
                logger.warning(
                    "QC start_load_kv: attn_metadata is None but load_reqs=%d — load SKIPPED",
                    n_load,
                )
            return

        for request in metadata.requests:
            if request.is_store:
                continue
            logger.warning(
                "QC loading %d tokens of KV from the store (prefix %s)",
                len(request.slot_mapping),
                request.prefix_hash[:12],
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue  # skip non-attention layers (MLP/MoE)
                buf = self._get_bytes(self._layer_key(request.prefix_hash, layer_name))
                if buf is None:
                    continue  # miss / evicted / refused — vLLM recomputes this layer
                kv_cache = self._deserialize(buf, str(kv_cache_layer.device))
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer, kv_cache, request.slot_mapping, attn_metadata[layer_name]
                    )

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous loads (above) — nothing to await per layer.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Extract a layer's KV for each store request and offload it to the pool."""

        def extract_kv_from_layer(layer: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
            # Verbatim from vLLM's ExampleConnector — inverse of inject.
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            elif isinstance(attn_metadata, TritonAttentionMetadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                return layer[block_idxs, :, offsets]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, QuillCacheConnectorMetadata)
        for request in metadata.requests:
            if not request.is_store:
                continue
            kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
            self._put_bytes(self._layer_key(request.prefix_hash, layer_name), self._serialize(kv_cache))
            self._saved_this_step[request.prefix_hash] = int(request.token_ids.numel())

    def wait_for_save(self) -> None:
        """Commit a manifest per saved prefix — the marker a later match check reads."""
        for prefix_hash, num_tokens in self._saved_this_step.items():
            manifest = json.dumps({"tokens": num_tokens, "block_size": self._block_size}).encode()
            self._put_bytes(self._manifest_key(prefix_hash), manifest)
            logger.warning(
                "QC committed %d-token prefix %s to the store",
                num_tokens,
                prefix_hash[:12],
            )
        self._saved_this_step.clear()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """How many *additional* prefix tokens the store can serve for this request."""
        token_ids = list(request.prompt_token_ids or [])
        mm_hashes = [f.identifier for f in request.mm_features]
        prefix_hash, aligned = _prefix_hash(token_ids, self._block_size, mm_hashes)
        exists = self._exists(self._manifest_key(prefix_hash))
        logger.warning(
            "QC match-check req=%s prefix=%s manifest=%s num_computed=%d aligned=%d block_size=%d ntok=%d",
            getattr(request, "request_id", "?"),
            prefix_hash[:12],
            exists,
            num_computed_tokens,
            aligned,
            self._block_size,
            len(token_ids),
        )
        if aligned <= num_computed_tokens or not exists:
            return 0, False
        logger.warning(
            "QC external cache HIT prefix=%s (+%d tokens)",
            prefix_hash[:12],
            aligned - num_computed_tokens,
        )
        # Synchronous load (we inject during the forward pass), so async=False.
        return aligned - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = QuillCacheConnectorMetadata()
        total_need_load = 0

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            prefix_hash, _ = _prefix_hash(list(token_ids), self._block_size, mm_hashes)
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=list(token_ids),
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    prefix_hash=prefix_hash,
                )
                total_need_load += 1
            elif not self._exists(self._manifest_key(prefix_hash)):
                # Not already in the store -> save this prefix after the forward.
                meta.add_request(
                    token_ids=list(token_ids),
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                    prefix_hash=prefix_hash,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]
            mm_hashes = [f.identifier for f in request.mm_features]
            prefix_hash, _ = _prefix_hash(list(token_ids), self._block_size, mm_hashes)
            assert new_block_ids is not None
            meta.add_request(
                token_ids=list(token_ids),
                block_ids=new_block_ids[0],
                block_size=self._block_size,
                is_store=False,
                prefix_hash=prefix_hash,
            )
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta
