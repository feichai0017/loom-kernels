from __future__ import annotations

import copy
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from loom_attention import vllm_plugin
from test_local_delegate import FakeTensor


class FakeFlashAttentionImpl:
    def __init__(self, *args, **kwargs) -> None:
        self.delegate_calls = 0

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        self.delegate_calls += 1
        if layer == "fail":
            raise RuntimeError("delegate failure")
        return output


class FakeFlashAttentionBackend:
    @staticmethod
    def get_builder_cls():
        return FakeFlashAttentionMetadataBuilder


class FakeFlashAttentionMetadataBuilder:
    supports_update_block_table = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.num_heads_q = 8
        self.num_heads_kv = 2
        self.headdim = 64
        self.delegate_builds = 0

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        self.delegate_builds += 1
        return SimpleNamespace(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        updated = copy.copy(metadata)
        updated.block_table = blk_table
        updated.slot_mapping = slot_mapping
        return updated


class VllmPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        vllm_plugin._REGISTERED = False
        vllm_plugin._FORWARD_OBSERVERS.clear()
        vllm_plugin._STEP_OBSERVERS.clear()
        for name in (
            "LoomFlashAttentionBackend",
            "LoomFlashAttentionImpl",
            "LoomFlashAttentionMetadataBuilder",
        ):
            vllm_plugin.__dict__.pop(name, None)

    def fake_modules(self, registrations: list[tuple[object, str]]) -> dict[str, ModuleType]:
        packages = {
            name: ModuleType(name)
            for name in (
                "vllm",
                "vllm.v1",
                "vllm.v1.attention",
                "vllm.v1.attention.backends",
            )
        }
        flash = ModuleType("vllm.v1.attention.backends.flash_attn")
        flash.FlashAttentionBackend = FakeFlashAttentionBackend
        flash.FlashAttentionImpl = FakeFlashAttentionImpl
        flash.FlashAttentionMetadataBuilder = FakeFlashAttentionMetadataBuilder

        registry = ModuleType("vllm.v1.attention.backends.registry")
        custom = object()
        registry.AttentionBackendEnum = SimpleNamespace(CUSTOM=custom)

        def register_backend(backend, class_path=None):
            registrations.append((backend, class_path))
            return lambda value: value

        registry.register_backend = register_backend
        packages[flash.__name__] = flash
        packages[registry.__name__] = registry
        return packages

    def tensors(self) -> dict[str, FakeTensor]:
        return {
            "query": FakeTensor((4, 8 * 64)),
            "key": FakeTensor((4, 2 * 64)),
            "value": FakeTensor((4, 2 * 64)),
            "kv_cache": FakeTensor((2, 32, 16, 2, 64)),
            "output": FakeTensor((4, 8 * 64)),
        }

    def common_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=3,
            max_query_len=2,
            max_seq_len=32,
            query_start_loc_cpu=FakeTensor(
                (3,),
                device="cpu",
                dtype="torch.int32",
                address=0x100,
                values=[0, 1, 3],
            ),
            query_start_loc=FakeTensor(
                (3,), dtype="torch.int32", address=0x200
            ),
            seq_lens=FakeTensor((2,), dtype="torch.int32", address=0x300),
            block_table_tensor=FakeTensor(
                (2, 4), dtype="torch.int32", address=0x400
            ),
            slot_mapping=FakeTensor((3,), dtype="torch.int64", address=0x500),
        )

    def test_registers_custom_backend_and_delegates_unchanged_output(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LOOM_VLLM_DELEGATE", None)
                vllm_plugin.register()

        self.assertEqual(len(registrations), 1)
        self.assertEqual(
            registrations[0][1],
            "loom_attention.vllm_plugin.LoomFlashAttentionBackend",
        )
        self.assertEqual(
            vllm_plugin.LoomFlashAttentionBackend.get_name(), "CUSTOM"
        )
        implementation = vllm_plugin.LoomFlashAttentionBackend.get_impl_cls()(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            kv_cache_dtype="bfloat16",
            attn_type="decoder",
        )
        tensors = self.tensors()
        result = implementation.forward(
            "layer", attn_metadata=object(), **tensors
        )
        self.assertIs(result, tensors["output"])
        self.assertEqual(implementation.delegate_calls, 1)
        self.assertEqual(implementation.loom_observer.snapshot().calls, 1)
        self.assertFalse(implementation.loom_observer.validate_every_call)
        telemetry = vllm_plugin.telemetry_snapshot()
        self.assertEqual(telemetry["implementation_count"], 1)
        self.assertEqual(telemetry["forward_calls"], 1)
        self.assertEqual(telemetry["forward_failures"], 0)

    def test_registration_is_idempotent(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            vllm_plugin.register()
            vllm_plugin.register()
        self.assertEqual(len(registrations), 1)

    def test_metadata_builder_attaches_and_updates_step_snapshot(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            vllm_plugin.register()

        builder_class = vllm_plugin.LoomFlashAttentionBackend.get_builder_cls()
        builder = builder_class(
            SimpleNamespace(block_size=16, dtype="torch.bfloat16"),
            ["model.layers.0.self_attn"],
            SimpleNamespace(),
            "cuda:0",
        )
        common_metadata = self.common_metadata()
        metadata = builder.build(16, common_metadata)
        snapshot = metadata.loom_step_snapshot

        self.assertEqual(builder.delegate_builds, 1)
        self.assertEqual(snapshot.generation, 1)
        self.assertEqual(snapshot.common_prefix_tokens, 16)
        self.assertEqual(common_metadata.block_table_tensor.tolist_calls, 0)

        updated = builder.update_block_table(
            metadata,
            FakeTensor((2, 6), dtype="torch.int32", address=0x600),
            FakeTensor((3,), dtype="torch.int64", address=0x700),
        )
        self.assertEqual(updated.loom_step_snapshot.generation, 2)
        self.assertEqual(updated.loom_step_snapshot.block_table.data_ptr, 0x600)
        telemetry = vllm_plugin.telemetry_snapshot()
        self.assertEqual(telemetry["metadata_builder_count"], 1)
        self.assertEqual(telemetry["max_step_generation"], 2)

    def test_delegate_failure_is_recorded_and_propagated(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            vllm_plugin.register()
        implementation = vllm_plugin.LoomFlashAttentionBackend.get_impl_cls()(
            8, 64, 0.125, 2
        )
        with self.assertRaisesRegex(RuntimeError, "delegate failure"):
            implementation.forward("fail", attn_metadata=object(), **self.tensors())
        snapshot = implementation.loom_observer.snapshot()
        self.assertEqual(snapshot.calls, 1)
        self.assertEqual(snapshot.failures, 1)

    def test_rejects_unimplemented_delegate(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(
                os.environ, {"LOOM_VLLM_DELEGATE": "triton"}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "supports only"):
                    vllm_plugin.register()

    def test_rejects_invalid_validation_mode(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(
                os.environ,
                {"LOOM_VALIDATE_EVERY_FORWARD": "sometimes"},
                clear=False,
            ):
                vllm_plugin.register()
                implementation_class = (
                    vllm_plugin.LoomFlashAttentionBackend.get_impl_cls()
                )
                with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                    implementation_class(8, 64, 0.125, 2)


if __name__ == "__main__":
    unittest.main()
