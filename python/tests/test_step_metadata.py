from __future__ import annotations

from types import SimpleNamespace
import unittest

from loom_attention.step_metadata import (
    StepMetadataContractError,
    StepMetadataObserver,
)
from test_local_delegate import FakeTensor


class StepMetadataObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observer = StepMetadataObserver(
            layer_names=("model.layers.0.self_attn",),
            block_size=16,
            num_attention_heads=8,
            num_kv_heads=2,
            head_size=64,
            kv_cache_dtype="torch.bfloat16",
        )

    def metadata(self) -> SimpleNamespace:
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

    def test_captures_zero_copy_step_snapshot(self) -> None:
        metadata = self.metadata()
        snapshot = self.observer.capture(
            common_prefix_tokens=16,
            common_metadata=metadata,
            fast_build=False,
        )

        self.assertEqual(snapshot.generation, 1)
        self.assertEqual(snapshot.query_start_offsets, (0, 1, 3))
        self.assertEqual(snapshot.query_tokens, 3)
        self.assertEqual(snapshot.block_table.shape, (2, 4))
        self.assertEqual(snapshot.block_table.data_ptr, 0x400)
        self.assertEqual(snapshot.block_table.bytes, 2 * 4 * 4)
        self.assertEqual(metadata.query_start_loc_cpu.tolist_calls, 1)
        self.assertEqual(metadata.block_table_tensor.tolist_calls, 0)
        self.assertEqual(metadata.seq_lens.tolist_calls, 0)

    def test_block_table_update_advances_generation_without_reading_device(self) -> None:
        metadata = self.metadata()
        first = self.observer.capture(
            common_prefix_tokens=0,
            common_metadata=metadata,
            fast_build=True,
        )
        block_table = FakeTensor((2, 6), dtype="torch.int32", address=0x600)
        slot_mapping = FakeTensor((3,), dtype="torch.int64", address=0x700)

        second = self.observer.update_block_table(
            first,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

        self.assertEqual(second.generation, 2)
        self.assertEqual(second.block_table.shape, (2, 6))
        self.assertEqual(second.block_table.data_ptr, 0x600)
        self.assertEqual(block_table.tolist_calls, 0)
        self.assertEqual(slot_mapping.tolist_calls, 0)

    def test_rejects_non_monotonic_cpu_query_offsets(self) -> None:
        metadata = self.metadata()
        metadata.query_start_loc_cpu.values = [0, 2, 1]
        with self.assertRaisesRegex(
            StepMetadataContractError, "monotonically increasing"
        ):
            self.observer.capture(
                common_prefix_tokens=0,
                common_metadata=metadata,
                fast_build=False,
            )

    def test_rejects_cross_device_metadata(self) -> None:
        metadata = self.metadata()
        metadata.slot_mapping.device = "cuda:1"
        with self.assertRaisesRegex(StepMetadataContractError, "share one device"):
            self.observer.capture(
                common_prefix_tokens=0,
                common_metadata=metadata,
                fast_build=False,
            )

    def test_refuses_to_copy_query_offsets_from_gpu(self) -> None:
        metadata = self.metadata()
        metadata.query_start_loc_cpu.device = "cuda:0"
        with self.assertRaisesRegex(StepMetadataContractError, "refusing"):
            self.observer.capture(
                common_prefix_tokens=0,
                common_metadata=metadata,
                fast_build=False,
            )

    def test_rejects_floating_point_block_table(self) -> None:
        metadata = self.metadata()
        metadata.block_table_tensor.dtype = "torch.float32"
        with self.assertRaisesRegex(StepMetadataContractError, "int32 or int64"):
            self.observer.capture(
                common_prefix_tokens=0,
                common_metadata=metadata,
                fast_build=False,
            )

    def test_rejects_non_contiguous_device_metadata(self) -> None:
        metadata = self.metadata()
        metadata.block_table_tensor.is_contiguous = lambda: False
        with self.assertRaisesRegex(StepMetadataContractError, "must be contiguous"):
            self.observer.capture(
                common_prefix_tokens=0,
                common_metadata=metadata,
                fast_build=False,
            )


if __name__ == "__main__":
    unittest.main()
