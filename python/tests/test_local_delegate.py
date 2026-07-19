from __future__ import annotations

import unittest

from loom_attention.local_delegate import LocalForwardObserver, TensorContractError


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        device: str = "cuda:0",
        *,
        dtype: str = "torch.bfloat16",
        address: int = 0x1000,
        values: list[int] | None = None,
    ) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.address = address
        self.values = values
        self.tolist_calls = 0

    def numel(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def data_ptr(self) -> int:
        return self.address

    def element_size(self) -> int:
        if "64" in self.dtype:
            return 8
        if "32" in self.dtype:
            return 4
        return 2

    def is_contiguous(self) -> bool:
        return True

    def tolist(self) -> list[int]:
        self.tolist_calls += 1
        if self.values is None:
            raise AssertionError("test attempted to read an opaque device tensor")
        return list(self.values)


class LocalForwardObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observer = LocalForwardObserver(
            num_heads=8,
            head_size=64,
            num_kv_heads=2,
            kv_cache_dtype="bfloat16",
            attention_type="decoder",
        )

    def tensors(self) -> dict[str, FakeTensor]:
        return {
            "query": FakeTensor((4, 8 * 64)),
            "key": FakeTensor((4, 2 * 64)),
            "value": FakeTensor((4, 2 * 64)),
            "kv_cache": FakeTensor((2, 32, 16, 2, 64)),
            "output": FakeTensor((4, 8 * 64)),
        }

    def test_valid_call_updates_process_local_telemetry(self) -> None:
        token = self.observer.before_forward(**self.tensors())
        self.observer.after_forward(token)
        snapshot = self.observer.snapshot()
        self.assertEqual(snapshot.calls, 1)
        self.assertEqual(snapshot.failures, 0)
        self.assertEqual(snapshot.last_device, "cuda:0")
        self.assertEqual(len(snapshot.layout_digest), 64)

    def test_rejects_cross_device_local_attention(self) -> None:
        tensors = self.tensors()
        tensors["value"] = FakeTensor((4, 2 * 64), device="cuda:1")
        with self.assertRaisesRegex(TensorContractError, "share one device"):
            self.observer.before_forward(**tensors)

    def test_rejects_head_layout_mismatch(self) -> None:
        tensors = self.tensors()
        tensors["query"] = FakeTensor((4, 17))
        with self.assertRaisesRegex(TensorContractError, "head_size/head_count"):
            self.observer.before_forward(**tensors)

    def test_rejects_explicit_three_dimensional_head_mismatch(self) -> None:
        tensors = self.tensors()
        tensors["query"] = FakeTensor((4, 7, 64))
        with self.assertRaisesRegex(TensorContractError, "expected 8"):
            self.observer.before_forward(**tensors)

    def test_can_validate_only_the_first_forward(self) -> None:
        observer = LocalForwardObserver(
            num_heads=8,
            head_size=64,
            num_kv_heads=2,
            kv_cache_dtype="bfloat16",
            attention_type="decoder",
            validate_every_call=False,
        )
        first = observer.before_forward(**self.tensors())
        observer.after_forward(first)

        tensors = self.tensors()
        tensors["value"] = FakeTensor((4, 2 * 64), device="cuda:1")
        second = observer.before_forward(**tensors)
        observer.after_forward(second)
        self.assertEqual(observer.snapshot().calls, 2)


if __name__ == "__main__":
    unittest.main()
