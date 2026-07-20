import io
import json
import unittest
from unittest.mock import patch

from integration.two_gpu_benchmark import (
    BenchmarkConfig,
    _residual_samples,
    _route_residual_samples,
    percentile,
    projected_transfer_bytes,
)
from integration.two_gpu_smoke import main


class TwoGpuSmokeContractTest(unittest.TestCase):
    def test_projected_bytes_show_route_query_asymmetry(self) -> None:
        config = BenchmarkConfig(
            prefix_tokens=4096,
            rows=1,
            query_heads=32,
            kv_heads=8,
            head_dim=128,
            dtype="float16",
        )
        payload = projected_transfer_bytes(config)
        self.assertEqual(payload["query"], 8192)
        self.assertEqual(payload["output"], 8192)
        self.assertEqual(payload["logsumexp"], 128)
        self.assertEqual(payload["attention_state"], 8320)
        self.assertEqual(payload["route_query_total"], 16512)
        self.assertEqual(payload["stage_kv_total"], 16_777_216)
        self.assertLess(payload["route_query_total"], payload["stage_kv_total"])

    def test_rejects_invalid_gqa_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "kv_heads must divide query_heads"):
            BenchmarkConfig(query_heads=12, kv_heads=5).validate()

    def test_rejects_unknown_attention_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported attention backend"):
            BenchmarkConfig(attention_backend="unknown").validate()

    def test_rejects_unknown_route_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported Route-Q strategy"):
            BenchmarkConfig(route_strategy="unknown").validate()

    def test_fused_route_strategy_enforces_kernel_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "1..=64 tail tokens"):
            BenchmarkConfig(route_strategy="fused", tail_tokens=0).validate()
        with self.assertRaisesRegex(ValueError, "head_dim <= 256"):
            BenchmarkConfig(route_strategy="fused", head_dim=512).validate()

    def test_rejects_invalid_precondition_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "values must be positive"):
            BenchmarkConfig(precondition_dimension=0).validate()
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            BenchmarkConfig(precondition_iterations=-1).validate()

    def test_accepts_paged_flashinfer_backend(self) -> None:
        config = BenchmarkConfig(
            attention_backend="flashinfer-paged", page_size=32
        )
        config.validate()
        self.assertEqual(config.page_size, 32)

    def test_default_tolerance_tracks_wire_dtype(self) -> None:
        fp16 = BenchmarkConfig(dtype="float16")
        bf16 = BenchmarkConfig(dtype="bfloat16")
        self.assertEqual((fp16.atol, fp16.rtol), (2e-3, 2e-3))
        self.assertEqual((bf16.atol, bf16.rtol), (2e-2, 2e-2))

    def test_percentile_interpolates_ordered_samples(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.99), 2.98)

    def test_phase_residual_subtracts_kernels_and_clamps_noise(self) -> None:
        residual = _residual_samples(
            [1.0, 2.0], [0.2, 1.8], [0.3, 0.3]
        )
        self.assertAlmostEqual(residual[0], 0.5)
        self.assertEqual(residual[1], 0.0)

    def test_phase_residual_rejects_mismatched_sample_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample counts must match"):
            _residual_samples([1.0, 2.0], [0.2])

    def test_route_residual_tracks_each_strategy_critical_path(self) -> None:
        sequential = _route_residual_samples(
            [2.0], [0.8], [0.3], [0.1], strategy="sequential"
        )
        overlap = _route_residual_samples(
            [2.0], [0.8], [0.3], [0.1], strategy="overlap"
        )
        fused = _route_residual_samples(
            [2.0], [0.8], [], [0.4], strategy="fused"
        )
        self.assertAlmostEqual(sequential[0], 0.8)
        self.assertAlmostEqual(overlap[0], 1.1)
        self.assertAlmostEqual(fused[0], 0.8)

    def test_plan_command_does_not_import_torch(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output):
            status = main(
                [
                    "plan",
                    "--prefix-tokens",
                    "128",
                    "--query-heads",
                    "8",
                    "--kv-heads",
                    "2",
                    "--head-dim",
                    "64",
                    "--attention-backend",
                    "flashinfer-paged",
                    "--page-size",
                    "32",
                    "--route-strategy",
                    "overlap",
                ]
            )
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["workload"]["prefix_tokens"], 128)
        self.assertEqual(
            report["workload"]["attention_backend"], "flashinfer-paged"
        )
        self.assertEqual(report["workload"]["page_size"], 32)
        self.assertEqual(report["workload"]["route_strategy"], "overlap")
        self.assertGreater(report["payload_bytes"]["stage_kv_total"], 0)

    def test_run_reports_environment_failure_as_exit_two(self) -> None:
        error = io.StringIO()
        with (
            patch(
                "integration.two_gpu_smoke.run_benchmark",
                side_effect=RuntimeError("CUDA unavailable"),
            ),
            patch("sys.stderr", error),
        ):
            status = main(["run", "--iterations", "1", "--warmup", "0"])
        self.assertEqual(status, 2)
        self.assertIn("CUDA unavailable", error.getvalue())


if __name__ == "__main__":
    unittest.main()
