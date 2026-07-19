import io
import json
import unittest
from unittest.mock import patch

from loom_attention.two_gpu_smoke import (
    BenchmarkConfig,
    main,
    percentile,
    projected_transfer_bytes,
)


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
        self.assertEqual(payload["partial"], 16640)
        self.assertEqual(payload["stage_kv_total"], 16_777_216)
        self.assertLess(payload["route_query_total"], payload["stage_kv_total"])

    def test_rejects_invalid_gqa_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "kv_heads must divide query_heads"):
            BenchmarkConfig(query_heads=12, kv_heads=5).validate()

    def test_percentile_interpolates_ordered_samples(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.99), 2.98)

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
                ]
            )
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["workload"]["prefix_tokens"], 128)
        self.assertGreater(report["payload_bytes"]["stage_kv_total"], 0)

    def test_run_reports_environment_failure_as_exit_two(self) -> None:
        error = io.StringIO()
        with (
            patch(
                "loom_attention.two_gpu_smoke._run",
                side_effect=RuntimeError("CUDA unavailable"),
            ),
            patch("sys.stderr", error),
        ):
            status = main(["run", "--iterations", "1", "--warmup", "0"])
        self.assertEqual(status, 2)
        self.assertIn("CUDA unavailable", error.getvalue())


if __name__ == "__main__":
    unittest.main()
