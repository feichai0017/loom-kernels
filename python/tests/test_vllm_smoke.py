import unittest

from integration.vllm_smoke import _compare_run_payloads


def run_payload(*, backend: str, tokens=None, logprobs=None):
    return {
        "schema": 1,
        "backend": backend,
        "model": "model",
        "revision": "revision",
        "dtype": "float16",
        "seed": 7,
        "max_tokens": 2,
        "vllm_version": "0.25.1",
        "cuda_device": "test-gpu",
        "median_generation_seconds": 1.0 if backend == "FLASH_ATTN" else 1.1,
        "loom_telemetry": (
            {
                "implementation_count": 1,
                "metadata_builder_count": 1,
                "forward_calls": 4,
                "forward_failures": 0,
                "step_generations": [2],
                "max_step_generation": 2,
                "implementations": [],
            }
            if backend == "CUSTOM"
            else None
        ),
        "sequences": [
            {
                "repetition": 0,
                "prompt_index": 0,
                "prompt": "prompt",
                "prompt_token_ids": [1, 2],
                "output_token_ids": [3, 4] if tokens is None else tokens,
                "sampled_logprobs": [-0.5, -0.25] if logprobs is None else logprobs,
            }
        ],
    }


class VllmSmokeComparisonTest(unittest.TestCase):
    def test_accepts_matching_tokens_and_close_logprobs(self) -> None:
        native = run_payload(backend="FLASH_ATTN")
        custom = run_payload(backend="CUSTOM", logprobs=[-0.500001, -0.249999])

        report = _compare_run_payloads(native, custom, logprob_atol=1e-5)

        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["custom_over_native_time_ratio"], 1.1)

    def test_rejects_generated_token_difference(self) -> None:
        native = run_payload(backend="FLASH_ATTN")
        custom = run_payload(backend="CUSTOM", tokens=[3, 5])

        report = _compare_run_payloads(native, custom, logprob_atol=1e-5)

        self.assertFalse(report["passed"])
        self.assertIn("sequence 0 generated token IDs differ", report["differences"])

    def test_rejects_logprob_difference_beyond_tolerance(self) -> None:
        native = run_payload(backend="FLASH_ATTN")
        custom = run_payload(backend="CUSTOM", logprobs=[-0.6, -0.25])

        report = _compare_run_payloads(native, custom, logprob_atol=1e-5)

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("logprob delta" in difference for difference in report["differences"])
        )

    def test_rejects_non_finite_logprob(self) -> None:
        native = run_payload(backend="FLASH_ATTN")
        custom = run_payload(backend="CUSTOM", logprobs=[float("nan"), -0.25])

        report = _compare_run_payloads(native, custom, logprob_atol=1e-5)

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("not finite" in difference for difference in report["differences"])
        )

    def test_rejects_custom_report_without_execution_telemetry(self) -> None:
        native = run_payload(backend="FLASH_ATTN")
        custom = run_payload(backend="CUSTOM")
        custom["loom_telemetry"] = None

        report = _compare_run_payloads(native, custom, logprob_atol=1e-5)

        self.assertFalse(report["passed"])
        self.assertIn(
            "custom report has no Loom execution telemetry",
            report["differences"],
        )


if __name__ == "__main__":
    unittest.main()
