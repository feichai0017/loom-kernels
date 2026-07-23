from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from loom_kernels.vllm import (
    ACT_QUANT_OVERRIDE_ENV,
    ACT_QUANT_OVERRIDE_KEY,
    DEFAULT_PROVIDER,
    GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY,
    GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY,
    MIN_P_OVERRIDE_ENV,
    MIN_P_OVERRIDE_KEY,
    PAGED_DECODE_OVERRIDE_ENV,
    PAGED_DECODE_OVERRIDE_KEY,
    ROPE_PAGED_KV_OVERRIDE_KEY,
    SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY,
    SILU_OVERRIDE_ENV,
    SILU_OVERRIDE_KEY,
    SUPPORTED_VLLM_SERIES,
    configure_vllm_rope_paged_kv,
    installed_vllm_version,
    provider_metadata,
    register_vllm_ir,
    register_vllm_min_p,
    register_vllm_paged_decode_attention,
    register_vllm_greedy_sample_logprobs,
    register_vllm_greedy_speculative_verify,
    register_vllm_rope_paged_kv,
    register_vllm_selected_token_logprobs,
    register_vllm_silu_and_mul,
    register_vllm_silu_and_mul_dynamic_fp8,
    supports_installed_vllm,
    supports_vllm_paged_decode_shape,
)


def test_installed_vllm_series_is_supported():
    assert SUPPORTED_VLLM_SERIES == ((0, 24), (0, 25))
    assert supports_installed_vllm()
    assert installed_vllm_version() is not None
    assert provider_metadata()["vllm_supported"] is True


def test_unqualified_vllm_series_is_rejected(monkeypatch):
    import loom_kernels.vllm as integration
    import loom_kernels.vllm._runtime as runtime

    monkeypatch.setattr(
        runtime, "installed_vllm_version", lambda: "0.26.0"
    )
    assert integration.supports_installed_vllm() is False
    assert integration.register_vllm_ir() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_greedy_sample_logprobs_fast_path_matches_sampler_semantics():
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_greedy_sample_logprobs()
        == GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY
    )
    logits = torch.randn((5, 4096), device="cuda", dtype=torch.float32)
    metadata = SimpleNamespace(
        all_greedy=True,
        max_num_logprobs=0,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
    )
    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    output = Sampler().forward(logits, metadata)
    expected_ids = logits.argmax(-1).to(torch.int32)
    expected_logprobs = logits.log_softmax(-1).gather(
        -1, expected_ids.long().unsqueeze(-1)
    )
    torch.cuda.synchronize()

    assert torch.equal(output.sampled_token_ids[:, 0], expected_ids)
    assert output.logprobs_tensors is not None
    torch.testing.assert_close(
        output.logprobs_tensors.logprobs,
        expected_logprobs,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    expected_ranks = (
        logits
        >= logits.gather(-1, expected_ids.long().unsqueeze(-1))
    ).sum(dim=-1)
    assert torch.equal(
        output.logprobs_tensors.selected_token_ranks,
        expected_ranks,
    )
    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 1
    assert provider_metadata()["greedy_sample_logprobs_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_greedy_speculative_verify_matches_rejection_semantics(
    monkeypatch,
):
    from vllm.v1.sample import rejection_sampler

    import loom_kernels.vllm.speculative as speculative_integration
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert (
        register_vllm_greedy_speculative_verify()
        == GREEDY_SPECULATIVE_VERIFY_OVERRIDE_KEY
    )
    draft = torch.tensor(
        [10, 11, 12, 20, 21, 22, 23],
        dtype=torch.int32,
        device="cuda",
    )
    target_ids = torch.tensor(
        [10, 99, 12, 20, 21, 22, 23],
        dtype=torch.int64,
        device="cuda",
    )
    target_logits = torch.full(
        (7, 128), -100.0, dtype=torch.float32, device="cuda"
    )
    target_logits.scatter_(1, target_ids.unsqueeze(1), 100.0)
    bonus = torch.tensor(
        [[100], [101], [102]], dtype=torch.int32, device="cuda"
    )
    cumulative = torch.tensor([3, 3, 7], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(all_greedy=True)

    reset_launch_count(Operator.GREEDY_SPECULATIVE_VERIFY)
    output = rejection_sampler.rejection_sample(
        draft,
        [3, 0, 4],
        4,
        cumulative,
        None,
        target_logits,
        bonus,
        metadata,
    )
    torch.cuda.synchronize()

    assert output.tolist() == [
        [10, 99, -1, -1, -1],
        [101, -1, -1, -1, -1],
        [20, 21, 22, 23, 102],
    ]
    assert launch_count(Operator.GREEDY_SPECULATIVE_VERIFY) == 1
    assert provider_metadata()["greedy_speculative_verify_override"] is True

    sentinel = torch.empty(0, dtype=torch.int32, device="cuda")
    fallback_calls = 0

    def fallback(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return sentinel

    monkeypatch.setattr(
        speculative_integration,
        "_GREEDY_SPECULATIVE_VERIFY_ORIGINAL",
        fallback,
    )
    metadata.all_greedy = False
    reset_launch_count(Operator.GREEDY_SPECULATIVE_VERIFY)
    fallback_output = rejection_sampler.rejection_sample(
        draft,
        [3, 0, 4],
        4,
        cumulative,
        None,
        target_logits,
        bonus,
        metadata,
    )

    assert fallback_output is sentinel
    assert fallback_calls == 1
    assert launch_count(Operator.GREEDY_SPECULATIVE_VERIFY) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_selected_token_fast_path_preserves_engine_selection(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    assert (
        register_vllm_selected_token_logprobs()
        == SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY
    )
    logits = torch.randn((5, 4096), device="cuda", dtype=torch.bfloat16)
    sampled = torch.tensor([0, 17, 2048, 4095, 7], device="cuda")
    metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        max_num_logprobs=0,
        logprob_token_ids=None,
        top_k=torch.full((5,), 50, device="cuda", dtype=torch.int32),
        top_p=torch.full((5,), 0.9, device="cuda"),
        no_penalties=False,
    )
    sampler = Sampler()
    observed = {}

    def apply_processors(sampling_logits, received_metadata, predict_bonus_token):
        observed["input_dtype"] = sampling_logits.dtype
        observed["metadata"] = received_metadata
        observed["predict_bonus_token"] = predict_bonus_token
        sampling_logits.add_(1.0)
        return sampling_logits

    def sample(sampling_logits, received_metadata):
        observed["sample_logits_dtype"] = sampling_logits.dtype
        observed["sample_metadata"] = received_metadata
        return sampled, None

    monkeypatch.setattr(sampler, "apply_logits_processors", apply_processors)
    monkeypatch.setattr(sampler, "sample", sample)
    output = sampler.forward(logits, metadata, predict_bonus_token=True)
    expected_logprobs = logits.float().log_softmax(-1).gather(
        -1, sampled.unsqueeze(-1)
    )
    selected = logits.float().gather(-1, sampled.unsqueeze(-1))
    expected_ranks = (logits.float() >= selected).sum(dim=-1)
    torch.cuda.synchronize()

    assert observed == {
        "input_dtype": torch.float32,
        "metadata": metadata,
        "predict_bonus_token": True,
        "sample_logits_dtype": torch.float32,
        "sample_metadata": metadata,
    }
    assert torch.equal(output.sampled_token_ids[:, 0], sampled.to(torch.int32))
    assert output.logprobs_tensors is not None
    torch.testing.assert_close(
        output.logprobs_tensors.logprobs,
        expected_logprobs,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert torch.equal(
        output.logprobs_tensors.selected_token_ranks, expected_ranks
    )
    assert provider_metadata()["selected_token_logprobs_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_selected_token_path_handles_processed_greedy_batches(monkeypatch):
    from vllm.v1.sample.sampler import Sampler

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    register_vllm_selected_token_logprobs()
    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    reset_launch_count(Operator.SELECTED_TOKEN_LOGPROBS)
    logits = torch.randn((3, 1024), device="cuda", dtype=torch.float16)
    sampled = torch.tensor([7, 511, 1023], device="cuda")
    metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        max_num_logprobs=0,
        logprob_token_ids=None,
        no_penalties=False,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        thinking_budget_state_holder=None,
        top_k=None,
        top_p=None,
    )
    sampler = Sampler()
    monkeypatch.setattr(
        sampler,
        "apply_logits_processors",
        lambda sampling_logits, _metadata, _predict_bonus: sampling_logits,
    )
    monkeypatch.setattr(
        sampler,
        "sample",
        lambda _sampling_logits, _metadata: (sampled, None),
    )

    output = sampler.forward(logits, metadata)
    torch.cuda.synchronize()

    assert torch.equal(output.sampled_token_ids[:, 0], sampled.to(torch.int32))
    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 0
    assert launch_count(Operator.SELECTED_TOKEN_LOGPROBS) == 1


def test_configures_vllm_rope_paged_kv_fusion():
    from vllm.config import CompilationConfig
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    assert register_vllm_rope_paged_kv() == ROPE_PAGED_KV_OVERRIDE_KEY
    config = configure_vllm_rope_paged_kv(max_token_num=128)

    assert isinstance(config, CompilationConfig)
    assert config.pass_config.fuse_rope_kvcache is True
    assert config.pass_config.rope_kvcache_fusion_max_token_num == 128
    assert config.splitting_ops == []
    assert "+rotary_embedding" in config.custom_ops
    assert "+quant_fp8" in config.custom_ops
    assert FlashAttentionImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm.rope_kv"
    )
    assert FlashInferImpl.fused_rope_kvcache_supported.__module__ == (
        "loom_kernels.vllm.rope_kv"
    )
    for cache_dtype in ("auto", "fp8", "fp8_e4m3", torch.bfloat16):
        attention = SimpleNamespace(
            attn_type=AttentionType.DECODER,
            kv_cache_dtype=cache_dtype,
            kv_sharing_target_layer_name=None,
        )
        assert FlashAttentionImpl.fused_rope_kvcache_supported(attention)
        assert FlashInferImpl.fused_rope_kvcache_supported(attention)
    for cache_dtype in (
        "fp8_e5m2",
        "fp8_per_token_head",
        "int8",
        "nvfp4",
    ):
        attention = SimpleNamespace(
            attn_type=AttentionType.DECODER,
            kv_cache_dtype=cache_dtype,
            kv_sharing_target_layer_name=None,
        )
        assert not FlashAttentionImpl.fused_rope_kvcache_supported(attention)
        assert not FlashInferImpl.fused_rope_kvcache_supported(attention)
    shared_attention = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="fp8",
        kv_sharing_target_layer_name="model.layers.0.self_attn",
    )
    assert not FlashAttentionImpl.fused_rope_kvcache_supported(shared_attention)
    encoder_attention = SimpleNamespace(
        attn_type=AttentionType.ENCODER,
        kv_cache_dtype="fp8",
        kv_sharing_target_layer_name=None,
    )
    assert not FlashAttentionImpl.fused_rope_kvcache_supported(encoder_attention)
    assert provider_metadata()["rope_paged_kv_override"] is True


def test_vllm_paged_decode_shape_gate_is_conservative():
    qualified = {
        "dtype": torch.bfloat16,
        "batch": 32,
        "query_heads": 32,
        "kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_sequence_length": 32,
    }
    assert supports_vllm_paged_decode_shape(**qualified)
    for field, rejected in (
        ("dtype", torch.float32),
        ("batch", 129),
        ("query_heads", 64),
        ("kv_heads", 4),
        ("head_size", 64),
        ("block_size", 8),
        ("max_sequence_length", 64),
    ):
        candidate = {**qualified, field: rejected}
        assert not supports_vllm_paged_decode_shape(**candidate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_paged_decode_fast_path_matches_flash_attention():
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    batch = 8
    context = 32
    block_size = 16
    max_blocks = context // block_size
    num_blocks = batch * max_blocks
    query = torch.randn((batch, 32, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((batch, 8, 128), device="cuda", dtype=query.dtype)
    value = torch.empty_like(key)
    kv_cache = torch.randn(
        (num_blocks, 2, block_size, 8, 128),
        device="cuda",
        dtype=query.dtype,
    )
    block_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
    block_table = block_table.reshape(batch, max_blocks).to(torch.int32)
    seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        max_seq_len=context,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch, device="cuda", dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    attention = FlashAttentionImpl(
        num_heads=32,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    expected = torch.empty((batch, 32, 128), device="cuda", dtype=query.dtype)
    attention.forward(
        layer, query, key, value, kv_cache, metadata, expected
    )
    # Real FA3 decode metadata carries an AOT scheduler tensor and represents
    # the inactive DCP context length as zero. Neither changes attention
    # semantics, so Loom must not reject the otherwise qualified path.
    metadata.max_dcp_context_kv_len = 0
    metadata.scheduler_metadata = torch.zeros(
        (1,), device="cuda", dtype=torch.int32
    )

    assert (
        register_vllm_paged_decode_attention()
        == PAGED_DECODE_OVERRIDE_KEY
    )
    reset_launch_count(Operator.PAGED_DECODE_ATTENTION)
    actual = torch.empty_like(expected)
    returned = attention.forward(
        layer, query, key, value, kv_cache, metadata, actual
    )
    torch.cuda.synchronize()

    assert returned is actual
    assert launch_count(Operator.PAGED_DECODE_ATTENTION) == 1
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
    assert provider_metadata()["paged_decode_override"] is True


def test_paged_decode_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(PAGED_DECODE_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["paged_decode_override_requested"] is False
    monkeypatch.setenv(PAGED_DECODE_OVERRIDE_ENV, "true")
    assert provider_metadata()["paged_decode_override_requested"] is True


def test_registers_inplace_fused_add_rms_norm_provider():
    from vllm import ir

    assert register_vllm_ir() == DEFAULT_PROVIDER
    assert DEFAULT_PROVIDER in ir.ops.fused_add_rms_norm.impls
    implementation = ir.ops.fused_add_rms_norm.impls[DEFAULT_PROVIDER]
    assert implementation.inplace is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_min_p_processor_uses_loom_without_probability_tensor():
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_min_p() == MIN_P_OVERRIDE_KEY
    reset_launch_count(Operator.MIN_P_FILTER)
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 31
    processor.min_p = torch.linspace(0.0, 0.8, 32, device="cuda").unsqueeze(1)
    logits = torch.randn((32, 151936), device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    expected = logits.clone().masked_fill_(
        probabilities
        < probabilities.amax(dim=-1, keepdim=True) * processor.min_p,
        -float("inf"),
    )

    returned = processor.apply(logits)
    torch.cuda.synchronize()

    assert returned is logits
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    assert launch_count(Operator.MIN_P_FILTER) == 1
    assert provider_metadata()["min_p_override"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_min_p_processor_falls_back_below_measured_fast_path():
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    assert register_vllm_min_p() == MIN_P_OVERRIDE_KEY
    reset_launch_count(Operator.MIN_P_FILTER)
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 2
    processor.min_p = torch.tensor([[0.0], [0.2], [0.8]], device="cuda")
    logits = torch.randn((3, 4096), device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    expected = logits.clone().masked_fill_(
        probabilities
        < probabilities.amax(dim=-1, keepdim=True) * processor.min_p,
        -float("inf"),
    )

    returned = processor.apply(logits)
    torch.cuda.synchronize()

    assert returned is logits
    assert torch.equal(logits, expected)
    assert launch_count(Operator.MIN_P_FILTER) == 0


def test_min_p_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(MIN_P_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["min_p_override_requested"] is False
    monkeypatch.setenv(MIN_P_OVERRIDE_ENV, "yes")
    assert provider_metadata()["min_p_override_requested"] is True


def test_registers_vllm_silu_and_mul_override():
    from vllm.model_executor.custom_op import op_registry_oot

    assert register_vllm_silu_and_mul() == SILU_OVERRIDE_KEY
    assert SILU_OVERRIDE_KEY in op_registry_oot


def test_silu_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(SILU_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_override_requested"] is False
    monkeypatch.setenv(SILU_OVERRIDE_ENV, "true")
    assert provider_metadata()["silu_and_mul_override_requested"] is True


def test_registers_vllm_silu_and_mul_dynamic_fp8_fusion():
    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    assert (
        register_vllm_silu_and_mul_dynamic_fp8() == ACT_QUANT_OVERRIDE_KEY
    )
    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    assert FUSED_OPS[kFp8Dynamic64Sym] == implementation
    assert FUSED_OPS[kFp8Dynamic128Sym] == implementation


def test_act_quant_override_metadata_tracks_opt_in(monkeypatch):
    monkeypatch.delenv(ACT_QUANT_OVERRIDE_ENV, raising=False)
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is False
    monkeypatch.setenv(ACT_QUANT_OVERRIDE_ENV, "on")
    assert provider_metadata()["silu_and_mul_fp8_override_requested"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_activation_quant_pattern_rewrites_to_loom():
    from vllm.compilation.passes.fusion.act_quant_fusion import (
        SiluMulBlockQuantPattern,
    )
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        enable_fake_mode,
    )
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )

    config = VllmConfig()
    with set_current_vllm_config(config):
        register_vllm_silu_and_mul_dynamic_fp8()
        pattern = SiluMulBlockQuantPattern(kFp8Dynamic128Sym)
        fusion_pass = VllmFusionPatternMatcherPass(
            config, "loom_activation_quant_test"
        )
        fusion_pass.register(pattern)

        @enable_fake_mode
        def trace_official_pattern():
            return fusion_pass._trace_fn(pattern.pattern, pattern.get_inputs())

        graph_module = trace_official_pattern()
        fusion_pass(graph_module.graph)

    loom_operator = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    loom_target_present = any(
        node.op == "call_function"
        and node.args
        and node.args[0] == loom_operator
        for node in graph_module.graph.nodes
    )
    assert fusion_pass.matched_count == 1
    assert loom_target_present


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_silu_layer_dispatches_to_loom():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul

    register_vllm_silu_and_mul()
    with set_current_vllm_config(VllmConfig()):
        activation = SiluAndMul()
    assert type(activation).__name__ == "LoomSiluAndMul"

    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(expected, input_tensor)
    actual = activation(input_tensor)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_ir_dispatches_to_loom_provider():
    from vllm import ir
    from vllm.platforms import current_platform

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    input_tensor = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    expected_residual = (input_tensor.float() + residual.float()).to(torch.bfloat16)

    reset_launch_count(Operator.ADD_RMS_NORM)
    with operation.set_priority([DEFAULT_PROVIDER, "native"]):
        assert (
            operation.dispatch(input_tensor, residual, weight, 1.0e-5).provider
            == DEFAULT_PROVIDER
        )
        output, residual_output = operation.maybe_inplace(
            input_tensor, residual, weight, 1.0e-5
        )
    torch.cuda.synchronize()

    assert output is input_tensor
    assert residual_output is residual
    assert launch_count(Operator.ADD_RMS_NORM) == 1
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 4096), (8, 4096), (128, 4096), (8, 8192)])
def test_loom_is_bitwise_equal_to_vllm_cuda_provider(shape):
    from vllm import ir
    from vllm.platforms import current_platform

    register_vllm_ir()
    current_platform.import_ir_kernels()
    operation = ir.ops.fused_add_rms_norm
    if "vllm_c" not in operation.impls or not operation.impls["vllm_c"].supported:
        pytest.skip("vLLM CUDA provider is unavailable")

    torch.manual_seed(20260721)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    outputs = {}
    for provider in (DEFAULT_PROVIDER, "vllm_c"):
        provider_input = input_tensor.clone()
        provider_residual = residual.clone()
        with operation.set_priority([provider, "native"]):
            outputs[provider] = operation.maybe_inplace(
                provider_input, provider_residual, weight, 1.0e-5
            )
    torch.cuda.synchronize()

    loom_output, loom_residual = outputs[DEFAULT_PROVIDER]
    vllm_output, vllm_residual = outputs["vllm_c"]
    assert torch.equal(loom_output, vllm_output)
    assert torch.equal(loom_residual, vllm_residual)
