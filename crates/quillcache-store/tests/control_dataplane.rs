//! The control plane driving the real `StoreDataPlane` tier machine — lives here
//! (an integration test of `quillcache-store`) because it depends on both the
//! control plane (in `quillcache-core`) and the store.

use quillcache_core::{
    CacheTier, ControlPlane, DataPlaneActionKind, EngineEndpoint, EngineKind, EngineRole,
    KvBlockKey, RequestShape, SloTarget,
};
use quillcache_store::{StoreDataPlane, StoreTierConfig};
use std::collections::HashMap;

fn engine() -> EngineEndpoint {
    EngineEndpoint {
        id: "vllm-a".to_string(),
        kind: EngineKind::Vllm,
        role: EngineRole::Aggregated,
        base_url: "http://127.0.0.1:8001".to_string(),
        model_id: "Qwen/Qwen3-0.6B".to_string(),
        tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
        tenant_id: "tenant-a".to_string(),
        locality_domain: "local".to_string(),
    }
}

#[test]
fn store_data_plane_updates_runtime_residency_via_control_plane() {
    let mut control = ControlPlane::new(vec![engine()]).with_data_plane(Box::new(
        StoreDataPlane::new(StoreTierConfig {
            hbm_capacity_bytes: 100,
            cpu_dram_capacity_bytes: 100,
            local_ssd_capacity_bytes: 100,
        }),
    ));
    let blocks = ["a", "b", "c", "d"]
        .iter()
        .enumerate()
        .map(|(idx, hash)| {
            KvBlockKey::new(
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "tenant-a",
                "root",
                *hash,
                idx as u32,
                64,
            )
        })
        .collect::<Vec<_>>();
    let request = RequestShape {
        id: "tiered".to_string(),
        model_id: "Qwen/Qwen3-0.6B".to_string(),
        tokenizer_id: "Qwen/Qwen3-0.6B".to_string(),
        adapter_id: None,
        tenant_id: "tenant-a".to_string(),
        session_id: None,
        blocks,
        estimated_decode_tokens: 16,
        slo: SloTarget::default(),
    };

    let actions = control.observe_placement("vllm-a", &request, 100);
    assert!(actions
        .iter()
        .any(|action| action.kind == DataPlaneActionKind::Evict));
    let tiers = control
        .residency()
        .snapshot()
        .into_iter()
        .map(|entry| (entry.key.block_hash, entry.tier))
        .collect::<HashMap<_, _>>();
    assert_eq!(tiers.len(), 3);
    assert_eq!(tiers.get("d"), Some(&CacheTier::Hbm));
    assert_eq!(tiers.get("c"), Some(&CacheTier::CpuDram));
    assert_eq!(tiers.get("b"), Some(&CacheTier::LocalSsd));
    assert!(!tiers.contains_key("a"));
}
