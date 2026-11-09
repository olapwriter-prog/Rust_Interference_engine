//! GPU backend validation against the (reference-validated) CPU backend.
//! Skips when no GPU adapter is available. CI uses Mesa lavapipe (software Vulkan).

use rie_core::engine::load_model_from;
use rie_core::gguf::GgufFile;
use rie_core::model::cpu::CpuBackend;
use rie_core::model::Backend;
use rie_core::quant::GgmlType;
use rie_core::testing::SyntheticSpec;
use rie_core::threadpool::{PoolConfig, ThreadPool};
use rie_gpu::{GpuBackend, GpuConfig};
use std::sync::Arc;
use std::time::Duration;

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

fn gpu_available() -> bool {
    let ok = !rie_gpu::list_adapters().is_empty();
    if !ok {
        eprintln!("no GPU adapter: skipping");
    }
    ok
}

fn compare(spec: SyntheticSpec, tol: f32) {
    compare_kv(spec, tol, false)
}

fn compare_kv(spec: SyntheticSpec, tol: f32, kv_q8: bool) {
    let (model, _) = load_model_from(Arc::new(GgufFile::from_bytes(spec.build()).unwrap())).unwrap();
    let pool = Arc::new(ThreadPool::new(PoolConfig { n_threads: 2, cores: vec![], spin: Duration::from_micros(50) }));
    let mut cpu = CpuBackend::new(model.clone(), pool, 64, 4);
    let mut gpu = GpuBackend::new(model.clone(), GpuConfig { n_ctx: 64, max_batch: 4, low_power: false, kv_q8 }).unwrap();
    let nv = spec.n_vocab;
    let tokens: Vec<u32> = (0..11u32).map(|i| (i * 53 + 7) % nv as u32).collect();
    let (mut lc, mut lg) = (vec![0.0; nv], vec![0.0; nv]);
    // batched prefill in chunks of 4, then single-token decode
    let mut pos = 0;
    while pos < 8 {
        cpu.forward(&tokens[pos..pos + 4], pos, false, &mut lc).unwrap();
        gpu.forward(&tokens[pos..pos + 4], pos, false, &mut lg).unwrap();
        let err = max_rel_err(&lg, &lc);
        assert!(err < tol, "{} {:?} prefill@{pos}: rel err {err}", spec.arch, spec.wtype);
        pos += 4;
    }
    for t in 8..tokens.len() {
        cpu.forward(&tokens[t..t + 1], t, false, &mut lc).unwrap();
        gpu.forward(&tokens[t..t + 1], t, false, &mut lg).unwrap();
        let err = max_rel_err(&lg, &lc);
        if std::env::var("RIE_PRINT_ERR").is_ok() {
            eprintln!("{} {:?} decode@{t}: {err}", spec.arch, spec.wtype);
        }
        assert!(err < tol, "{} {:?} decode@{t}: rel err {err}", spec.arch, spec.wtype);
    }
}

#[test]
fn gpu_matches_cpu_all_architectures_f16() {
    if !gpu_available() {
        return;
    }
    for arch in ["llama", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "granite"] {
        let mut s = SyntheticSpec::tiny(arch, GgmlType::F16);
        if arch == "phi3" {
            s.rope_dim = 32;
        }
        compare(s, 3e-3);
    }
}

#[test]
fn gpu_quantized_formats() {
    if !gpu_available() {
        return;
    }
    // CPU quantizes activations to 8 bit, the GPU keeps them in f32: allow that noise.
    for ty in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q5_K, GgmlType::Q5_0, GgmlType::F32] {
        compare(SyntheticSpec::tiny("llama", ty), 0.08);
    }
}

#[test]
fn gpu_q8_kv_cache() {
    if !gpu_available() {
        return;
    }
    // 8-bit KV on the GPU vs f16 KV on the CPU: allow quantization noise.
    for arch in ["llama", "qwen3", "gemma3", "gemma2"] {
        compare_kv(SyntheticSpec::tiny(arch, GgmlType::F16), 0.05, true);
    }
}
