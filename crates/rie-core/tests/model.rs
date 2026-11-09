//! End-to-end forward-pass validation against an independent, naive f32 implementation.

mod common;
use common::*;
use rie_core::engine::{load_model_from, Engine, GenerateParams};
use rie_core::gguf::GgufFile;
use rie_core::model::cpu::CpuBackend;
use rie_core::model::Backend;
use rie_core::sampler::SamplerParams;
use rie_core::testing::SyntheticSpec;
use rie_core::threadpool::{PoolConfig, ThreadPool};
use std::sync::Arc;
use std::time::Duration;

/// Naive reference transformer: everything in f32, no KV cache tricks, recomputes the
/// whole sequence. Written independently of the engine (straight from the model math).
struct Reference {
    g: GgufFile,
    arch: String,
    /// Quantize matmul inputs to 8 bits exactly like the engine (isolates kernel bugs
    /// from the inherent activation-quantization error).
    emulate_act_quant: bool,
    wtype: GgmlType,
}

fn quantize_like_engine(ty: GgmlType, x: &[f32]) -> Vec<f32> {
    use rie_core::quant::*;
    match ty.act_format() {
        ActFormat::F32 => x.to_vec(),
        ActFormat::Q8_0 => {
            let mut b = vec![BlockQ8_0::default(); x.len() / QK];
            quantize_row_q8_0(x, &mut b);
            dequantize_q8_0(&b)
        }
        ActFormat::Q8K => {
            let mut b = vec![BlockQ8K::default(); x.len() / QK_K];
            quantize_row_q8_k(x, &mut b);
            dequantize_q8_k(&b)
        }
    }
}

impl Reference {
    fn t(&self, name: &str) -> Option<(Vec<f32>, GgmlType)> {
        let ti = self.g.tensor(name)?;
        let n = ti.n_elements() as usize;
        let mut out = vec![0.0; n];
        rie_core::quant::dequantize_row(ti.ty, self.g.tensor_data(ti), &mut out);
        Some((out, ti.ty))
    }
    fn u(&self, k: &str) -> usize {
        self.g.get_u64(&format!("{}.{k}", self.arch)).unwrap() as usize
    }

    fn logits(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let a = self.arch.as_str();
        let e = self.u("embedding_length");
        let nl = self.u("block_count");
        let nh = self.u("attention.head_count");
        let nkv = self.u("attention.head_count_kv");
        let hd = self.u("attention.key_length");
        let ff = self.u("feed_forward_length");
        let rd = self.u("rope.dimension_count");
        let eps = 1e-6f32;
        let gemma = a.starts_with("gemma");
        let neox = a != "llama" && a != "granite";
        let swa_win = self.g.get_u64(&format!("{a}.attention.sliding_window")).unwrap_or(0) as usize;
        let pattern = match a {
            "gemma2" => 2,
            "gemma3" => 6,
            _ => 0,
        };
        let granite = a == "granite";
        let gf = |k: &str, d: f32| self.g.get_f32(&format!("{a}.{k}")).unwrap_or(d);
        let embd_scale = if gemma { (e as f32).sqrt() } else if granite { gf("embedding_scale", 1.0) } else { 1.0 };
        let res_scale = if granite { gf("residual_scale", 1.0) } else { 1.0 };
        let attn_scale = if granite { gf("attention.scale", 1.0) } else { 1.0 / (hd as f32).sqrt() };
        let softcap_attn = if a == "gemma2" { Some(50.0f32) } else { None };
        let softcap_final = if a == "gemma2" { Some(30.0f32) } else { None };
        let logit_scale = if granite { gf("logit_scale", 1.0) } else { 1.0 };
        let base_global = gf("rope.freq_base", 10000.0);

        let rms = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let s = 1.0 / (ss + eps).sqrt();
            x.iter().zip(w).map(|(v, w)| v * s * w).collect()
        };
        let emulate = self.emulate_act_quant;
        let wtype = self.wtype;
        let matvec = |w: &[f32], cols: usize, x: &[f32]| -> Vec<f32> {
            let xq = if emulate { quantize_like_engine(wtype, x) } else { x.to_vec() };
            w.chunks(cols).map(|r| r.iter().zip(&xq).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32).collect()
        };
        let rope = |x: &mut [f32], pos: usize, base: f32| {
            for i in 0..rd / 2 {
                let theta = pos as f32 * base.powf(-2.0 * i as f32 / rd as f32);
                let (s, c) = theta.sin_cos();
                let (i0, i1) = if neox { (i, i + rd / 2) } else { (2 * i, 2 * i + 1) };
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * c - x1 * s;
                x[i1] = x0 * s + x1 * c;
            }
        };
        let (embd, embd_ty) = self.t("token_embd.weight").unwrap();
        let (onorm, _) = self.t("output_norm.weight").unwrap();
        let (out_w, out_ty) = self.t("output.weight").unwrap_or_else(|| (embd.clone(), embd_ty));
        let n = tokens.len();
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| embd[t as usize * e..(t as usize + 1) * e].iter().map(|v| v * embd_scale).collect()).collect();
        let qd = nh * hd;
        let kvd = nkv * hd;
        for il in 0..nl {
            let p = |s: &str| format!("blk.{il}.{s}");
            let get = |s: &str| self.t(&p(s)).map(|t| t.0);
            let (wq, wk, wv) = match get("attn_qkv.weight") {
                Some(f) => (f[..qd * e].to_vec(), f[qd * e..(qd + kvd) * e].to_vec(), f[(qd + kvd) * e..].to_vec()),
                None => (get("attn_q.weight").unwrap(), get("attn_k.weight").unwrap(), get("attn_v.weight").unwrap()),
            };
            let (wg, wu) = match get("ffn_gate.weight") {
                Some(g) => (g, get("ffn_up.weight").unwrap()),
                None => {
                    let f = get("ffn_up.weight").unwrap();
                    (f[..ff * e].to_vec(), f[ff * e..].to_vec())
                }
            };
            let wo = get("attn_output.weight").unwrap();
            let wd = get("ffn_down.weight").unwrap();
            let is_swa = pattern > 0 && swa_win > 0 && il % pattern < pattern - 1;
            let base = if a == "gemma3" && is_swa { 10000.0 } else { base_global };
            let mut qs = Vec::new();
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            for (pos, x) in xs.iter().enumerate() {
                let h = rms(x, &get("attn_norm.weight").unwrap());
                let mut q = matvec(&wq, e, &h);
                let mut k = matvec(&wk, e, &h);
                let mut v = matvec(&wv, e, &h);
                if let Some(b) = get("attn_q.bias") {
                    q.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                    k.iter_mut().zip(&get("attn_k.bias").unwrap()).for_each(|(a, b)| *a += b);
                    v.iter_mut().zip(&get("attn_v.bias").unwrap()).for_each(|(a, b)| *a += b);
                }
                for hh in 0..nh {
                    let s = &mut q[hh * hd..(hh + 1) * hd];
                    if let Some(w) = get("attn_q_norm.weight") {
                        let r = rms(s, &w);
                        s.copy_from_slice(&r);
                    }
                    rope(s, pos, base);
                }
                for hh in 0..nkv {
                    let s = &mut k[hh * hd..(hh + 1) * hd];
                    if let Some(w) = get("attn_k_norm.weight") {
                        let r = rms(s, &w);
                        s.copy_from_slice(&r);
                    }
                    rope(s, pos, base);
                }
                if emulate {
                    // the engine stores K/V as f16
                    for x in k.iter_mut().chain(v.iter_mut()) {
                        *x = half::f16::from_f32(*x).to_f32();
                    }
                }
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }
            for i in 0..n {
                let mut att = vec![0.0f32; qd];
                for hh in 0..nh {
                    let g = hh / (nh / nkv);
                    let lo = if is_swa { (i + 1).saturating_sub(swa_win) } else { 0 };
                    let mut sc: Vec<f32> = (lo..=i)
                        .map(|j| {
                            let mut s: f32 = (0..hd).map(|d| qs[i][hh * hd + d] * ks[j][g * hd + d]).sum::<f32>() * attn_scale;
                            if let Some(c) = softcap_attn {
                                s = c * (s / c).tanh();
                            }
                            s
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = sc.iter_mut().map(|s| {
                        *s = (*s - m).exp();
                        *s
                    }).sum();
                    for (jj, j) in (lo..=i).enumerate() {
                        for d in 0..hd {
                            att[hh * hd + d] += sc[jj] / sum * vs[j][g * hd + d];
                        }
                    }
                }
                let mut o = matvec(&wo, qd, &att);
                if let Some(w) = get("post_attention_norm.weight") {
                    o = rms(&o, &w);
                }
                xs[i].iter_mut().zip(&o).for_each(|(a, b)| *a += b * res_scale);
                let h = rms(&xs[i], &get("ffn_norm.weight").unwrap());
                let gt = matvec(&wg, e, &h);
                let up = matvec(&wu, e, &h);
                let act: Vec<f32> = gt
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| {
                        let a = if gemma {
                            0.5 * g * (1.0 + (0.7978845608f32 * (g + 0.044715 * g * g * g)).tanh())
                        } else {
                            g / (1.0 + (-g).exp())
                        };
                        a * u
                    })
                    .collect();
                let mut d = matvec(&wd, ff, &act);
                if let Some(w) = get("post_ffw_norm.weight") {
                    d = rms(&d, &w);
                }
                xs[i].iter_mut().zip(&d).for_each(|(a, b)| *a += b * res_scale);
            }
        }
        xs.iter()
            .map(|x| {
                let h = rms(x, &onorm);
                let hq = if emulate { quantize_like_engine(out_ty, &h) } else { h };
                let mut l: Vec<f32> = out_w.chunks(e).map(|r| r.iter().zip(&hq).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32).collect();
                for v in l.iter_mut() {
                    *v /= logit_scale;
                    if let Some(c) = softcap_final {
                        *v = c * (*v / c).tanh();
                    }
                }
                l
            })
            .collect()
    }
}

fn check_arch(spec: SyntheticSpec, tol: f32, threads: usize) {
    let bytes = spec.build();
    let reference = Reference {
        g: GgufFile::from_bytes(bytes.clone()).unwrap(),
        arch: spec.arch.clone(),
        emulate_act_quant: true,
        wtype: spec.wtype,
    };
    let (model, _tok) = load_model_from(Arc::new(GgufFile::from_bytes(bytes).unwrap())).unwrap();
    let pool = Arc::new(ThreadPool::new(PoolConfig { n_threads: threads, cores: vec![], spin: Duration::from_micros(50) }));
    let tokens: Vec<u32> = (0..13u32).map(|i| (i * 37 + 5) % spec.n_vocab as u32).collect();
    let expect = reference.logits(&tokens);
    let nv = spec.n_vocab;

    // (a) batched prefill with batch 4 (exercises multi-batch + SWA ring buffer wrap)
    let mut be = CpuBackend::new(model.clone(), pool.clone(), 64, 4);
    let mut logits = vec![0.0; 4 * nv];
    let mut pos = 0;
    while pos < tokens.len() {
        let end = (pos + 4).min(tokens.len());
        be.forward(&tokens[pos..end], pos, true, &mut logits).unwrap();
        for t in pos..end {
            let err = max_rel_err(&logits[(t - pos) * nv..(t - pos + 1) * nv], &expect[t]);
            if std::env::var("RIE_PRINT_ERR").is_ok() { eprintln!("{} {:?} batched t{t} err {err}", spec.arch, spec.wtype); }
            assert!(err < tol, "{} {:?}: batched token {t}: rel err {err}", spec.arch, spec.wtype);
        }
        pos = end;
    }
    // (b) token-by-token decoding
    let mut be = CpuBackend::new(model, pool, 64, 1);
    let mut l1 = vec![0.0; nv];
    for (t, &tok) in tokens.iter().enumerate() {
        be.forward(&[tok], t, false, &mut l1).unwrap();
        let err = max_rel_err(&l1, &expect[t]);
        assert!(err < tol, "{} {:?}: decode token {t}: rel err {err}", spec.arch, spec.wtype);
    }
}

#[test]
fn llama_f32_exact() {
    check_arch(SyntheticSpec::tiny("llama", GgmlType::F32), 2e-3, 3);
}

#[test]
fn all_architectures_f16() {
    for arch in ["llama", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "granite"] {
        let mut s = SyntheticSpec::tiny(arch, GgmlType::F16);
        if arch == "phi3" {
            s.rope_dim = 32; // partial rotary like Phi-4-mini
        }
        check_arch(s, 3e-3, 4);
    }
}

#[test]
fn quantized_weights() {
    for ty in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q5_0, GgmlType::Q4_1, GgmlType::Q3_K, GgmlType::Q2_K] {
        // the reference emulates 8-bit activations and the f16 KV cache, so remaining
        // differences are float summation order and rare rounding-tie flips
        check_arch(SyntheticSpec::tiny("llama", ty), 0.03, 2);
    }
    check_arch(SyntheticSpec::tiny("qwen3", GgmlType::Q4_K), 0.03, 4);
    check_arch(SyntheticSpec::tiny("gemma3", GgmlType::Q4_0), 0.03, 3);
}

#[test]
fn tied_embeddings_and_single_thread() {
    let mut s = SyntheticSpec::tiny("qwen2", GgmlType::F16);
    s.tied_output = true;
    check_arch(s, 3e-3, 1);
}

#[test]
fn engine_generation_prefix_reuse_and_determinism() {
    let spec = SyntheticSpec::tiny("llama", GgmlType::Q4_0);
    let (model, tok) = load_model_from(Arc::new(GgufFile::from_bytes(spec.build()).unwrap())).unwrap();
    let pool = Arc::new(ThreadPool::new(PoolConfig { n_threads: 2, cores: vec![], spin: Duration::from_micros(50) }));
    let mk = || Engine::with_backend(model.clone(), tok.clone(), Box::new(CpuBackend::new(model.clone(), pool.clone(), 128, 8)));
    let params = GenerateParams { max_tokens: 20, sampler: SamplerParams::greedy(), ..Default::default() };
    let mut e1 = mk();
    let mut out1 = String::new();
    let st = e1.generate("the phone is fast", &params, |s| {
        out1.push_str(s);
        true
    }).unwrap();
    assert_eq!(st.reused_tokens, 0);
    assert!(st.generated_tokens > 0);
    // same prompt again: prefix reused, identical greedy output
    let mut out2 = String::new();
    let st2 = e1.generate("the phone is fast", &params, |s| {
        out2.push_str(s);
        true
    }).unwrap();
    assert!(st2.reused_tokens > 0, "expected KV reuse");
    assert_eq!(out1, out2);
    // fresh engine gives the same result
    let mut e2 = mk();
    let mut out3 = String::new();
    e2.generate("the phone is fast", &params, |s| {
        out3.push_str(s);
        true
    }).unwrap();
    assert_eq!(out1, out3);
    // tokenizer round trip on the SPM byte vocabulary
    let ids = tok.encode("héllo phone", true, false);
    assert_eq!(ids[0], 1);
    assert_eq!(tok.decode(&ids[1..], false).trim_start(), "héllo phone");
}
