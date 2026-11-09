//! Model conversion tools.
//!
//! * [`quantize_model`] — re-quantize a GGUF (typically F32/F16/BF16, 32 or 16 bits per
//!   weight) to Q4_0 or Q8_0 (4.5 / 8.5 bits per weight). A 4-bit model needs ~7× less
//!   RAM and memory bandwidth than F32, which is what bounds token speed on phones.
//! * [`moefy`] — convert a dense model into a mixture of experts ("MoEfication",
//!   Zhang et al. 2022): each feed-forward block's neurons are partitioned into `E`
//!   experts by clustering neurons that fire together on calibration text, and a router
//!   is fitted (ridge regression, no back-propagation) to predict which experts matter
//!   for a token. At run time only the top-`k` experts are evaluated, so feed-forward
//!   compute and weight reads drop to `k/E`. With `k = E` the converted model is exactly
//!   the original. Quality at `k < E` depends on how sparse the model's activations are;
//!   measure it (e.g. `rie perplexity`) before deploying, and prefer a short fine-tune
//!   of the converted model when the loss is too high.

use crate::engine::load_model_from;
use crate::error::{bail, Result};
use crate::gguf::{writer::GgufWriter, GgufFile, Value};
use crate::model::cpu::CpuBackend;
use crate::model::{Activation, Backend, Ffn};
use crate::ops::{gelu_tanh, silu};
use crate::quant::{dequantize_row, quantize, GgmlType};
use crate::threadpool::{PoolConfig, ThreadPool};
use std::sync::Arc;

fn file_type(t: GgmlType) -> u32 {
    match t {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q8_0 => 7,
        _ => 0,
    }
}

fn encode(t: GgmlType, x: &[f32]) -> Vec<u8> {
    match t {
        GgmlType::Q4_0 => quantize::q4_0(x),
        GgmlType::Q8_0 => quantize::q8_0(x),
        GgmlType::F16 => quantize::f16(x),
        _ => quantize::f32(x),
    }
}

/// All values of a tensor as f32.
fn tensor_f32(g: &GgufFile, name: &str) -> Result<(Vec<u64>, Vec<f32>)> {
    let Some(t) = g.tensor(name) else { bail!(Missing, "tensor {name}") };
    let (n, rows) = (t.row_len(), t.n_rows());
    let data = g.tensor_data(t);
    let rb = t.ty.row_bytes(n);
    let mut out = vec![0.0; n * rows];
    for r in 0..rows {
        dequantize_row(t.ty, &data[r * rb..(r + 1) * rb], &mut out[r * n..(r + 1) * n]);
    }
    Ok((t.dims.clone(), out))
}

fn copy_metadata(src: &GgufFile, w: &mut GgufWriter, skip: &[&str]) {
    let mut keys: Vec<&String> = src.metadata.keys().collect();
    keys.sort();
    for k in keys {
        if k == "general.alignment" || skip.contains(&k.as_str()) {
            continue;
        }
        w.kv(k, src.metadata[k].clone());
    }
}

/// Re-quantize every weight matrix to `target` (Q4_0, Q8_0 or F16). Norms, biases and
/// other vectors stay as they are; the output head (or the tied embedding) gets Q8_0
/// when the target is Q4_0, as it is the most sensitive matrix.
pub fn quantize_model(src: &GgufFile, target: GgmlType) -> Result<Vec<u8>> {
    if !matches!(target, GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::F16) {
        bail!(Unsupported, "quantize target must be Q4_0, Q8_0 or F16");
    }
    let mut w = GgufWriter::new();
    copy_metadata(src, &mut w, &["general.file_type"]);
    w.kv("general.file_type", Value::U32(file_type(target)));
    let tied = src.tensor("output.weight").is_none();
    for t in &src.tensors {
        let matrix = t.dims.len() >= 2 && t.name.ends_with(".weight") && !t.name.contains("norm") && !t.name.contains("ffn_gate_inp");
        let mut ty = target;
        if target == GgmlType::Q4_0 && (t.name == "output.weight" || (tied && t.name == "token_embd.weight")) {
            ty = GgmlType::Q8_0;
        }
        if !matrix || t.row_len() % ty.block_size() != 0 || t.ty == ty {
            w.tensor(&t.name, &t.dims, t.ty, src.tensor_data(t).to_vec());
            continue;
        }
        let (dims, x) = tensor_f32(src, &t.name)?;
        w.tensor(&t.name, &dims, ty, encode(ty, &x));
    }
    Ok(w.to_bytes())
}

#[derive(Debug, Clone)]
pub struct MoefyOptions {
    /// experts per layer
    pub n_expert: usize,
    /// experts evaluated per token
    pub n_used: usize,
    /// calibration tokens (text from the target domain); `None` = data-free heuristics
    pub calib: Option<Vec<u32>>,
    pub threads: usize,
    pub kmeans_iters: usize,
}

impl Default for MoefyOptions {
    fn default() -> Self {
        MoefyOptions { n_expert: 8, n_used: 2, calib: None, threads: 4, kmeans_iters: 8 }
    }
}

/// What happened to one layer.
#[derive(Debug, Clone)]
pub struct LayerReport {
    pub layer: usize,
    pub expert_size: usize,
    /// on the calibration tokens: share of feed-forward activation mass inside the
    /// experts the router picks (1.0 = routing loses nothing)
    pub routed_mass: Option<f32>,
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::kernels::dot_f32(a, b)
}

/// Balanced clustering of `n` unit feature vectors (`dim` each) into `k` groups of `n/k`.
fn balanced_kmeans(feat: &[f32], n: usize, dim: usize, k: usize, iters: usize) -> Vec<usize> {
    let size = n / k;
    let mut cent: Vec<f32> = (0..k).flat_map(|c| feat[(c * size) * dim..(c * size + 1) * dim].to_vec()).collect();
    // spread the initial centroids: farthest-point picks
    for c in 1..k {
        let mut best = (f32::INFINITY, 0);
        for i in 0..n {
            let f = &feat[i * dim..(i + 1) * dim];
            let m = (0..c).map(|j| dot(f, &cent[j * dim..(j + 1) * dim])).fold(f32::NEG_INFINITY, f32::max);
            if m < best.0 {
                best = (m, i);
            }
        }
        cent[c * dim..(c + 1) * dim].copy_from_slice(&feat[best.1 * dim..(best.1 + 1) * dim]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters.max(1) {
        let mut pairs: Vec<(f32, u32, u32)> = Vec::with_capacity(n * k);
        for i in 0..n {
            let f = &feat[i * dim..(i + 1) * dim];
            for c in 0..k {
                pairs.push((dot(f, &cent[c * dim..(c + 1) * dim]), i as u32, c as u32));
            }
        }
        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut done = vec![false; n];
        let mut fill = vec![0usize; k];
        for (_, i, c) in pairs {
            let (i, c) = (i as usize, c as usize);
            if !done[i] && fill[c] < size {
                done[i] = true;
                fill[c] += 1;
                assign[i] = c;
            }
        }
        cent.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            let c = assign[i];
            for d in 0..dim {
                cent[c * dim + d] += feat[i * dim + d];
            }
        }
        for c in 0..k {
            normalize(&mut cent[c * dim..(c + 1) * dim]);
        }
    }
    assign
}

/// Solve `(K + λI) α = S` for symmetric positive-definite `K` (n×n), `S` n×m (Cholesky).
fn solve_spd(mut kmat: Vec<f64>, n: usize, s: &[f64], m: usize) -> Vec<f64> {
    for j in 0..n {
        let mut d = kmat[j * n + j];
        for p in 0..j {
            d -= kmat[j * n + p] * kmat[j * n + p];
        }
        let d = d.max(1e-12).sqrt();
        kmat[j * n + j] = d;
        for i in j + 1..n {
            let mut v = kmat[i * n + j];
            for p in 0..j {
                v -= kmat[i * n + p] * kmat[j * n + p];
            }
            kmat[i * n + j] = v / d;
        }
    }
    let mut x = s.to_vec();
    for c in 0..m {
        for i in 0..n {
            let mut v = x[i * m + c];
            for p in 0..i {
                v -= kmat[i * n + p] * x[p * m + c];
            }
            x[i * m + c] = v / kmat[i * n + i];
        }
        for i in (0..n).rev() {
            let mut v = x[i * m + c];
            for p in i + 1..n {
                v -= kmat[p * n + i] * x[p * m + c];
            }
            x[i * m + c] = v / kmat[i * n + i];
        }
    }
    x
}

/// Convert a dense model into a mixture of experts. Returns the new GGUF image and a
/// per-layer report.
pub fn moefy(gguf: Arc<GgufFile>, opts: &MoefyOptions) -> Result<(Vec<u8>, Vec<LayerReport>)> {
    let (model, _) = load_model_from(gguf.clone())?;
    let c = &model.cfg;
    let (ne, k) = (opts.n_expert, opts.n_used);
    if ne < 2 || k == 0 || k > ne {
        bail!(Unsupported, "need 2 ≤ experts and 1 ≤ used ≤ experts");
    }
    if model.layers.iter().any(|l| !matches!(l.ffn, Ffn::Dense(_))) {
        bail!(Unsupported, "model already uses mixture-of-experts layers");
    }
    let ff = c.n_ff;
    if ff % ne != 0 {
        bail!(Unsupported, "feed-forward size {ff} is not divisible by {ne} experts");
    }
    let m = ff / ne;
    let e = c.n_embd;
    let act = |x: f32| match c.act {
        Activation::Silu => silu(x),
        Activation::GeluTanh => gelu_tanh(x),
    };

    // calibration: feed-forward inputs of every layer
    let calib: Option<(Vec<Vec<f32>>, usize)> = match &opts.calib {
        Some(toks) if toks.len() >= 8 => {
            let toks = &toks[..toks.len().min(c.ctx_train).min(1024)];
            let pool = Arc::new(ThreadPool::new(PoolConfig {
                n_threads: opts.threads.max(1),
                cores: vec![],
                spin: std::time::Duration::from_micros(50),
            }));
            let mut be = CpuBackend::new(model.clone(), pool, toks.len(), 64);
            be.capture_ffn_inputs();
            let mut logits = vec![0.0; c.n_vocab];
            for (i, ch) in toks.chunks(64).enumerate() {
                be.forward(ch, i * 64, false, &mut logits)?;
            }
            Some((be.take_ffn_inputs(), toks.len()))
        }
        _ => None,
    };

    let mut w = GgufWriter::new();
    let arch = gguf.get_str("general.architecture").unwrap_or("llama").to_string();
    let key = |s: &str| format!("{arch}.{s}");
    let (k_count, k_used, k_ff) = (key("expert_count"), key("expert_used_count"), key("expert_feed_forward_length"));
    copy_metadata(&gguf, &mut w, &[&k_count, &k_used, &k_ff, "rie.expert_gating"]);
    w.kv(&k_count, Value::U32(ne as u32));
    w.kv(&k_used, Value::U32(k as u32));
    w.kv(&k_ff, Value::U32(m as u32));
    w.kv("rie.expert_gating", Value::String("unit".into()));

    let ffn_tensor = |n: &str| {
        n.starts_with("blk.") && (n.ends_with(".ffn_gate.weight") || n.ends_with(".ffn_up.weight") || n.ends_with(".ffn_down.weight"))
    };
    for t in &gguf.tensors {
        if !ffn_tensor(&t.name) {
            w.tensor(&t.name, &t.dims, t.ty, gguf.tensor_data(t).to_vec());
        }
    }

    let mut reports = Vec::new();
    for (il, lw) in model.layers.iter().enumerate() {
        let Ffn::Dense(d) = &lw.ffn else { unreachable!() };
        let (g, u, dn) = (d.gate.to_f32(), d.up.to_f32(), d.down.to_f32());
        let x = calib.as_ref().map(|(xs, n)| (&xs[il], *n));
        // neuron activations on the calibration tokens: h[t][i] = act(g_i·x_t)·(u_i·x_t)
        let h: Option<Vec<f32>> = x.map(|(xs, n)| {
            let mut h = vec![0.0f32; n * ff];
            std::thread::scope(|sc| {
                let per = n.div_ceil(opts.threads.max(1));
                for (ci, hc) in h.chunks_mut(per * ff).enumerate() {
                    let (g, u) = (&g, &u);
                    sc.spawn(move || {
                        for (tt, row) in hc.chunks_mut(ff).enumerate() {
                            let xt = &xs[(ci * per + tt) * e..(ci * per + tt + 1) * e];
                            for i in 0..ff {
                                row[i] = act(dot(&g[i * e..(i + 1) * e], xt)) * dot(&u[i * e..(i + 1) * e], xt);
                            }
                        }
                    });
                }
            });
            h
        });
        // column norms of down: how much each neuron can move the output
        let dnorm: Vec<f32> = (0..ff).map(|i| (0..e).map(|r| dn[r * ff + i] * dn[r * ff + i]).sum::<f32>().sqrt()).collect();
        // neuron features for clustering
        let (feat, dim) = match (&h, x) {
            (Some(h), Some((_, n))) => {
                let mut f = vec![0.0f32; ff * n];
                for i in 0..ff {
                    for t in 0..n {
                        f[i * n + t] = h[t * ff + i].abs() * dnorm[i];
                    }
                    normalize(&mut f[i * n..(i + 1) * n]);
                }
                (f, n)
            }
            _ => {
                let mut f = g.clone();
                for i in 0..ff {
                    normalize(&mut f[i * e..(i + 1) * e]);
                }
                (f, e)
            }
        };
        let assign = balanced_kmeans(&feat, ff, dim, ne, opts.kmeans_iters);
        let members: Vec<Vec<usize>> = (0..ne).map(|cl| (0..ff).filter(|&i| assign[i] == cl).collect()).collect();

        // router
        let mut router = vec![0.0f32; ne * e];
        let mut routed_mass = None;
        match (&h, x) {
            (Some(h), Some((xs, n))) => {
                // target: activation mass of each expert per token; ridge regression in
                // dual form (n×n system) so it is cheap even for large n_embd
                let mut s = vec![0.0f64; n * ne];
                for t in 0..n {
                    for (cl, mem) in members.iter().enumerate() {
                        s[t * ne + cl] = mem.iter().map(|&i| (h[t * ff + i].abs() * dnorm[i]) as f64).sum();
                    }
                }
                let mut kmat = vec![0.0f64; n * n];
                for a in 0..n {
                    for b in 0..=a {
                        let v = dot(&xs[a * e..(a + 1) * e], &xs[b * e..(b + 1) * e]) as f64;
                        kmat[a * n + b] = v;
                        kmat[b * n + a] = v;
                    }
                }
                let tr: f64 = (0..n).map(|i| kmat[i * n + i]).sum::<f64>() / n as f64;
                for i in 0..n {
                    kmat[i * n + i] += 1e-2 * tr.max(1e-9);
                }
                let alpha = solve_spd(kmat, n, &s, ne);
                for cl in 0..ne {
                    for t in 0..n {
                        let a = alpha[t * ne + cl] as f32;
                        if a != 0.0 {
                            router[cl * e..(cl + 1) * e].iter_mut().zip(&xs[t * e..(t + 1) * e]).for_each(|(r, xv)| *r += a * xv);
                        }
                    }
                }
                // how much activation mass the fitted router keeps on the calibration set
                let (mut kept, mut total) = (0.0f64, 0.0f64);
                for t in 0..n {
                    let xt = &xs[t * e..(t + 1) * e];
                    let mut sc: Vec<(f32, usize)> = (0..ne).map(|cl| (dot(&router[cl * e..(cl + 1) * e], xt), cl)).collect();
                    sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    kept += sc[..k].iter().map(|&(_, cl)| s[t * ne + cl]).sum::<f64>();
                    total += s[t * ne..(t + 1) * ne].iter().sum::<f64>();
                }
                routed_mass = Some(if total > 0.0 { (kept / total) as f32 } else { 1.0 });
            }
            _ => {
                // data-free: an expert fires when x aligns with its gate rows
                for (cl, mem) in members.iter().enumerate() {
                    let r = &mut router[cl * e..(cl + 1) * e];
                    for &i in mem {
                        r.iter_mut().zip(&g[i * e..(i + 1) * e]).for_each(|(a, b)| *a += b * dnorm[i]);
                    }
                    normalize(r);
                }
            }
        }

        // expert tensors (rows regrouped by expert; down's columns likewise)
        let wty = match d.gate.ty {
            t @ (GgmlType::Q4_0 | GgmlType::Q8_0) if m % 32 == 0 => t,
            GgmlType::F32 => GgmlType::F32,
            _ if m % 32 == 0 => GgmlType::Q8_0,
            _ => GgmlType::F16,
        };
        let (mut ge, mut ue, mut de) = (Vec::with_capacity(ff * e), Vec::with_capacity(ff * e), Vec::with_capacity(ff * e));
        for mem in &members {
            for &i in mem {
                ge.extend_from_slice(&g[i * e..(i + 1) * e]);
                ue.extend_from_slice(&u[i * e..(i + 1) * e]);
            }
            for r in 0..e {
                de.extend(mem.iter().map(|&i| dn[r * ff + i]));
            }
        }
        let p = |s: &str| format!("blk.{il}.{s}");
        w.tensor(&p("ffn_gate_inp.weight"), &[e as u64, ne as u64], GgmlType::F32, quantize::f32(&router));
        let gty = if e % 32 == 0 { wty } else { GgmlType::F16 };
        w.tensor(&p("ffn_gate_exps.weight"), &[e as u64, m as u64, ne as u64], gty, encode(gty, &ge));
        w.tensor(&p("ffn_up_exps.weight"), &[e as u64, m as u64, ne as u64], gty, encode(gty, &ue));
        w.tensor(&p("ffn_down_exps.weight"), &[m as u64, e as u64, ne as u64], wty, encode(wty, &de));
        reports.push(LayerReport { layer: il, expert_size: m, routed_mass });
    }
    Ok((w.to_bytes(), reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::SyntheticSpec;

    fn logits(g: Arc<GgufFile>, toks: &[u32]) -> Vec<f32> {
        let (model, _) = load_model_from(g).unwrap();
        let pool = Arc::new(ThreadPool::new(PoolConfig { n_threads: 2, cores: vec![], spin: std::time::Duration::from_micros(20) }));
        let mut be = CpuBackend::new(model.clone(), pool, 64, 16);
        let mut l = vec![0.0; model.cfg.n_vocab];
        be.forward(toks, 0, false, &mut l).unwrap();
        l
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let n = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / n
    }

    #[test]
    fn quantize_f32_to_4_bit() {
        let spec = SyntheticSpec::tiny("llama", GgmlType::F32);
        let src = GgufFile::from_bytes(spec.build()).unwrap();
        let q = quantize_model(&src, GgmlType::Q4_0).unwrap();
        assert!(q.len() * 4 < spec.build().len(), "{} vs {}", q.len(), spec.build().len());
        let dst = Arc::new(GgufFile::from_bytes(q).unwrap());
        assert_eq!(dst.tensor("blk.0.attn_q.weight").unwrap().ty, GgmlType::Q4_0);
        assert_eq!(dst.tensor("output.weight").map(|t| t.ty).unwrap_or(GgmlType::Q8_0), GgmlType::Q8_0);
        let toks = [1u32, 7, 3, 9, 2];
        let src = Arc::new(src);
        let e4 = rel(&logits(dst, &toks), &logits(src.clone(), &toks));
        let q8 = Arc::new(GgufFile::from_bytes(quantize_model(&src, GgmlType::Q8_0).unwrap()).unwrap());
        let e8 = rel(&logits(q8, &toks), &logits(src, &toks));
        eprintln!("logit error vs f32: Q4_0 {e4}, Q8_0 {e8}");
        // random weights are the worst case for 4-bit (no structure); real models lose far less
        assert!(e4 < 0.35 && e8 < 0.03 && e8 < e4 / 4.0, "{e4} {e8}");
    }

    #[test]
    fn moefy_is_exact_with_all_experts_and_routes_with_fewer() {
        let mut spec = SyntheticSpec::tiny("llama", GgmlType::F32);
        spec.ctx = 256;
        let src = Arc::new(GgufFile::from_bytes(spec.build()).unwrap());
        let toks = [1u32, 7, 3, 9, 2, 5];
        let base = logits(src.clone(), &toks);
        let calib: Vec<u32> = (0..96u32).map(|i| (i * 37 + 11) % spec.n_vocab as u32).collect();
        // all experts used: identical to the dense model
        let (img, rep) =
            moefy(src.clone(), &MoefyOptions { n_expert: 4, n_used: 4, calib: Some(calib.clone()), threads: 2, kmeans_iters: 4 }).unwrap();
        let g = Arc::new(GgufFile::from_bytes(img).unwrap());
        assert!(crate::model::is_moe(&load_model_from(g.clone()).unwrap().0));
        let e = rel(&logits(g, &toks), &base);
        assert!(e < 1e-4, "k = E must reproduce the dense model: {e}");
        assert!(rep.iter().all(|r| (r.routed_mass.unwrap() - 1.0).abs() < 1e-5));
        // half of the experts: runs, routes, and keeps most of the activation mass
        for calib in [Some(calib), None] {
            let has_calib = calib.is_some();
            let (img, rep) = moefy(src.clone(), &MoefyOptions { n_expert: 4, n_used: 2, calib, threads: 2, kmeans_iters: 4 }).unwrap();
            let e = rel(&logits(Arc::new(GgufFile::from_bytes(img).unwrap()), &toks), &base);
            eprintln!(
                "k=2/4 calibrated={has_calib}: rel logit error {e}, routed mass {:?}",
                rep.iter().map(|r| r.routed_mass).collect::<Vec<_>>()
            );
            assert!(e.is_finite() && e < 1.0);
            if has_calib {
                assert!(rep.iter().all(|r| r.routed_mass.unwrap() > 0.5));
            }
        }
    }
}
