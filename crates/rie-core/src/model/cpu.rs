//! CPU backend: batched transformer forward pass with an f16 KV cache.

use super::{Activation, Backend, Model};
use crate::error::{bail, Result};
use crate::kernels::matmul::{formats_for, matmul, matmul_multi, MatmulJob, QuantActs, SyncPtr};
use crate::kernels::{axpy_f16, dot_f16_f32, f32_to_f16_slice};
use crate::ops::{gelu_tanh, rmsnorm, rmsnorm_inplace, silu, Rope};
use crate::threadpool::ThreadPool;
use std::sync::Arc;

/// Per-layer KV storage, layout `[kv_head][slot][head_dim]` as f16 bits.
/// Sliding-window layers use a ring buffer of `window + max_batch` slots.
struct LayerCache {
    k: Vec<u16>,
    v: Vec<u16>,
    slots: usize,
}

pub struct CpuBackend {
    model: Arc<Model>,
    pool: Arc<ThreadPool>,
    n_ctx: usize,
    max_batch: usize,
    cache: Vec<LayerCache>,
    rope_global: Rope,
    rope_swa: Option<Rope>,
    // scratch
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    cos_swa: Vec<f32>,
    sin_swa: Vec<f32>,
    acts: QuantActs,
}

impl CpuBackend {
    pub fn new(model: Arc<Model>, pool: Arc<ThreadPool>, n_ctx: usize, max_batch: usize) -> Self {
        let c = &model.cfg;
        let max_batch = max_batch.clamp(1, n_ctx.max(1));
        let cache = (0..c.n_layer)
            .map(|il| {
                let slots = if c.is_swa(il) { (c.swa_window + max_batch).min(n_ctx) } else { n_ctx };
                let n = c.n_head_kv * slots * c.head_dim;
                LayerCache { k: vec![0; n], v: vec![0; n], slots }
            })
            .collect();
        let rope_global = Rope::new(c.rope_dim, c.rope_neox, c.rope_base, c.rope_freq_scale, model.rope_factors.as_deref());
        let rope_swa = c.rope_base_swa.map(|b| Rope::new(c.rope_dim, c.rope_neox, b, 1.0, None));
        let half = c.rope_dim / 2;
        let nt = pool.n_threads();
        CpuBackend {
            x: vec![0.0; max_batch * c.n_embd],
            xn: vec![0.0; max_batch * c.n_embd],
            q: vec![0.0; max_batch * c.q_dim()],
            k: vec![0.0; max_batch * c.kv_dim()],
            v: vec![0.0; max_batch * c.kv_dim()],
            att: vec![0.0; max_batch * c.q_dim()],
            o: vec![0.0; max_batch * c.n_embd],
            gate: vec![0.0; max_batch * c.n_ff],
            up: vec![0.0; max_batch * c.n_ff],
            scores: vec![0.0; nt * n_ctx],
            cos: vec![0.0; max_batch * half],
            sin: vec![0.0; max_batch * half],
            cos_swa: vec![0.0; max_batch * half],
            sin_swa: vec![0.0; max_batch * half],
            acts: QuantActs::new(),
            model,
            pool,
            n_ctx,
            max_batch,
            cache,
            rope_global,
            rope_swa,
        }
    }

    /// Bytes used by the KV cache.
    pub fn kv_bytes(&self) -> usize {
        self.cache.iter().map(|c| (c.k.len() + c.v.len()) * 2).sum()
    }

    pub fn pool(&self) -> &Arc<ThreadPool> {
        &self.pool
    }

    fn forward_impl(&mut self, tokens: &[u32], pos0: usize, all_logits: bool, logits: &mut [f32]) -> Result<()> {
        let model = self.model.clone();
        let c = &model.cfg;
        let pool = &*self.pool.clone();
        let n = tokens.len();
        if n == 0 {
            bail!(Runtime, "empty batch");
        }
        if n > self.max_batch {
            bail!(Runtime, "batch of {n} tokens exceeds max batch {}", self.max_batch);
        }
        if pos0 + n > self.n_ctx {
            bail!(Runtime, "context window full ({} tokens)", self.n_ctx);
        }
        let (e, qd, kvd, hd, ff) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim, c.n_ff);
        let half = c.rope_dim / 2;

        // Token embeddings.
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= model.token_embd.rows {
                bail!(Runtime, "token id {tok} out of range");
            }
            let row = &mut self.x[t * e..(t + 1) * e];
            model.token_embd.dequant_row(tok as usize, row);
            if c.embd_scale != 1.0 {
                row.iter_mut().for_each(|v| *v *= c.embd_scale);
            }
        }
        // RoPE tables for this batch.
        for t in 0..n {
            let p = pos0 + t;
            self.rope_global.table(p, &mut self.cos[t * half..(t + 1) * half], &mut self.sin[t * half..(t + 1) * half]);
            if let Some(r) = &self.rope_swa {
                r.table(p, &mut self.cos_swa[t * half..(t + 1) * half], &mut self.sin_swa[t * half..(t + 1) * half]);
            }
        }

        for (il, lw) in model.layers.iter().enumerate() {
            let swa = c.is_swa(il);
            // ---- attention block ----
            par_rmsnorm(pool, &mut self.xn, &self.x, &lw.attn_norm, n, e, c.rms_eps);
            self.acts.prepare(pool, &self.xn, n, e, &formats_for(&[&lw.wq, &lw.wk, &lw.wv]));
            matmul_multi(
                pool,
                &self.acts,
                &mut [
                    MatmulJob { w: &lw.wq, out: &mut self.q[..n * qd], ld: qd },
                    MatmulJob { w: &lw.wk, out: &mut self.k[..n * kvd], ld: kvd },
                    MatmulJob { w: &lw.wv, out: &mut self.v[..n * kvd], ld: kvd },
                ],
            );
            for t in 0..n {
                if let Some(b) = &lw.bq {
                    add_inplace(&mut self.q[t * qd..(t + 1) * qd], b);
                }
                if let Some(b) = &lw.bk {
                    add_inplace(&mut self.k[t * kvd..(t + 1) * kvd], b);
                }
                if let Some(b) = &lw.bv {
                    add_inplace(&mut self.v[t * kvd..(t + 1) * kvd], b);
                }
            }
            let (rope, cos, sin) = match (&self.rope_swa, swa) {
                (Some(r), true) => (r, &self.cos_swa, &self.sin_swa),
                _ => (&self.rope_global, &self.cos, &self.sin),
            };
            for t in 0..n {
                let (ct, st) = (&cos[t * half..(t + 1) * half], &sin[t * half..(t + 1) * half]);
                for h in 0..c.n_head {
                    let qh = &mut self.q[t * qd + h * hd..t * qd + (h + 1) * hd];
                    if let Some(w) = &lw.q_norm {
                        rmsnorm_inplace(qh, w, c.rms_eps);
                    }
                    rope.apply(qh, ct, st);
                    qh.iter_mut().for_each(|v| *v *= c.attn_scale);
                }
                for h in 0..c.n_head_kv {
                    let kh = &mut self.k[t * kvd + h * hd..t * kvd + (h + 1) * hd];
                    if let Some(w) = &lw.k_norm {
                        rmsnorm_inplace(kh, w, c.rms_eps);
                    }
                    rope.apply(kh, ct, st);
                }
            }
            // Append K/V to the cache.
            let lc = &mut self.cache[il];
            for t in 0..n {
                let slot = (pos0 + t) % lc.slots;
                for h in 0..c.n_head_kv {
                    let dst = (h * lc.slots + slot) * hd;
                    f32_to_f16_slice(&self.k[t * kvd + h * hd..t * kvd + (h + 1) * hd], &mut lc.k[dst..dst + hd]);
                    f32_to_f16_slice(&self.v[t * kvd + h * hd..t * kvd + (h + 1) * hd], &mut lc.v[dst..dst + hd]);
                }
            }
            let window = if swa { c.swa_window } else { 0 };
            attention(
                pool,
                c.n_head,
                c.n_head_kv,
                hd,
                n,
                pos0,
                window,
                c.attn_softcap,
                &self.q,
                &self.cache[il],
                &mut self.att,
                &mut self.scores,
                self.n_ctx,
            );
            self.acts.prepare(pool, &self.att, n, qd, &formats_for(&[&lw.wo]));
            matmul(pool, &self.acts, &lw.wo, &mut self.o[..n * e]);
            if let Some(w) = &lw.post_attn_norm {
                for t in 0..n {
                    rmsnorm_inplace(&mut self.o[t * e..(t + 1) * e], w, c.rms_eps);
                }
            }
            residual_add(&mut self.x[..n * e], &self.o[..n * e], c.residual_scale);

            // ---- feed-forward block ----
            par_rmsnorm(pool, &mut self.xn, &self.x, &lw.ffn_norm, n, e, c.rms_eps);
            self.acts.prepare(pool, &self.xn, n, e, &formats_for(&[&lw.w_gate, &lw.w_up]));
            matmul_multi(
                pool,
                &self.acts,
                &mut [
                    MatmulJob { w: &lw.w_gate, out: &mut self.gate[..n * ff], ld: ff },
                    MatmulJob { w: &lw.w_up, out: &mut self.up[..n * ff], ld: ff },
                ],
            );
            {
                let gp = SyncPtr(self.gate.as_mut_ptr());
                let up = &self.up;
                let act = c.act;
                let chunk = 1024;
                pool.for_each((n * ff).div_ceil(chunk), 4, |ci, _| {
                    let s = ci * chunk;
                    let end = (s + chunk).min(n * ff);
                    let g = unsafe { std::slice::from_raw_parts_mut(gp.get().add(s), end - s) };
                    match act {
                        Activation::Silu => g.iter_mut().zip(&up[s..end]).for_each(|(a, b)| *a = silu(*a) * b),
                        Activation::GeluTanh => g.iter_mut().zip(&up[s..end]).for_each(|(a, b)| *a = gelu_tanh(*a) * b),
                    }
                });
            }
            self.acts.prepare(pool, &self.gate, n, ff, &formats_for(&[&lw.w_down]));
            matmul(pool, &self.acts, &lw.w_down, &mut self.o[..n * e]);
            if let Some(w) = &lw.post_ffn_norm {
                for t in 0..n {
                    rmsnorm_inplace(&mut self.o[t * e..(t + 1) * e], w, c.rms_eps);
                }
            }
            residual_add(&mut self.x[..n * e], &self.o[..n * e], c.residual_scale);
        }

        // ---- output head ----
        let first = if all_logits { 0 } else { n - 1 };
        let m = n - first;
        for t in 0..m {
            rmsnorm(&mut self.xn[t * e..(t + 1) * e], &self.x[(first + t) * e..(first + t + 1) * e], &model.output_norm, c.rms_eps);
        }
        let nv = c.n_vocab;
        if logits.len() < m * nv {
            bail!(Runtime, "logits buffer too small");
        }
        self.acts.prepare(pool, &self.xn, m, e, &formats_for(&[&model.output]));
        let out_w = if model.output.rows > nv { model.output.slice_rows(0, nv) } else { model.output.clone() };
        matmul(pool, &self.acts, &out_w, &mut logits[..m * nv]);
        if c.logit_scale != 1.0 {
            logits[..m * nv].iter_mut().for_each(|v| *v /= c.logit_scale);
        }
        if let Some(cap) = c.final_softcap {
            logits[..m * nv].iter_mut().for_each(|v| *v = cap * (*v / cap).tanh());
        }
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("CPU ({} threads, {})", self.pool.n_threads(), crate::kernels::CpuFeatures::get().kernel_name())
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn forward(&mut self, tokens: &[u32], pos: usize, all_logits: bool, logits: &mut [f32]) -> Result<()> {
        self.forward_impl(tokens, pos, all_logits, logits)
    }
}

fn add_inplace(x: &mut [f32], b: &[f32]) {
    x.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

fn residual_add(x: &mut [f32], y: &[f32], scale: f32) {
    if scale == 1.0 {
        x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
    } else {
        x.iter_mut().zip(y).for_each(|(a, b)| *a += b * scale);
    }
}

fn par_rmsnorm(pool: &ThreadPool, out: &mut [f32], x: &[f32], w: &[f32], n: usize, e: usize, eps: f32) {
    if n == 1 {
        rmsnorm(&mut out[..e], &x[..e], w, eps);
        return;
    }
    let op = SyncPtr(out.as_mut_ptr());
    pool.for_each(n, 1, |t, _| {
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(t * e), e) };
        rmsnorm(o, &x[t * e..(t + 1) * e], w, eps);
    });
}

/// Causal (optionally sliding-window) multi-head attention with grouped-query KV heads.
#[allow(clippy::too_many_arguments)]
fn attention(
    pool: &ThreadPool,
    n_head: usize,
    n_head_kv: usize,
    hd: usize,
    n_tok: usize,
    pos0: usize,
    window: usize,
    softcap: Option<f32>,
    q: &[f32],
    cache: &LayerCache,
    out: &mut [f32],
    scores: &mut [f32],
    n_ctx: usize,
) {
    let qd = n_head * hd;
    let group = n_head / n_head_kv;
    let out_p = SyncPtr(out.as_mut_ptr());
    let sc_p = SyncPtr(scores.as_mut_ptr());
    let slots = cache.slots;
    pool.for_each(n_tok * n_head, 1, |task, tid| {
        let t = task / n_head;
        let h = task % n_head;
        let g = h / group;
        let p = pos0 + t;
        let start = if window > 0 { (p + 1).saturating_sub(window) } else { 0 };
        let len = p + 1 - start;
        let qh = &q[t * qd + h * hd..t * qd + (h + 1) * hd];
        let sc = unsafe { std::slice::from_raw_parts_mut(sc_p.get().add(tid * n_ctx), len) };
        let kbase = &cache.k[g * slots * hd..(g + 1) * slots * hd];
        let vbase = &cache.v[g * slots * hd..(g + 1) * slots * hd];
        let mut max = f32::NEG_INFINITY;
        for (i, s) in sc.iter_mut().enumerate() {
            let slot = (start + i) % slots;
            let mut v = dot_f16_f32(&kbase[slot * hd..(slot + 1) * hd], qh);
            if let Some(cap) = softcap {
                v = cap * (v / cap).tanh();
            }
            *s = v;
            max = max.max(v);
        }
        let mut sum = 0.0f32;
        for s in sc.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        let o = unsafe { std::slice::from_raw_parts_mut(out_p.get().add(t * qd + h * hd), hd) };
        o.fill(0.0);
        for (i, &s) in sc.iter().enumerate() {
            let slot = (start + i) % slots;
            axpy_f16(o, s * inv, &vbase[slot * hd..(slot + 1) * hd]);
        }
    });
}
