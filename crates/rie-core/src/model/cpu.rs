//! CPU backend: batched transformer forward pass with an f16, Q8_0 or TurboQuant-4 KV cache.

use super::{Activation, Backend, DenseFfn, Ffn, Model, MoeFfn};
use crate::error::{bail, Result};
use crate::kernels::matmul::{formats_for, matmul, matmul_multi, MatmulJob, QuantActs, SyncPtr};
use crate::kernels::{axpy_f16, dot_f16_f32, f32_to_f16_slice, fp16_to_f32, vec_dot, ActRow};
use crate::ops::{gelu_tanh, rmsnorm, rmsnorm_inplace, silu, Rope};
use crate::quant::{quantize_row_q8_0, BlockQ8_0, GgmlType, QK};
use crate::threadpool::ThreadPool;
use std::sync::Arc;

/// Storage format of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvType {
    /// 16-bit floats (2 bytes / value).
    #[default]
    F16,
    /// ggml Q8_0 blocks (1.06 bytes / value): ~47% less memory, slightly lower precision.
    /// Attention scores then use the int8 dot-product kernels.
    Q8_0,
    /// TurboQuant 4-bit (random rotation + Lloyd–Max codes, one f16 scale per rotation
    /// block): ≈4.1 bits / value for head size 128 — 3.9× smaller than f16, 2× smaller
    /// than Q8_0. Attention runs in the rotated domain without dequantizing the cache.
    Turbo4,
}

impl KvType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Some(KvType::F16),
            "q8" | "q8_0" => Some(KvType::Q8_0),
            "turbo4" | "tq4" | "turbo" => Some(KvType::Turbo4),
            _ => None,
        }
    }
}

/// Per-layer KV storage, layout `[kv_head][slot][head_dim]`, as f16 bits (`k`/`v`) or as
/// Q8_0 blocks (`kq`/`vq`, `head_dim / 32 * 34` bytes per row).
/// Sliding-window layers use a ring buffer of `window + max_batch` slots.
struct LayerCache {
    k: Vec<u16>,
    v: Vec<u16>,
    kq: Vec<u8>,
    vq: Vec<u8>,
    slots: usize,
    ty: KvType,
    /// rotation of the TurboQuant cache
    rot: Option<crate::turbo::Rotation>,
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
    moe: MoeScratch,
    /// rows of `xn` holding the final hidden states of the last forward
    hidden_rows: usize,
    /// layer streaming (see [`CpuBackend::set_streaming`])
    stream: bool,
    /// per-layer feed-forward inputs, recorded for calibration (see [`CpuBackend::capture_ffn_inputs`])
    ffn_capture: Option<Vec<Vec<f32>>>,
}

/// Scratch buffers for mixture-of-experts layers.
struct MoeScratch {
    logits: Vec<f32>,
    xg: Vec<f32>,
    yg: Vec<f32>,
    ys: Vec<f32>,
}

impl CpuBackend {
    pub fn new(model: Arc<Model>, pool: Arc<ThreadPool>, n_ctx: usize, max_batch: usize) -> Self {
        Self::with_kv_type(model, pool, n_ctx, max_batch, KvType::F16)
    }

    /// Like [`CpuBackend::new`] with a chosen KV-cache format. Q8_0 requires a head
    /// dimension divisible by 32 and falls back to F16 otherwise.
    pub fn with_kv_type(model: Arc<Model>, pool: Arc<ThreadPool>, n_ctx: usize, max_batch: usize, kv: KvType) -> Self {
        let c = &model.cfg;
        let max_batch = max_batch.clamp(1, n_ctx.max(1));
        let kv = match kv {
            KvType::Q8_0 if c.head_dim % QK != 0 => KvType::F16,
            // the rotation needs power-of-two blocks of at least 16 values
            KvType::Turbo4 if c.head_dim % QK != 0 || c.head_dim > 512 => KvType::F16,
            k => k,
        };
        let cache = (0..c.n_layer)
            .map(|il| {
                let slots = if c.is_swa(il) { (c.swa_window + max_batch).min(n_ctx) } else { n_ctx };
                let rows = c.n_head_kv * slots;
                match kv {
                    KvType::F16 => {
                        let n = rows * c.head_dim;
                        LayerCache { k: vec![0; n], v: vec![0; n], kq: Vec::new(), vq: Vec::new(), slots, ty: kv, rot: None }
                    }
                    KvType::Q8_0 => {
                        let n = rows * GgmlType::Q8_0.row_bytes(c.head_dim);
                        LayerCache { k: Vec::new(), v: Vec::new(), kq: vec![0; n], vq: vec![0; n], slots, ty: kv, rot: None }
                    }
                    KvType::Turbo4 => {
                        let n = rows * crate::turbo::row_bytes_4(c.head_dim);
                        let rot = Some(crate::turbo::Rotation::new(c.head_dim, crate::turbo::KV_SEED));
                        LayerCache { k: Vec::new(), v: Vec::new(), kq: vec![0; n], vq: vec![0; n], slots, ty: kv, rot }
                    }
                }
            })
            .collect();
        let rope_global = Rope::new(c.rope_dim, c.rope_neox, c.rope_base, c.rope_freq_scale, model.rope_factors.as_deref());
        let rope_swa = c.rope_base_swa.map(|b| Rope::new(c.rope_dim, c.rope_neox, b, 1.0, None));
        let half = c.rope_dim / 2;
        let nt = pool.n_threads();
        // largest feed-forward hidden size across dense, expert and shared-expert blocks
        let ff_max = model
            .layers
            .iter()
            .flat_map(|l| match &l.ffn {
                Ffn::Dense(d) => vec![d.gate.rows],
                Ffn::Moe(m) => m.experts.iter().chain(m.shared.iter()).map(|d| d.gate.rows).collect(),
            })
            .max()
            .unwrap_or(c.n_ff);
        CpuBackend {
            x: vec![0.0; max_batch * c.n_embd],
            xn: vec![0.0; max_batch * c.n_embd],
            q: vec![0.0; max_batch * c.q_dim()],
            k: vec![0.0; max_batch * c.kv_dim()],
            v: vec![0.0; max_batch * c.kv_dim()],
            att: vec![0.0; max_batch * c.q_dim()],
            o: vec![0.0; max_batch * c.n_embd],
            gate: vec![0.0; max_batch * ff_max],
            up: vec![0.0; max_batch * ff_max],
            moe: MoeScratch {
                logits: vec![0.0; max_batch * c.n_expert.max(1)],
                xg: vec![0.0; max_batch * c.n_embd],
                yg: vec![0.0; max_batch * c.n_embd],
                ys: vec![0.0; max_batch * c.n_embd],
            },
            scores: vec![0.0; nt * n_ctx],
            cos: vec![0.0; max_batch * half],
            sin: vec![0.0; max_batch * half],
            cos_swa: vec![0.0; max_batch * half],
            sin_swa: vec![0.0; max_batch * half],
            acts: QuantActs::new(),
            hidden_rows: 0,
            stream: false,
            ffn_capture: None,
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
        self.cache.iter().map(|c| (c.k.len() + c.v.len()) * 2 + c.kq.len() + c.vq.len()).sum()
    }

    /// Layer streaming for models larger than RAM: while layer `l` computes, layer `l+1`
    /// is prefetched from flash and layer `l-1`'s pages are released. Peak weight memory
    /// becomes about two layers plus embeddings/output head, at the cost of re-reading
    /// weights every token (flash bandwidth bound).
    pub fn set_streaming(&mut self, on: bool) {
        self.stream = on;
    }

    /// Start recording the normalized input of every feed-forward block (calibration for
    /// `rie moefy`); [`CpuBackend::take_ffn_inputs`] returns `[layer][token * n_embd]`.
    pub fn capture_ffn_inputs(&mut self) {
        self.ffn_capture = Some(vec![Vec::new(); self.model.layers.len()]);
    }

    pub fn take_ffn_inputs(&mut self) -> Vec<Vec<f32>> {
        self.ffn_capture.take().unwrap_or_default()
    }

    pub fn kv_type(&self) -> KvType {
        self.cache.first().map(|c| c.ty).unwrap_or_default()
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
        let (e, qd, kvd, hd) = (c.n_embd, c.q_dim(), c.kv_dim(), c.head_dim);
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
            if self.stream {
                model.advise_layer(il + 1, false);
                if il > 0 {
                    model.advise_layer(il - 1, true);
                }
            }
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
            let mut scratch = vec![0.0f32; if lc.ty == KvType::Turbo4 { hd } else { 0 }];
            for t in 0..n {
                let slot = (pos0 + t) % lc.slots;
                for h in 0..c.n_head_kv {
                    let ks = &self.k[t * kvd + h * hd..t * kvd + (h + 1) * hd];
                    let vs = &self.v[t * kvd + h * hd..t * kvd + (h + 1) * hd];
                    let row = h * lc.slots + slot;
                    match lc.ty {
                        KvType::F16 => {
                            let dst = row * hd;
                            f32_to_f16_slice(ks, &mut lc.k[dst..dst + hd]);
                            f32_to_f16_slice(vs, &mut lc.v[dst..dst + hd]);
                        }
                        KvType::Q8_0 => {
                            let rb = GgmlType::Q8_0.row_bytes(hd);
                            store_q8_0(ks, &mut lc.kq[row * rb..(row + 1) * rb]);
                            store_q8_0(vs, &mut lc.vq[row * rb..(row + 1) * rb]);
                        }
                        KvType::Turbo4 => {
                            let rb = crate::turbo::row_bytes_4(hd);
                            let rot = lc.rot.as_ref().expect("turbo rotation");
                            crate::turbo::quantize_4(rot, ks, &mut scratch, &mut lc.kq[row * rb..(row + 1) * rb]);
                            crate::turbo::quantize_4(rot, vs, &mut scratch, &mut lc.vq[row * rb..(row + 1) * rb]);
                        }
                    }
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
            if let Some(cap) = &mut self.ffn_capture {
                cap[il].extend_from_slice(&self.xn[..n * e]);
            }
            match &lw.ffn {
                Ffn::Dense(d) => {
                    dense_ffn(pool, &mut self.acts, d, &self.xn[..n * e], n, c.act, &mut self.gate, &mut self.up, &mut self.o[..n * e]);
                }
                Ffn::Moe(m) => moe_ffn(
                    pool,
                    &mut self.acts,
                    m,
                    &self.xn[..n * e],
                    n,
                    c.act,
                    &mut self.moe,
                    &mut self.gate,
                    &mut self.up,
                    &mut self.o[..n * e],
                ),
            }
            if let Some(w) = &lw.post_ffn_norm {
                for t in 0..n {
                    rmsnorm_inplace(&mut self.o[t * e..(t + 1) * e], w, c.rms_eps);
                }
            }
            residual_add(&mut self.x[..n * e], &self.o[..n * e], c.residual_scale);
        }

        if self.stream {
            model.advise_layer(model.layers.len() - 1, true);
            model.advise_layer(0, false);
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
        self.hidden_rows = m;
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
        self.hidden_rows = 0;
        self.forward_impl(tokens, pos, all_logits, logits)
    }
    fn hidden(&self) -> Option<&[f32]> {
        (self.hidden_rows > 0).then(|| &self.xn[..self.hidden_rows * self.model.cfg.n_embd])
    }
}

/// `out = down(act(gate·x) * up·x)` for `n` rows of `x`.
#[allow(clippy::too_many_arguments)]
fn dense_ffn(
    pool: &ThreadPool,
    acts: &mut QuantActs,
    d: &DenseFfn,
    x: &[f32],
    n: usize,
    act: Activation,
    gate: &mut [f32],
    up: &mut [f32],
    out: &mut [f32],
) {
    let (e, ff) = (d.gate.cols, d.gate.rows);
    acts.prepare(pool, x, n, e, &formats_for(&[&d.gate, &d.up]));
    matmul_multi(
        pool,
        acts,
        &mut [MatmulJob { w: &d.gate, out: &mut gate[..n * ff], ld: ff }, MatmulJob { w: &d.up, out: &mut up[..n * ff], ld: ff }],
    );
    {
        let gp = SyncPtr(gate.as_mut_ptr());
        let up = &*up;
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
    acts.prepare(pool, gate, n, ff, &formats_for(&[&d.down]));
    matmul(pool, acts, &d.down, &mut out[..n * e]);
}

/// Mixture of experts: softmax router → top-k → (optionally renormalized) weighted sum of
/// expert FFNs, plus an optional gated shared expert. Tokens are grouped per expert so
/// each selected expert's weights are streamed once per batch.
#[allow(clippy::too_many_arguments)]
fn moe_ffn(
    pool: &ThreadPool,
    acts: &mut QuantActs,
    m: &MoeFfn,
    x: &[f32],
    n: usize,
    act: Activation,
    sc: &mut MoeScratch,
    gate: &mut [f32],
    up: &mut [f32],
    out: &mut [f32],
) {
    let e = m.router.cols;
    let ne = m.router.rows;
    acts.prepare(pool, x, n, e, &formats_for(&[&m.router]));
    matmul(pool, acts, &m.router, &mut sc.logits[..n * ne]);
    // routing
    let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
    let mut idx: Vec<usize> = (0..ne).collect();
    for t in 0..n {
        let l = &mut sc.logits[t * ne..(t + 1) * ne];
        crate::ops::softmax(l);
        idx.sort_unstable_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let top = &idx[..m.n_used];
        let norm = if m.norm_topk { top.iter().map(|&i| l[i]).sum::<f32>() } else { 1.0 };
        for &i in top {
            assign[i].push((t, if m.unit_weights { 1.0 } else { l[i] / norm }));
        }
    }
    out[..n * e].fill(0.0);
    for (ex, list) in assign.iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let k = list.len();
        for (j, &(t, _)) in list.iter().enumerate() {
            sc.xg[j * e..(j + 1) * e].copy_from_slice(&x[t * e..(t + 1) * e]);
        }
        dense_ffn(pool, acts, &m.experts[ex], &sc.xg[..k * e], k, act, gate, up, &mut sc.yg[..k * e]);
        for (j, &(t, w)) in list.iter().enumerate() {
            out[t * e..(t + 1) * e].iter_mut().zip(&sc.yg[j * e..(j + 1) * e]).for_each(|(o, y)| *o += w * y);
        }
    }
    if let Some(sh) = &m.shared {
        dense_ffn(pool, acts, sh, x, n, act, gate, up, &mut sc.ys[..n * e]);
        for t in 0..n {
            let xt = &x[t * e..(t + 1) * e];
            let scale = m.shared_gate.as_ref().map(|g| 1.0 / (1.0 + (-crate::kernels::dot_f32(g, xt)).exp())).unwrap_or(1.0);
            out[t * e..(t + 1) * e].iter_mut().zip(&sc.ys[t * e..(t + 1) * e]).for_each(|(o, y)| *o += scale * y);
        }
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
        let rb = match cache.ty {
            KvType::Turbo4 => crate::turbo::row_bytes_4(hd),
            _ => GgmlType::Q8_0.row_bytes(hd),
        };
        // TurboQuant: work in the rotated domain (rotation is orthogonal, so q·k = Rq·Rk)
        let mut qrot = [0.0f32; 512];
        if let Some(rot) = &cache.rot {
            qrot[..hd].copy_from_slice(qh);
            rot.apply(&mut qrot[..hd]);
        }
        // For the Q8_0 cache the query is quantized once so scores use int8 dot kernels.
        let mut qq = [BlockQ8_0::default(); 16];
        let nqb = hd / QK;
        match cache.ty {
            KvType::Q8_0 => quantize_row_q8_0(qh, &mut qq[..nqb]),
            KvType::Turbo4 => quantize_row_q8_0(&qrot[..hd], &mut qq[..nqb]),
            KvType::F16 => {}
        }
        let mut max = f32::NEG_INFINITY;
        for (i, s) in sc.iter_mut().enumerate() {
            let slot = (start + i) % slots;
            let row = g * slots + slot;
            let mut v = match cache.ty {
                KvType::F16 => dot_f16_f32(&cache.k[row * hd..(row + 1) * hd], qh),
                KvType::Q8_0 => vec_dot(GgmlType::Q8_0, &cache.kq[row * rb..(row + 1) * rb], ActRow::Q8_0(&qq[..nqb]), hd),
                KvType::Turbo4 => crate::turbo::dot_rotated_4(&qq[..nqb], &cache.kq[row * rb..(row + 1) * rb]),
            };
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
            let row = g * slots + (start + i) % slots;
            match cache.ty {
                KvType::F16 => axpy_f16(o, s * inv, &cache.v[row * hd..(row + 1) * hd]),
                KvType::Q8_0 => axpy_q8_0(o, s * inv, &cache.vq[row * rb..(row + 1) * rb]),
                KvType::Turbo4 => crate::turbo::axpy_rotated_4(o, s * inv, &cache.vq[row * rb..(row + 1) * rb]),
            }
        }
        if let Some(rot) = &cache.rot {
            rot.apply_inverse(o);
        }
    });
}

/// Quantize one head vector into Q8_0 blocks (ggml layout).
fn store_q8_0(x: &[f32], dst: &mut [u8]) {
    for (xb, ob) in x.chunks_exact(QK).zip(dst.chunks_exact_mut(34)) {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        ob[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for (q, &v) in ob[2..].iter_mut().zip(xb) {
            *q = ((v * id).round() as i32).clamp(-127, 127) as i8 as u8;
        }
    }
}

/// `y += a * dequant(row)` for a Q8_0 row.
#[inline]
fn axpy_q8_0(y: &mut [f32], a: f32, row: &[u8]) {
    for (yb, b) in y.chunks_exact_mut(QK).zip(row.chunks_exact(34)) {
        let s = a * fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for (o, &q) in yb.iter_mut().zip(&b[2..]) {
            *o += s * (q as i8) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threadpool::PoolConfig;

    /// Rotated-domain attention over a TurboQuant cache must equal ordinary attention over
    /// an f16 cache holding the reconstructed (dequantized) keys and values.
    #[test]
    fn turbo_attention_equals_attention_over_reconstruction() {
        let (n_head, n_kv, hd, slots) = (4, 2, 64, 24);
        let rot = crate::turbo::Rotation::new(hd, crate::turbo::KV_SEED);
        let rb = crate::turbo::row_bytes_4(hd);
        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let mut tq = LayerCache {
            k: vec![],
            v: vec![],
            kq: vec![0; n_kv * slots * rb],
            vq: vec![0; n_kv * slots * rb],
            slots,
            ty: KvType::Turbo4,
            rot: Some(rot.clone()),
        };
        let mut f = LayerCache {
            k: vec![0; n_kv * slots * hd],
            v: vec![0; n_kv * slots * hd],
            kq: vec![],
            vq: vec![],
            slots,
            ty: KvType::F16,
            rot: None,
        };
        let (mut scratch, mut rec) = (vec![0.0; hd], vec![0.0; hd]);
        for row in 0..n_kv * slots {
            for (dst_q, dst_f) in [(&mut tq.kq, &mut f.k), (&mut tq.vq, &mut f.v)] {
                let x: Vec<f32> = (0..hd).map(|_| 3.0 * rnd()).collect();
                let r = &mut dst_q[row * rb..(row + 1) * rb];
                crate::turbo::quantize_4(&rot, &x, &mut scratch, r);
                crate::turbo::dequantize_4(&rot, r, &mut rec);
                f32_to_f16_slice(&rec, &mut dst_f[row * hd..(row + 1) * hd]);
            }
        }
        let pool = ThreadPool::new(PoolConfig { n_threads: 2, cores: vec![], spin: std::time::Duration::from_micros(10) });
        let n_tok = 3;
        // moderate scores: the query is int8-quantized for the dot products (as for Q8_0 KV)
        let q: Vec<f32> = (0..n_tok * n_head * hd).map(|_| 0.2 * rnd()).collect();
        let (mut oa, mut ob) = (vec![0.0; q.len()], vec![0.0; q.len()]);
        let mut sc = vec![0.0; 2 * slots];
        attention(&pool, n_head, n_kv, hd, n_tok, 20, 0, None, &q, &tq, &mut oa, &mut sc, slots);
        attention(&pool, n_head, n_kv, hd, n_tok, 20, 0, None, &q, &f, &mut ob, &mut sc, slots);
        for (a, b) in oa.iter().zip(&ob) {
            assert!((a - b).abs() < 5e-3, "{a} vs {b}");
        }
    }
}
