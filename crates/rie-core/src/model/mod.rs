//! Model definition: hyper-parameters, weights and the backend interface.

pub mod config;
pub mod cpu;

pub use config::{Activation, Arch, ModelConfig};

/// True if any layer uses a mixture-of-experts feed-forward block.
pub fn is_moe(m: &Model) -> bool {
    m.layers.iter().any(|l| matches!(l.ffn, Ffn::Moe(_)))
}

use crate::error::{bail, Error, Result};
use crate::gguf::{GgufFile, TensorInfo};
use crate::quant::GgmlType;
use crate::tensor::Weight;
use std::sync::Arc;

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub post_attn_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn: Ffn,
    pub post_ffn_norm: Option<Vec<f32>>,
}

/// Gated feed-forward block `down(act(gate·x) * up·x)`.
pub struct DenseFfn {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

/// Mixture-of-experts feed-forward block (Mixtral, Qwen2-MoE, Qwen3-MoE).
pub struct MoeFfn {
    /// `[n_expert, n_embd]` router.
    pub router: Weight,
    pub experts: Vec<DenseFfn>,
    /// Experts evaluated per token (top-k).
    pub n_used: usize,
    /// Renormalize the top-k router probabilities to sum to one.
    pub norm_topk: bool,
    /// Always-on shared expert (Qwen2-MoE), optionally scaled by `sigmoid(shared_gate · x)`.
    pub shared: Option<DenseFfn>,
    pub shared_gate: Option<Vec<f32>>,
    /// Selected experts are summed with weight 1 (models converted from dense ones by
    /// `rie moefy`, GGUF key `rie.expert_gating = "unit"`): with all experts selected the
    /// result equals the original dense feed-forward block exactly.
    pub unit_weights: bool,
}

pub enum Ffn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

impl Ffn {
    /// Every weight matrix in the block.
    pub fn weights(&self) -> Vec<&Weight> {
        match self {
            Ffn::Dense(d) => vec![&d.gate, &d.up, &d.down],
            Ffn::Moe(m) => {
                let mut v = vec![&m.router];
                for d in m.experts.iter().chain(m.shared.iter()) {
                    v.extend([&d.gate, &d.up, &d.down]);
                }
                v
            }
        }
    }
}

impl LayerWeights {
    pub fn all_weights(&self) -> Vec<&Weight> {
        let mut v = vec![&self.wq, &self.wk, &self.wv, &self.wo];
        v.extend(self.ffn.weights());
        v
    }
}

/// Loaded model: configuration + weights (zero-copy views into the GGUF mmap).
pub struct Model {
    pub cfg: ModelConfig,
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub rope_factors: Option<Vec<f32>>,
    pub layers: Vec<LayerWeights>,
    pub gguf: Arc<GgufFile>,
}

fn weight(g: &GgufFile, t: &TensorInfo) -> Result<Weight> {
    // 3-D tensors are only valid for stacked experts (`*_exps`): [cols, rows, n_expert]
    // is stored expert after expert, i.e. as one `rows * n_expert` row-major matrix.
    let stacked_ok = t.name.contains("_exps") && t.dims.len() == 3;
    if t.dims.len() > 2 && !stacked_ok && t.dims[2..].iter().any(|&d| d != 1) {
        bail!(Unsupported, "tensor {} has shape {:?}", t.name, t.dims);
    }
    // SAFETY: the returned view is stored next to an Arc<GgufFile> that owns the mapping.
    Ok(unsafe { Weight::from_raw(t.ty, t.n_rows(), t.row_len(), g.tensor_data(t)) })
}

impl Model {
    pub fn load(gguf: Arc<GgufFile>) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(&gguf)?;
        let g = &*gguf;
        let get = |name: &str| -> Result<Weight> {
            let t = g.tensor(name).ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
            weight(g, t)
        };
        let opt = |name: &str| -> Result<Option<Weight>> { g.tensor(name).map(|t| weight(g, t)).transpose() };
        let vec = |name: &str| -> Result<Vec<f32>> { Ok(get(name)?.to_f32()) };
        let opt_vec = |name: &str| -> Result<Option<Vec<f32>>> { Ok(opt(name)?.map(|w| w.to_f32())) };

        let token_embd = get("token_embd.weight")?;
        if token_embd.cols != cfg.n_embd {
            bail!(Format, "token_embd has {} columns, expected {}", token_embd.cols, cfg.n_embd);
        }
        let output = opt("output.weight")?.unwrap_or_else(|| token_embd.clone());
        let output_norm = vec("output_norm.weight")?;
        let rope_factors = match opt_vec("rope_freqs.weight")? {
            Some(v) => Some(v),
            None => opt_vec("rope_factors_short.weight")?,
        };

        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let (wq, wk, wv) = if let Some(qkv) = opt(&p("attn_qkv.weight"))? {
                if qkv.rows != q_dim + 2 * kv_dim {
                    bail!(Format, "fused qkv has {} rows, expected {}", qkv.rows, q_dim + 2 * kv_dim);
                }
                (qkv.slice_rows(0, q_dim), qkv.slice_rows(q_dim, kv_dim), qkv.slice_rows(q_dim + kv_dim, kv_dim))
            } else {
                (get(&p("attn_q.weight"))?, get(&p("attn_k.weight"))?, get(&p("attn_v.weight"))?)
            };
            let ffn = if let Some(router) = opt(&p("ffn_gate_inp.weight"))? {
                let (ne, ff) = (cfg.n_expert, cfg.n_ff_exp);
                if ne == 0 || router.rows != ne || router.cols != cfg.n_embd {
                    bail!(Format, "blk.{il}.ffn_gate_inp: shape {}x{} (expert_count {ne})", router.rows, router.cols);
                }
                let stacked = |name: &str, rows: usize, cols: usize| -> Result<Weight> {
                    let w = get(&p(name))?;
                    if w.rows != rows * ne || w.cols != cols {
                        bail!(Format, "blk.{il}.{name}: shape {}x{}, expected {}x{cols}", w.rows, w.cols, rows * ne);
                    }
                    Ok(w)
                };
                let g_all = stacked("ffn_gate_exps.weight", ff, cfg.n_embd)?;
                let u_all = stacked("ffn_up_exps.weight", ff, cfg.n_embd)?;
                let d_all = stacked("ffn_down_exps.weight", cfg.n_embd, ff)?;
                let experts = (0..ne)
                    .map(|x| DenseFfn {
                        gate: g_all.slice_rows(x * ff, ff),
                        up: u_all.slice_rows(x * ff, ff),
                        down: d_all.slice_rows(x * cfg.n_embd, cfg.n_embd),
                    })
                    .collect();
                let shared = match opt(&p("ffn_gate_shexp.weight"))? {
                    Some(gate) => Some(DenseFfn { gate, up: get(&p("ffn_up_shexp.weight"))?, down: get(&p("ffn_down_shexp.weight"))? }),
                    None => None,
                };
                Ffn::Moe(MoeFfn {
                    router,
                    experts,
                    n_used: cfg.n_expert_used.clamp(1, ne),
                    norm_topk: cfg.expert_norm_topk,
                    shared,
                    shared_gate: opt_vec(&p("ffn_gate_inp_shexp.weight"))?,
                    unit_weights: gguf.get_str("rie.expert_gating") == Some("unit"),
                })
            } else {
                let (gate, up) = match opt(&p("ffn_gate.weight"))? {
                    Some(gw) => (gw, get(&p("ffn_up.weight"))?),
                    None => {
                        let up = get(&p("ffn_up.weight"))?;
                        if up.rows != 2 * cfg.n_ff {
                            bail!(Unsupported, "layer {il}: missing ffn_gate and ffn_up is not fused");
                        }
                        (up.slice_rows(0, cfg.n_ff), up.slice_rows(cfg.n_ff, cfg.n_ff))
                    }
                };
                let down = get(&p("ffn_down.weight"))?;
                for (name, w, rows, cols) in [
                    ("ffn_gate", &gate, cfg.n_ff, cfg.n_embd),
                    ("ffn_up", &up, cfg.n_ff, cfg.n_embd),
                    ("ffn_down", &down, cfg.n_embd, cfg.n_ff),
                ] {
                    if w.rows != rows || w.cols != cols {
                        bail!(Format, "blk.{il}.{name}: shape {}x{}, expected {rows}x{cols}", w.rows, w.cols);
                    }
                }
                Ffn::Dense(DenseFfn { gate, up, down })
            };
            let lw = LayerWeights {
                attn_norm: vec(&p("attn_norm.weight"))?,
                wo: get(&p("attn_output.weight"))?,
                bq: opt_vec(&p("attn_q.bias"))?,
                bk: opt_vec(&p("attn_k.bias"))?,
                bv: opt_vec(&p("attn_v.bias"))?,
                q_norm: opt_vec(&p("attn_q_norm.weight"))?,
                k_norm: opt_vec(&p("attn_k_norm.weight"))?,
                post_attn_norm: opt_vec(&p("post_attention_norm.weight"))?,
                ffn_norm: match opt_vec(&p("ffn_norm.weight"))? {
                    Some(v) => v,
                    None => bail!(Missing, "tensor {}", p("ffn_norm.weight")),
                },
                ffn,
                post_ffn_norm: opt_vec(&p("post_ffw_norm.weight"))?,
                wq,
                wk,
                wv,
            };
            // Shape validation catches mismatched metadata early with a clear message.
            let checks = [
                ("attn_q", &lw.wq, q_dim, cfg.n_embd),
                ("attn_k", &lw.wk, kv_dim, cfg.n_embd),
                ("attn_v", &lw.wv, kv_dim, cfg.n_embd),
                ("attn_output", &lw.wo, cfg.n_embd, q_dim),
            ];
            for (name, w, rows, cols) in checks {
                if w.rows != rows || w.cols != cols {
                    bail!(Format, "blk.{il}.{name}: shape {}x{}, expected {rows}x{cols}", w.rows, w.cols);
                }
            }
            if let Some(qn) = &lw.q_norm {
                if qn.len() != cfg.head_dim {
                    bail!(Unsupported, "q_norm over {} elements (expected per-head {})", qn.len(), cfg.head_dim);
                }
            }
            layers.push(lw);
        }
        if output.cols != cfg.n_embd || output.rows < cfg.n_vocab {
            bail!(Format, "output matrix {}x{} does not cover vocabulary {} x {}", output.rows, output.cols, cfg.n_vocab, cfg.n_embd);
        }
        Ok(Model { cfg, token_embd, output_norm, output, rope_factors, layers, gguf })
    }

    /// Layer streaming: ask the OS to start reading layer `il` from flash
    /// (`release == false`), or to drop its pages from RAM (`release == true`). Dropped
    /// pages of a read-only mapping are transparently re-read on next use, so models
    /// larger than free RAM keep running instead of being killed by the low-memory killer.
    /// No-op for models held in memory.
    pub fn advise_layer(&self, il: usize, release: bool) {
        #[cfg(unix)]
        if self.gguf.is_mapped() {
            if let Some(l) = self.layers.get(il) {
                let adv = if release { libc::MADV_DONTNEED } else { libc::MADV_WILLNEED };
                l.all_weights().iter().for_each(|w| w.advise(adv));
            }
        }
        #[cfg(not(unix))]
        let _ = (il, release);
    }

    /// Total bytes of weights (as stored).
    pub fn weight_bytes(&self) -> usize {
        let mut n = self.token_embd.size_bytes();
        if !std::ptr::eq(self.output.data().as_ptr(), self.token_embd.data().as_ptr()) {
            n += self.output.size_bytes();
        }
        for l in &self.layers {
            for w in l.all_weights() {
                n += w.size_bytes();
            }
        }
        n
    }

    pub fn n_params(&self) -> usize {
        let c = &self.cfg;
        let ffn = if c.n_expert > 0 { c.n_expert * 3 * c.n_embd * c.n_ff_exp + c.n_expert * c.n_embd } else { 3 * c.n_embd * c.n_ff };
        let per_layer = c.n_embd * (c.q_dim() + 2 * c.kv_dim()) + c.q_dim() * c.n_embd + ffn;
        c.n_vocab * c.n_embd * if std::ptr::eq(self.output.data().as_ptr(), self.token_embd.data().as_ptr()) { 1 } else { 2 }
            + c.n_layer * per_layer
    }

    /// Dominant weight quantization type (by bytes), for display.
    pub fn main_type(&self) -> GgmlType {
        let mut counts: Vec<(GgmlType, usize)> = Vec::new();
        for l in &self.layers {
            for w in l.all_weights() {
                match counts.iter_mut().find(|(t, _)| *t == w.ty) {
                    Some((_, n)) => *n += w.size_bytes(),
                    None => counts.push((w.ty, w.size_bytes())),
                }
            }
        }
        counts.into_iter().max_by_key(|(_, n)| *n).map(|(t, _)| t).unwrap_or(GgmlType::F32)
    }
}

/// A compute backend able to run the transformer forward pass.
pub trait Backend: Send {
    fn name(&self) -> String;
    /// Maximum number of tokens per `forward` call.
    fn max_batch(&self) -> usize;
    /// Context window size (KV cache capacity in positions).
    fn n_ctx(&self) -> usize;
    /// Evaluate `tokens` at positions `pos..pos+n` (the KV cache must already hold
    /// positions `< pos`). Writes the logits of the last token into `logits`
    /// (length `n_vocab`), or of every token (`n * n_vocab`) when `all_logits`.
    fn forward(&mut self, tokens: &[u32], pos: usize, all_logits: bool, logits: &mut [f32]) -> Result<()>;
    /// Final normalized hidden states (the output-head input) of the rows whose logits the
    /// last `forward` produced, `rows × n_embd`. Used by the online-learning adapters.
    /// Backends that cannot provide them return `None` (adapters are then disabled).
    fn hidden(&self) -> Option<&[f32]> {
        None
    }
}
