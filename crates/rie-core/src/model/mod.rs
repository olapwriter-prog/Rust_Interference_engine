//! Model definition: hyper-parameters, weights and the backend interface.

pub mod config;
pub mod cpu;

pub use config::{Activation, Arch, ModelConfig};

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
    pub w_gate: Weight,
    pub w_up: Weight,
    pub w_down: Weight,
    pub post_ffn_norm: Option<Vec<f32>>,
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
    if t.dims.len() > 2 && t.dims[2..].iter().any(|&d| d != 1) {
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
            let (w_gate, w_up) = match opt(&p("ffn_gate.weight"))? {
                Some(gw) => (gw, get(&p("ffn_up.weight"))?),
                None => {
                    let up = get(&p("ffn_up.weight"))?;
                    if up.rows != 2 * cfg.n_ff {
                        bail!(Unsupported, "layer {il}: missing ffn_gate and ffn_up is not fused");
                    }
                    (up.slice_rows(0, cfg.n_ff), up.slice_rows(cfg.n_ff, cfg.n_ff))
                }
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
                w_down: get(&p("ffn_down.weight"))?,
                post_ffn_norm: opt_vec(&p("post_ffw_norm.weight"))?,
                wq,
                wk,
                wv,
                w_gate,
                w_up,
            };
            // Shape validation catches mismatched metadata early with a clear message.
            let checks = [
                ("attn_q", &lw.wq, q_dim, cfg.n_embd),
                ("attn_k", &lw.wk, kv_dim, cfg.n_embd),
                ("attn_v", &lw.wv, kv_dim, cfg.n_embd),
                ("attn_output", &lw.wo, cfg.n_embd, q_dim),
                ("ffn_gate", &lw.w_gate, cfg.n_ff, cfg.n_embd),
                ("ffn_up", &lw.w_up, cfg.n_ff, cfg.n_embd),
                ("ffn_down", &lw.w_down, cfg.n_embd, cfg.n_ff),
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

    /// Total bytes of weights (as stored).
    pub fn weight_bytes(&self) -> usize {
        let mut n = self.token_embd.size_bytes();
        if !std::ptr::eq(self.output.data().as_ptr(), self.token_embd.data().as_ptr()) {
            n += self.output.size_bytes();
        }
        for l in &self.layers {
            for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                n += w.size_bytes();
            }
        }
        n
    }

    pub fn n_params(&self) -> usize {
        let c = &self.cfg;
        let per_layer = c.n_embd * (c.q_dim() + 2 * c.kv_dim()) + c.q_dim() * c.n_embd + 3 * c.n_embd * c.n_ff;
        c.n_vocab * c.n_embd * if std::ptr::eq(self.output.data().as_ptr(), self.token_embd.data().as_ptr()) { 1 } else { 2 }
            + c.n_layer * per_layer
    }

    /// Dominant weight quantization type (by bytes), for display.
    pub fn main_type(&self) -> GgmlType {
        let mut counts: Vec<(GgmlType, usize)> = Vec::new();
        for l in &self.layers {
            for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
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
}
