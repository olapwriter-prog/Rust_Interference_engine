//! Architecture hyper-parameters read from GGUF metadata.

use crate::error::{bail, Error, Result};
use crate::gguf::GgufFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Llama 2/3.x, Mistral, SmolLM, TinyLlama, DeepSeek-R1-Distill-Llama, …
    Llama,
    /// Qwen2 / Qwen2.5 / DeepSeek-R1-Distill-Qwen (QKV bias).
    Qwen2,
    /// Qwen3 dense (per-head Q/K RMSNorm).
    Qwen3,
    /// Phi-3 / Phi-3.5 / Phi-4-mini (fused QKV & gate-up).
    Phi3,
    Gemma,
    Gemma2,
    Gemma3,
    /// IBM Granite 3.x (Llama with residual/logit multipliers).
    Granite,
}

impl Arch {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "llama" | "mistral" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "phi3" => Arch::Phi3,
            "gemma" => Arch::Gemma,
            "gemma2" => Arch::Gemma2,
            "gemma3" => Arch::Gemma3,
            "granite" => Arch::Granite,
            _ => return None,
        })
    }
    pub fn supported() -> &'static [&'static str] {
        &["llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "granite"]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    GeluTanh,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: Arch,
    pub arch_name: String,
    pub name: Option<String>,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f32,
    pub ctx_train: usize,
    pub rope_dim: usize,
    pub rope_neox: bool,
    pub rope_base: f32,
    pub rope_freq_scale: f32,
    /// RoPE base for sliding-window layers (Gemma 3).
    pub rope_base_swa: Option<f32>,
    pub act: Activation,
    pub embd_scale: f32,
    pub attn_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub attn_softcap: Option<f32>,
    pub final_softcap: Option<f32>,
    /// Sliding window size (0 = full attention on every layer).
    pub swa_window: usize,
    /// Every `swa_pattern`-th layer is global, the rest use the sliding window.
    pub swa_pattern: usize,
}

impl ModelConfig {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa_window > 0 && self.swa_pattern > 0 && (layer % self.swa_pattern) < self.swa_pattern - 1
    }

    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch_name = g.get_str("general.architecture").ok_or_else(|| Error::Missing("general.architecture".into()))?.to_string();
        let arch = Arch::from_name(&arch_name)
            .ok_or_else(|| Error::Unsupported(format!("architecture '{arch_name}' (supported: {})", Arch::supported().join(", "))))?;
        let key = |k: &str| format!("{arch_name}.{k}");
        let req = |k: &str| -> Result<usize> {
            g.get_u64(&key(k)).map(|v| v as usize).ok_or_else(|| Error::Missing(format!("metadata {}", key(k))))
        };
        let n_embd = req("embedding_length")?;
        let n_layer = req("block_count")?;
        let n_head = req("attention.head_count")?;
        let n_head_kv = g.get_u64(&key("attention.head_count_kv")).map(|v| v as usize).unwrap_or(n_head);
        let head_dim = g.get_u64(&key("attention.key_length")).map(|v| v as usize).unwrap_or(n_embd / n_head);
        if let Some(vl) = g.get_u64(&key("attention.value_length")) {
            if vl as usize != head_dim {
                bail!(Unsupported, "value head dim {vl} != key head dim {head_dim}");
            }
        }
        let n_ff = req("feed_forward_length")?;
        let n_vocab = g
            .get_arr("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .or_else(|| g.get_u64(&key("vocab_size")).map(|v| v as usize))
            .ok_or_else(|| Error::Missing("vocabulary".into()))?;
        let rms_eps = g.get_f32(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let ctx_train = g.get_u64(&key("context_length")).map(|v| v as usize).unwrap_or(4096);
        let rope_dim = g.get_u64(&key("rope.dimension_count")).map(|v| v as usize).unwrap_or(head_dim);
        let rope_base = g.get_f32(&key("rope.freq_base")).unwrap_or(10000.0);
        let mut rope_freq_scale = 1.0;
        match g.get_str(&key("rope.scaling.type")) {
            Some("linear") => {
                if let Some(f) = g.get_f32(&key("rope.scaling.factor")) {
                    if f > 0.0 {
                        rope_freq_scale = 1.0 / f;
                    }
                }
            }
            Some("yarn") => {
                // YaRN only matters beyond the original training context; we run at
                // ≤ original context on mobile, where YaRN reduces to plain RoPE.
            }
            _ => {}
        }
        if let Some(s) = g.get_f32(&key("rope.scale_linear")) {
            if s > 0.0 {
                rope_freq_scale = 1.0 / s;
            }
        }

        let rope_neox = matches!(arch, Arch::Qwen2 | Arch::Qwen3 | Arch::Phi3 | Arch::Gemma | Arch::Gemma2 | Arch::Gemma3);
        let gemma = matches!(arch, Arch::Gemma | Arch::Gemma2 | Arch::Gemma3);
        let act = if gemma { Activation::GeluTanh } else { Activation::Silu };
        let mut embd_scale = if gemma { (n_embd as f32).sqrt() } else { 1.0 };
        let mut attn_scale = 1.0 / (head_dim as f32).sqrt();
        let mut residual_scale = 1.0;
        let mut logit_scale = 1.0;
        let mut attn_softcap = None;
        let mut final_softcap = None;
        let mut swa_window = 0;
        let mut swa_pattern = 0;
        let mut rope_base_swa = None;
        match arch {
            Arch::Gemma2 => {
                attn_softcap = Some(g.get_f32(&key("attn_logit_softcapping")).unwrap_or(50.0));
                final_softcap = Some(g.get_f32(&key("final_logit_softcapping")).unwrap_or(30.0));
                swa_window = g.get_u64(&key("attention.sliding_window")).unwrap_or(4096) as usize;
                swa_pattern = 2;
                if n_layer == 46 {
                    attn_scale = 1.0 / ((n_embd / n_head) as f32).sqrt(); // 27B
                }
            }
            Arch::Gemma3 => {
                swa_window = g.get_u64(&key("attention.sliding_window")).unwrap_or(1024) as usize;
                swa_pattern = 6;
                rope_base_swa = Some(10000.0);
                if n_layer == 62 {
                    attn_scale = 1.0 / ((n_embd / n_head) as f32).sqrt(); // 27B
                }
                final_softcap = g.get_f32(&key("final_logit_softcapping"));
                attn_softcap = g.get_f32(&key("attn_logit_softcapping"));
            }
            Arch::Granite => {
                embd_scale = g.get_f32(&key("embedding_scale")).unwrap_or(1.0);
                residual_scale = g.get_f32(&key("residual_scale")).unwrap_or(1.0);
                logit_scale = g.get_f32(&key("logit_scale")).unwrap_or(1.0);
                if let Some(s) = g.get_f32(&key("attention.scale")) {
                    attn_scale = s;
                }
            }
            _ => {}
        }
        if n_head % n_head_kv != 0 {
            bail!(Format, "n_head {n_head} not divisible by n_head_kv {n_head_kv}");
        }
        if rope_dim > head_dim || rope_dim % 2 != 0 {
            bail!(Format, "invalid rope dimension {rope_dim}");
        }
        Ok(ModelConfig {
            arch,
            arch_name,
            name: g.get_str("general.name").map(str::to_string),
            n_vocab,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            rms_eps,
            ctx_train,
            rope_dim,
            rope_neox,
            rope_base,
            rope_freq_scale,
            rope_base_swa,
            act,
            embd_scale,
            attn_scale,
            residual_scale,
            logit_scale,
            attn_softcap,
            final_softcap,
            swa_window,
            swa_pattern,
        })
    }
}
