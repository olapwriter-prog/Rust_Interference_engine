//! Synthetic model generation for tests, benchmarks and bring-up on new devices
//! (`rie bench --synthetic` runs without downloading any model).

use crate::gguf::writer::GgufWriter;
use crate::gguf::Value;
use crate::quant::GgmlType;
use half::f16;

/// Small deterministic PRNG.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn sym(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.sym() * scale).collect()
    }
}

/// Byte offsets of the f16 scale fields in one block.
pub fn scale_offsets(ty: GgmlType) -> &'static [usize] {
    match ty {
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
        GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        _ => &[],
    }
}

/// Random but valid weight data for `rows × cols` of `ty`. `gain` scales the magnitude
/// (1.0 ≈ values within ±0.1, typical for trained checkpoints).
pub fn random_weight(rng: &mut Rng, ty: GgmlType, rows: usize, cols: usize, gain: f32) -> Vec<u8> {
    match ty {
        GgmlType::F32 => rng.vec(rows * cols, 0.1 * gain).iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => rng.vec(rows * cols, 0.1 * gain).iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        GgmlType::BF16 => rng.vec(rows * cols, 0.1 * gain).iter().flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        _ => {
            let n_blocks = rows * cols / ty.block_size();
            let bb = ty.block_bytes();
            let mut data: Vec<u8> = Vec::with_capacity(n_blocks * bb);
            while data.len() < n_blocks * bb {
                data.extend_from_slice(&rng.next_u64().to_le_bytes());
            }
            data.truncate(n_blocks * bb);
            let scale = gain
                * match ty {
                    GgmlType::Q4_0 | GgmlType::Q4_1 => 0.01,
                    GgmlType::Q5_0 | GgmlType::Q5_1 => 0.005,
                    GgmlType::Q8_0 => 0.001,
                    GgmlType::Q2_K => 0.004,
                    GgmlType::Q3_K => 0.003,
                    GgmlType::Q4_K => 0.001,
                    GgmlType::Q5_K => 0.0007,
                    GgmlType::Q6_K => 0.0001,
                    _ => 0.01,
                };
            for b in 0..n_blocks {
                for (i, &off) in scale_offsets(ty).iter().enumerate() {
                    let mut v = (rng.f32() * 0.5 + 0.5) * scale;
                    if i == 0
                        && matches!(ty, GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q3_K | GgmlType::Q6_K)
                        && rng.f32() < 0.3
                    {
                        v = -v;
                    }
                    if i == 1 && matches!(ty, GgmlType::Q4_1 | GgmlType::Q5_1) {
                        v = -v * 8.0;
                    }
                    let bits = f16::from_f32(v).to_bits().to_le_bytes();
                    data[b * bb + off] = bits[0];
                    data[b * bb + off + 1] = bits[1];
                }
            }
            data
        }
    }
}

/// Shape and feature description of a synthetic transformer.
#[derive(Debug, Clone)]
pub struct SyntheticSpec {
    pub arch: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub ctx: usize,
    pub wtype: GgmlType,
    /// Type of token embedding / output matrices.
    pub embd_type: GgmlType,
    pub tied_output: bool,
    pub qkv_bias: bool,
    pub qk_norm: bool,
    pub post_norms: bool,
    pub fused: bool,
    pub swa_window: usize,
    pub seed: u64,
}

impl SyntheticSpec {
    pub fn tiny(arch: &str, wtype: GgmlType) -> Self {
        SyntheticSpec {
            arch: arch.into(),
            n_vocab: 384,
            n_embd: 256,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 64,
            n_ff: 512,
            rope_dim: 64,
            ctx: 256,
            wtype,
            embd_type: if wtype.block_size() > 1 { GgmlType::Q8_0 } else { wtype },
            tied_output: false,
            qkv_bias: arch == "qwen2",
            qk_norm: matches!(arch, "qwen3" | "gemma3"),
            post_norms: matches!(arch, "gemma2" | "gemma3"),
            fused: arch == "phi3",
            swa_window: if matches!(arch, "gemma2" | "gemma3") { 5 } else { 0 },
            seed: 1,
        }
    }

    /// Shapes of popular on-device models (weights are random).
    pub fn preset(name: &str, wtype: GgmlType) -> Option<Self> {
        let (arch, v, e, l, h, kv, hd, ff) = match name {
            "llama-3.2-1b" => ("llama", 128256, 2048, 16, 32, 8, 64, 8192),
            "llama-3.2-3b" => ("llama", 128256, 3072, 28, 24, 8, 128, 8192),
            "qwen2.5-0.5b" => ("qwen2", 151936, 896, 24, 14, 2, 64, 4864),
            "qwen2.5-1.5b" => ("qwen2", 151936, 1536, 28, 12, 2, 128, 8960),
            "qwen3-0.6b" => ("qwen3", 151936, 1024, 28, 16, 8, 128, 3072),
            "qwen3-1.7b" => ("qwen3", 151936, 2048, 28, 16, 8, 128, 6144),
            "qwen3-4b" => ("qwen3", 151936, 2560, 36, 32, 8, 128, 9728),
            "gemma-3-1b" => ("gemma3", 262144, 1152, 26, 4, 1, 256, 6912),
            "phi-4-mini" => ("phi3", 200064, 3072, 32, 24, 8, 128, 8192),
            "smollm2-360m" => ("llama", 49152, 960, 32, 15, 5, 64, 2560),
            _ => return None,
        };
        let mut s = Self::tiny(arch, wtype);
        s.n_vocab = v;
        s.n_embd = e;
        s.n_layer = l;
        s.n_head = h;
        s.n_head_kv = kv;
        s.head_dim = hd;
        s.n_ff = ff;
        s.rope_dim = if arch == "phi3" { 96 } else { hd };
        s.ctx = 4096;
        s.swa_window = if arch == "gemma3" { 512 } else { 0 };
        s.tied_output = matches!(
            name,
            "llama-3.2-1b"
                | "llama-3.2-3b"
                | "qwen2.5-0.5b"
                | "qwen2.5-1.5b"
                | "qwen3-0.6b"
                | "qwen3-1.7b"
                | "qwen3-4b"
                | "gemma-3-1b"
                | "phi-4-mini"
                | "smollm2-360m"
        );
        s.embd_type = if wtype.block_size() > 1 { GgmlType::Q6_K } else { wtype };
        if e % 256 != 0 || ff % 256 != 0 {
            // k-quants need multiples of 256: fall back like llama-quantize does
            if matches!(wtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q2_K | GgmlType::Q3_K) {
                s.wtype = GgmlType::Q8_0;
            }
            if matches!(s.embd_type, GgmlType::Q6_K) {
                s.embd_type = GgmlType::Q8_0;
            }
        }
        Some(s)
    }

    pub const PRESETS: &'static [&'static str] = &[
        "smollm2-360m",
        "qwen2.5-0.5b",
        "qwen3-0.6b",
        "gemma-3-1b",
        "llama-3.2-1b",
        "qwen2.5-1.5b",
        "qwen3-1.7b",
        "llama-3.2-3b",
        "qwen3-4b",
        "phi-4-mini",
    ];

    /// Build an in-memory GGUF image (SentencePiece byte vocabulary).
    pub fn build(&self) -> Vec<u8> {
        let mut rng = Rng::new(self.seed);
        let mut w = GgufWriter::new();
        let a = &self.arch;
        let s = |v: &str| Value::String(v.to_string());
        w.kv("general.architecture", s(a));
        w.kv("general.name", s(&format!("synthetic-{a}")));
        w.kv(&format!("{a}.embedding_length"), Value::U32(self.n_embd as u32));
        w.kv(&format!("{a}.block_count"), Value::U32(self.n_layer as u32));
        w.kv(&format!("{a}.attention.head_count"), Value::U32(self.n_head as u32));
        w.kv(&format!("{a}.attention.head_count_kv"), Value::U32(self.n_head_kv as u32));
        w.kv(&format!("{a}.attention.key_length"), Value::U32(self.head_dim as u32));
        w.kv(&format!("{a}.attention.value_length"), Value::U32(self.head_dim as u32));
        w.kv(&format!("{a}.feed_forward_length"), Value::U32(self.n_ff as u32));
        w.kv(&format!("{a}.context_length"), Value::U32(self.ctx as u32));
        w.kv(&format!("{a}.rope.dimension_count"), Value::U32(self.rope_dim as u32));
        w.kv(&format!("{a}.rope.freq_base"), Value::F32(10000.0));
        w.kv(&format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-6));
        if self.swa_window > 0 {
            w.kv(&format!("{a}.attention.sliding_window"), Value::U32(self.swa_window as u32));
        }
        if a == "granite" {
            w.kv("granite.embedding_scale", Value::F32(12.0));
            w.kv("granite.residual_scale", Value::F32(0.22));
            w.kv("granite.logit_scale", Value::F32(8.0));
            w.kv("granite.attention.scale", Value::F32(0.0078125 * 8.0));
        }
        if a == "gemma3" {
            w.kv("gemma3.rope.freq_base", Value::F32(1_000_000.0));
        }
        // vocabulary: <unk> <s> </s> + 256 byte tokens + filler pieces
        let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let mut types = vec![Value::I32(2), Value::I32(3), Value::I32(3)];
        for b in 0..256 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(Value::I32(6));
        }
        let pieces = ["\u{2581}the", "\u{2581}a", "\u{2581}phone", "\u{2581}fast", "in", "er", "\u{2581}", "e", "t", "h", "o", "n"];
        for p in pieces {
            if tokens.len() < self.n_vocab {
                tokens.push(p.to_string());
                types.push(Value::I32(1));
            }
        }
        while tokens.len() < self.n_vocab {
            tokens.push(format!("[tok{}]", tokens.len()));
            types.push(Value::I32(1));
        }
        tokens.truncate(self.n_vocab);
        types.truncate(self.n_vocab);
        let scores: Vec<Value> = (0..self.n_vocab).map(|i| Value::F32(-(i as f32))).collect();
        w.kv("tokenizer.ggml.model", s("llama"));
        w.kv("tokenizer.ggml.tokens", Value::Array(tokens.into_iter().map(Value::String).collect()));
        w.kv("tokenizer.ggml.scores", Value::Array(scores));
        w.kv("tokenizer.ggml.token_type", Value::Array(types));
        w.kv("tokenizer.ggml.bos_token_id", Value::U32(1));
        w.kv("tokenizer.ggml.eos_token_id", Value::U32(2));
        w.kv("tokenizer.ggml.unknown_token_id", Value::U32(0));

        let (e, ff, qd, kvd, hd) = (
            self.n_embd as u64,
            self.n_ff as u64,
            (self.n_head * self.head_dim) as u64,
            (self.n_head_kv * self.head_dim) as u64,
            self.head_dim as u64,
        );
        let norm = |rng: &mut Rng, n: usize| -> Vec<u8> { (0..n).flat_map(|_| (1.0 + 0.1 * rng.sym()).to_le_bytes()).collect() };
        let vec_f32 = |rng: &mut Rng, n: usize, sc: f32| -> Vec<u8> { rng.vec(n, sc).iter().flat_map(|v| v.to_le_bytes()).collect() };
        let wt = self.wtype;
        let mat = |rng: &mut Rng, rows: u64, cols: u64, ty: GgmlType| random_weight(rng, ty, rows as usize, cols as usize, 1.0);

        let embd = mat(&mut rng, self.n_vocab as u64, e, self.embd_type);
        w.tensor("token_embd.weight", &[e, self.n_vocab as u64], self.embd_type, embd);
        w.tensor("output_norm.weight", &[e], GgmlType::F32, norm(&mut rng, self.n_embd));
        if !self.tied_output {
            let out = mat(&mut rng, self.n_vocab as u64, e, self.embd_type);
            w.tensor("output.weight", &[e, self.n_vocab as u64], self.embd_type, out);
        }
        for il in 0..self.n_layer {
            let p = |n: &str| format!("blk.{il}.{n}");
            w.tensor(&p("attn_norm.weight"), &[e], GgmlType::F32, norm(&mut rng, self.n_embd));
            if self.fused {
                w.tensor(&p("attn_qkv.weight"), &[e, qd + 2 * kvd], wt, mat(&mut rng, qd + 2 * kvd, e, wt));
            } else {
                w.tensor(&p("attn_q.weight"), &[e, qd], wt, mat(&mut rng, qd, e, wt));
                w.tensor(&p("attn_k.weight"), &[e, kvd], wt, mat(&mut rng, kvd, e, wt));
                w.tensor(&p("attn_v.weight"), &[e, kvd], wt, mat(&mut rng, kvd, e, wt));
            }
            if self.qkv_bias {
                w.tensor(&p("attn_q.bias"), &[qd], GgmlType::F32, vec_f32(&mut rng, qd as usize, 0.1));
                w.tensor(&p("attn_k.bias"), &[kvd], GgmlType::F32, vec_f32(&mut rng, kvd as usize, 0.1));
                w.tensor(&p("attn_v.bias"), &[kvd], GgmlType::F32, vec_f32(&mut rng, kvd as usize, 0.1));
            }
            if self.qk_norm {
                w.tensor(&p("attn_q_norm.weight"), &[hd], GgmlType::F32, norm(&mut rng, self.head_dim));
                w.tensor(&p("attn_k_norm.weight"), &[hd], GgmlType::F32, norm(&mut rng, self.head_dim));
            }
            w.tensor(&p("attn_output.weight"), &[qd, e], wt, mat(&mut rng, e, qd, wt));
            if self.post_norms {
                w.tensor(&p("post_attention_norm.weight"), &[e], GgmlType::F32, norm(&mut rng, self.n_embd));
                w.tensor(&p("post_ffw_norm.weight"), &[e], GgmlType::F32, norm(&mut rng, self.n_embd));
            }
            w.tensor(&p("ffn_norm.weight"), &[e], GgmlType::F32, norm(&mut rng, self.n_embd));
            if self.fused {
                w.tensor(&p("ffn_up.weight"), &[e, 2 * ff], wt, mat(&mut rng, 2 * ff, e, wt));
            } else {
                w.tensor(&p("ffn_gate.weight"), &[e, ff], wt, mat(&mut rng, ff, e, wt));
                w.tensor(&p("ffn_up.weight"), &[e, ff], wt, mat(&mut rng, ff, e, wt));
            }
            w.tensor(&p("ffn_down.weight"), &[ff, e], wt, mat(&mut rng, e, ff, wt));
        }
        w.to_bytes()
    }
}
