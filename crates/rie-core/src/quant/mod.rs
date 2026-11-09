//! ggml-compatible quantization formats.
//!
//! Weights stay in their on-disk block layout (no repacking needed for the CPU path),
//! activations are quantized on the fly to 8-bit (`Q8_0` blocks of 32 for the legacy
//! formats, `Q8_K` blocks of 256 for k-quants) so the inner loops are pure int8 dot
//! products — the operation phones accelerate with `SDOT`/`SMMLA`.

pub mod reference;

use half::{bf16, f16};

pub const QK: usize = 32;
pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            _ => return None,
        })
    }
    pub fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::BF16 => 30,
        }
    }
    /// Elements per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => QK,
            _ => QK_K,
        }
    }
    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_size() * self.block_bytes()
    }
    pub fn bits_per_weight(self) -> f32 {
        self.block_bytes() as f32 * 8.0 / self.block_size() as f32
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
        }
    }
    /// Which activation format the fast dot kernel for this weight type consumes.
    pub fn act_format(self) -> ActFormat {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => ActFormat::Q8_0,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q2_K | Self::Q3_K => ActFormat::Q8K,
            Self::F32 | Self::F16 | Self::BF16 => ActFormat::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActFormat {
    F32,
    Q8_0,
    Q8K,
}

#[inline(always)]
pub fn f16_at(b: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[off], b[off + 1]])).to_f32()
}

/// Activation block for the 32-element formats. `s = d * sum(qs)` is precomputed so that
/// formats with a min/offset (`Q4_1`, `Q5_1`) need no extra work per block.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct BlockQ8_0 {
    pub qs: [i8; QK],
    pub d: f32,
    pub s: f32,
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        BlockQ8_0 { qs: [0; QK], d: 0.0, s: 0.0 }
    }
}

/// Activation block for k-quants.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct BlockQ8K {
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
    pub d: f32,
}

impl Default for BlockQ8K {
    fn default() -> Self {
        BlockQ8K { qs: [0; QK_K], bsums: [0; QK_K / 16], d: 0.0 }
    }
}

pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    debug_assert_eq!(x.len(), out.len() * QK);
    crate::kernels::quantize_q8_0(x, out)
}

pub fn quantize_row_q8_0_scalar(x: &[f32], out: &mut [BlockQ8_0]) {
    for (xb, ob) in x.chunks_exact(QK).zip(out.iter_mut()) {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut sum = 0i32;
        for (q, &v) in ob.qs.iter_mut().zip(xb) {
            let r = (v * id).round() as i32;
            *q = r.clamp(-127, 127) as i8;
            sum += *q as i32;
        }
        ob.d = d;
        ob.s = d * sum as f32;
    }
}

pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) {
    debug_assert_eq!(x.len(), out.len() * QK_K);
    for (xb, ob) in x.chunks_exact(QK_K).zip(out.iter_mut()) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            ob.d = 0.0;
            ob.qs = [0; QK_K];
            ob.bsums = [0; QK_K / 16];
            continue;
        }
        let iscale = -127.0 / max;
        for (q, &v) in ob.qs.iter_mut().zip(xb) {
            let r = (iscale * v).round() as i32;
            *q = r.clamp(-127, 127) as i8;
        }
        for (j, s) in ob.bsums.iter_mut().enumerate() {
            *s = ob.qs[j * 16..j * 16 + 16].iter().map(|&q| q as i16).sum();
        }
        ob.d = 1.0 / iscale;
    }
}

/// Dequantize `n` contiguous elements (a whole number of blocks) of `ty` from `src` into `dst`.
pub fn dequantize_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    debug_assert!(src.len() >= ty.row_bytes(n));
    match ty {
        GgmlType::F32 => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *d = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
        _ => {
            let bs = ty.block_size();
            let bb = ty.block_bytes();
            for (i, out) in dst.chunks_exact_mut(bs).enumerate() {
                let b = &src[i * bb..(i + 1) * bb];
                reference::dequantize_block(ty, b, out);
            }
        }
    }
}

/// Dequantize the int8 activation blocks (used by tests).
pub fn dequantize_q8_0(blocks: &[BlockQ8_0]) -> Vec<f32> {
    blocks.iter().flat_map(|b| b.qs.iter().map(move |&q| q as f32 * b.d)).collect()
}
pub fn dequantize_q8_k(blocks: &[BlockQ8K]) -> Vec<f32> {
    blocks.iter().flat_map(|b| b.qs.iter().map(move |&q| q as f32 * b.d)).collect()
}

/// Simple quantizers producing valid blocks. They are not tuned for minimal error
/// (use llama.cpp's `llama-quantize` to produce production model files); they exist
/// for tests, the GPU repacker and for converting unsupported tensor types.
pub mod quantize {
    use super::*;

    pub fn q8_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / QK * 34);
        for xb in x.chunks_exact(QK) {
            let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for &v in xb {
                out.push(((v * id).round() as i32).clamp(-127, 127) as i8 as u8);
            }
        }
        out
    }

    pub fn q4_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / QK * 18);
        for xb in x.chunks_exact(QK) {
            let mut max = 0.0f32;
            for &v in xb {
                if v.abs() > max.abs() {
                    max = v;
                }
            }
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for j in 0..QK / 2 {
                let q0 = ((xb[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                let q1 = ((xb[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                out.push(q0 | (q1 << 4));
            }
        }
        out
    }

    pub fn f16(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes()).collect()
    }

    pub fn f32(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}
