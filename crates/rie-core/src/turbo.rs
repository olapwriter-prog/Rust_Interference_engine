//! TurboQuant (Zandieh et al., 2025, arXiv:2504.19874), implemented natively.
//!
//! A vector is multiplied by a fixed random orthogonal transform (random signs followed
//! by a normalized Walsh–Hadamard transform over power-of-two blocks). After the
//! rotation every coordinate is close to Gaussian whatever the input looked like —
//! outlier channels are spread over the whole block — so a single scalar quantizer that
//! is optimal for N(0,1) (the Lloyd–Max codebook) works for every coordinate with just
//! one scale per block. Nothing is calibrated or trained: the codebook comes from the
//! maths, so data can be quantized online, one vector at a time.
//!
//! Two uses in the engine:
//! * [`KvType::Turbo4`](crate::KvType): a 4-bit KV cache (≈4.1 bits/value for head
//!   size 128, 3.9× smaller than f16). Attention runs in the rotated domain — the query
//!   is rotated once, scores are dot products against codebook values, and the weighted
//!   sum of values is rotated back once per head — so the cache is never dequantized.
//! * [`TurboIndex`]: a compressed vector index (2 or 4 bits per dimension) with
//!   lookup-table scoring, used by the on-device memory ([`crate::memory`]). The same
//!   design as the `turbovec` crate, without its dependencies.

use crate::quant::BlockQ8_0;
use half::f16;

/// Lloyd–Max reconstruction levels for N(0,1), positive half (Max, 1960).
const LM2: [f32; 2] = [0.4528, 1.5104];
const LM4: [f32; 8] = [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326];

/// Sorted codebook (`2^bits` levels) for N(0,1).
pub fn codebook(bits: u32) -> &'static [f32] {
    use std::sync::OnceLock;
    static C2: OnceLock<Vec<f32>> = OnceLock::new();
    static C4: OnceLock<Vec<f32>> = OnceLock::new();
    let build = |half: &[f32]| half.iter().rev().map(|v| -v).chain(half.iter().copied()).collect::<Vec<f32>>();
    match bits {
        2 => C2.get_or_init(|| build(&LM2)),
        4 => C4.get_or_init(|| build(&LM4)),
        _ => panic!("TurboQuant supports 2 or 4 bits"),
    }
}

/// Index of the nearest codebook level (levels sorted, boundaries at midpoints).
#[inline]
fn nearest(c: &[f32], v: f32) -> u8 {
    // the codebook is tiny (≤16): a branch-light linear count is fastest
    let mut i = 0u8;
    for w in c.windows(2) {
        i += (v > 0.5 * (w[0] + w[1])) as u8;
    }
    i
}

/// Largest power-of-two block (≤ 256) dividing `d`.
pub fn block_size(d: usize) -> usize {
    let mut b = 1;
    while d % (b * 2) == 0 && b < 256 {
        b *= 2;
    }
    b
}

/// In-place normalized fast Walsh–Hadamard transform (`x.len()` a power of two).
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= s);
}

/// Deterministic ±1 pattern for coordinate `i`.
#[inline]
fn sign(seed: u64, i: usize) -> f32 {
    let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    if (z ^ (z >> 31)) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// The fixed orthogonal transform `R = H·D` (block Hadamard after random signs).
#[derive(Clone, Debug)]
pub struct Rotation {
    signs: Vec<f32>,
    block: usize,
}

impl Rotation {
    pub fn new(dim: usize, seed: u64) -> Self {
        Rotation { signs: (0..dim).map(|i| sign(seed, i)).collect(), block: block_size(dim) }
    }
    pub fn dim(&self) -> usize {
        self.signs.len()
    }
    pub fn block(&self) -> usize {
        self.block
    }
    /// `x ← R x`
    pub fn apply(&self, x: &mut [f32]) {
        x.iter_mut().zip(&self.signs).for_each(|(v, s)| *v *= s);
        x.chunks_exact_mut(self.block).for_each(fwht);
    }
    /// `x ← Rᵀ x`
    pub fn apply_inverse(&self, x: &mut [f32]) {
        x.chunks_exact_mut(self.block).for_each(fwht);
        x.iter_mut().zip(&self.signs).for_each(|(v, s)| *v *= s);
    }
}

/// Seed of the KV-cache rotation (fixed: the cache never leaves the process, but a fixed
/// seed keeps runs reproducible).
pub const KV_SEED: u64 = 0x7542_6F51_7561_6E74;

/// The 4-bit Lloyd–Max levels as int8 (× [`KV_I8_SCALE`]), so the KV cache decodes with
/// one byte shuffle (`pshufb` / `tbl`) and scores use int8 dot products.
pub const KV_LEVELS_I8: [i8; 16] = [-126, -95, -74, -58, -43, -30, -18, -6, 6, 18, 30, 43, 58, 74, 95, 126];
pub const KV_I8_SCALE: f32 = 46.0;

fn kv_levels() -> [f32; 16] {
    KV_LEVELS_I8.map(|v| v as f32 / KV_I8_SCALE)
}

/// Bytes of one 4-bit TurboQuant row of `d` values (`d` a multiple of 32): one f16
/// scale per rotation block, then 16 bytes per 32 values (ggml Q4_0 nibble order: low
/// nibbles hold values 0..15 of a chunk, high nibbles 16..31).
pub fn row_bytes_4(d: usize) -> usize {
    2 * (d / block_size(d)) + d / 2
}

/// Quantize an already-rotated row to 4 bits (see [`row_bytes_4`] for the layout).
pub fn quantize_rotated_4(y: &[f32], out: &mut [u8]) {
    let d = y.len();
    debug_assert!(d % 32 == 0);
    let b = block_size(d);
    let nb = d / b;
    let c = kv_levels();
    let (scales, codes) = out.split_at_mut(2 * nb);
    codes.fill(0);
    for (bi, yb) in y.chunks_exact(b).enumerate() {
        let rms = (yb.iter().map(|v| v * v).sum::<f32>() / b as f32).sqrt();
        let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
        // codes from the RMS scale, then the least-squares optimal scale for those codes
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (j, &v) in yb.iter().enumerate() {
            let q = nearest(&c, v * inv);
            let g = bi * b + j;
            codes[(g / 32) * 16 + g % 16] |= q << ((g % 32) / 16 * 4);
            num += v * c[q as usize];
            den += c[q as usize] * c[q as usize];
        }
        let s = if den > 0.0 { num / den } else { 0.0 };
        scales[2 * bi..2 * bi + 2].copy_from_slice(&f16::from_f32(s).to_bits().to_le_bytes());
    }
}

/// Rotate and quantize a row in one go (`scratch.len() == x.len()`).
pub fn quantize_4(rot: &Rotation, x: &[f32], scratch: &mut [f32], out: &mut [u8]) {
    scratch.copy_from_slice(x);
    rot.apply(scratch);
    quantize_rotated_4(scratch, out);
}

#[inline]
fn scale_at(row: &[u8], bi: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([row[2 * bi], row[2 * bi + 1]])).to_f32()
}

/// Dot product of a rotated, Q8_0-quantized query with a 4-bit row.
#[inline]
pub fn dot_rotated_4(q: &[BlockQ8_0], row: &[u8]) -> f32 {
    let d = q.len() * 32;
    let b = block_size(d);
    let nb = d / b;
    let cpb = b / 32;
    let codes = &row[2 * nb..];
    let mut total = 0.0f32;
    for bi in 0..nb {
        let qs = &q[bi * cpb..(bi + 1) * cpb];
        let cs = &codes[bi * cpb * 16..(bi + 1) * cpb * 16];
        total += scale_at(row, bi) * dot_chunks(qs, cs);
    }
    total * (1.0 / KV_I8_SCALE)
}

/// `Σ_chunks d_q · Σ level_i8[code] · q_i8` over consecutive 32-value chunks.
#[inline]
fn dot_chunks(q: &[BlockQ8_0], codes: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::kernels::CpuFeatures::get().x86_simd() {
        return unsafe { x86::dot_chunks(q, codes) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::kernels::CpuFeatures::get().neon {
        return unsafe { arm::dot_chunks(q, codes) };
    }
    dot_chunks_scalar(q, codes)
}

fn dot_chunks_scalar(q: &[BlockQ8_0], codes: &[u8]) -> f32 {
    let mut s = 0.0f32;
    for (qb, cb) in q.iter().zip(codes.chunks_exact(16)) {
        let mut acc = 0i32;
        for i in 0..16 {
            acc += KV_LEVELS_I8[(cb[i] & 15) as usize] as i32 * qb.qs[i] as i32;
            acc += KV_LEVELS_I8[(cb[i] >> 4) as usize] as i32 * qb.qs[i + 16] as i32;
        }
        s += qb.d * acc as f32;
    }
    s
}

/// `y_rot += a * row` in the rotated domain.
#[inline]
pub fn axpy_rotated_4(y_rot: &mut [f32], a: f32, row: &[u8]) {
    let d = y_rot.len();
    let b = block_size(d);
    let nb = d / b;
    let codes = &row[2 * nb..];
    for bi in 0..nb {
        let s = a * scale_at(row, bi) * (1.0 / KV_I8_SCALE);
        let yb = &mut y_rot[bi * b..(bi + 1) * b];
        let cb = &codes[bi * b / 2..(bi + 1) * b / 2];
        #[cfg(target_arch = "x86_64")]
        if crate::kernels::CpuFeatures::get().x86_simd() {
            unsafe { x86::axpy_chunks(yb, s, cb) };
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if crate::kernels::CpuFeatures::get().neon {
            unsafe { arm::axpy_chunks(yb, s, cb) };
            continue;
        }
        for (yc, cc) in yb.chunks_exact_mut(32).zip(cb.chunks_exact(16)) {
            for i in 0..16 {
                yc[i] += s * KV_LEVELS_I8[(cc[i] & 15) as usize] as f32;
                yc[i + 16] += s * KV_LEVELS_I8[(cc[i] >> 4) as usize] as f32;
            }
        }
    }
}

/// Reconstruct a row (back in the original domain).
pub fn dequantize_4(rot: &Rotation, row: &[u8], out: &mut [f32]) {
    out.fill(0.0);
    axpy_rotated_4(out, 1.0, row);
    rot.apply_inverse(out);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::KV_LEVELS_I8;
    use crate::quant::BlockQ8_0;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn table() -> __m256i {
        let t = _mm_loadu_si128(KV_LEVELS_I8.as_ptr() as *const __m128i);
        _mm256_set_m128i(t, t)
    }

    /// 16 code bytes → 32 int8 levels in value order.
    #[inline(always)]
    unsafe fn decode(p: *const u8, tbl: __m256i) -> __m256i {
        let q = _mm_loadu_si128(p as *const __m128i);
        let both = _mm256_set_m128i(_mm_srli_epi16(q, 4), q);
        _mm256_shuffle_epi8(tbl, _mm256_and_si256(both, _mm256_set1_epi8(0x0F)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_chunks(q: &[BlockQ8_0], codes: &[u8]) -> f32 {
        let tbl = table();
        let mut acc = _mm256_setzero_ps();
        for (i, qb) in q.iter().enumerate() {
            let x = decode(codes.as_ptr().add(i * 16), tbl);
            let y = _mm256_loadu_si256(qb.qs.as_ptr() as *const __m256i);
            let ax = _mm256_sign_epi8(x, x);
            let sy = _mm256_sign_epi8(y, x);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(qb.d), _mm256_cvtepi32_ps(p), acc);
        }
        let h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_chunks(y: &mut [f32], a: f32, codes: &[u8]) {
        let tbl = table();
        let va = _mm256_set1_ps(a);
        for (c, yc) in y.chunks_exact_mut(32).enumerate() {
            let x = decode(codes.as_ptr().add(c * 16), tbl);
            let lo = _mm256_castsi256_si128(x);
            let hi = _mm256_extracti128_si256(x, 1);
            let parts = [lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)];
            for (k, part) in parts.iter().enumerate() {
                let v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(*part));
                let yp = yc.as_mut_ptr().add(k * 8);
                _mm256_storeu_ps(yp, _mm256_fmadd_ps(va, v, _mm256_loadu_ps(yp)));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::KV_LEVELS_I8;
    use crate::quant::BlockQ8_0;
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn decode(p: *const u8, tbl: int8x16_t) -> (int8x16_t, int8x16_t) {
        let q = vld1q_u8(p);
        (vqtbl1q_s8(tbl, vandq_u8(q, vdupq_n_u8(15))), vqtbl1q_s8(tbl, vshrq_n_u8(q, 4)))
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_chunks(q: &[BlockQ8_0], codes: &[u8]) -> f32 {
        let tbl = vld1q_s8(KV_LEVELS_I8.as_ptr());
        let mut acc = vdupq_n_f32(0.0);
        for (i, qb) in q.iter().enumerate() {
            let (xl, xh) = decode(codes.as_ptr().add(i * 16), tbl);
            let ql = vld1q_s8(qb.qs.as_ptr());
            let qh = vld1q_s8(qb.qs.as_ptr().add(16));
            // |level| ≤ 126, |q| ≤ 127: two products fit in i16
            let p0 = vmlal_s8(vmull_s8(vget_low_s8(xl), vget_low_s8(ql)), vget_high_s8(xl), vget_high_s8(ql));
            let p1 = vmlal_s8(vmull_s8(vget_low_s8(xh), vget_low_s8(qh)), vget_high_s8(xh), vget_high_s8(qh));
            let s = vpadalq_s16(vpaddlq_s16(p0), p1);
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), qb.d);
        }
        vaddvq_f32(acc)
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn axpy_chunks(y: &mut [f32], a: f32, codes: &[u8]) {
        let tbl = vld1q_s8(KV_LEVELS_I8.as_ptr());
        for (c, yc) in y.chunks_exact_mut(32).enumerate() {
            let (xl, xh) = decode(codes.as_ptr().add(c * 16), tbl);
            for (k, x) in [xl, xh].into_iter().enumerate() {
                let w0 = vmovl_s8(vget_low_s8(x));
                let w1 = vmovl_s8(vget_high_s8(x));
                let parts =
                    [vmovl_s16(vget_low_s16(w0)), vmovl_s16(vget_high_s16(w0)), vmovl_s16(vget_low_s16(w1)), vmovl_s16(vget_high_s16(w1))];
                for (j, p) in parts.into_iter().enumerate() {
                    let yp = yc.as_mut_ptr().add(k * 16 + j * 4);
                    vst1q_f32(yp, vfmaq_n_f32(vld1q_f32(yp), vcvtq_f32_s32(p), a));
                }
            }
        }
    }
}

/// Compressed vector index: TurboQuant codes plus a per-vector norm, searched with a
/// per-query lookup table (no decompression). Scores approximate inner products.
#[derive(Clone, Debug)]
pub struct TurboIndex {
    dim: usize,
    bits: u32,
    rot: Rotation,
    /// per vector: norm · least-squares scale
    scales: Vec<f32>,
    codes: Vec<u8>,
}

impl TurboIndex {
    /// `dim` must be a power of two ≥ 8; `bits` 2 or 4.
    pub fn new(dim: usize, bits: u32, seed: u64) -> Self {
        assert!(dim.is_power_of_two() && dim >= 8, "TurboIndex dimension must be a power of two ≥ 8");
        assert!(bits == 2 || bits == 4, "TurboIndex supports 2 or 4 bits");
        TurboIndex { dim, bits, rot: Rotation::new(dim, seed), scales: Vec::new(), codes: Vec::new() }
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn len(&self) -> usize {
        self.scales.len()
    }
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }
    fn code_bytes(&self) -> usize {
        self.dim * self.bits as usize / 8
    }
    /// Bytes held by the index.
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.codes.len()
    }

    /// Append a vector; returns its slot.
    pub fn add(&mut self, x: &[f32]) -> usize {
        assert_eq!(x.len(), self.dim);
        let mut y = x.to_vec();
        self.rot.apply(&mut y);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / self.dim as f32).sqrt();
        let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
        let c = codebook(self.bits);
        let per = 8 / self.bits as usize;
        let mut codes = vec![0u8; self.code_bytes()];
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (j, &v) in y.iter().enumerate() {
            let q = nearest(c, v * inv);
            codes[j / per] |= q << ((j % per) * self.bits as usize);
            num += v * c[q as usize];
            den += c[q as usize] * c[q as usize];
        }
        self.scales.push(if den > 0.0 { num / den } else { 0.0 });
        self.codes.extend_from_slice(&codes);
        self.scales.len() - 1
    }

    /// Overwrite slot `i` with the last vector and drop the last (O(1) removal).
    pub fn swap_remove(&mut self, i: usize) {
        let cb = self.code_bytes();
        let last = self.len() - 1;
        self.scales.swap_remove(i);
        if i != last {
            self.codes.copy_within(last * cb..(last + 1) * cb, i * cb);
        }
        self.codes.truncate(last * cb);
    }

    /// Approximate inner products of `q` with every stored vector.
    pub fn scores(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim);
        let mut qr = q.to_vec();
        self.rot.apply(&mut qr);
        let c = codebook(self.bits);
        let levels = c.len();
        let per = 8 / self.bits as usize;
        // byte-wise lookup table: one entry per (byte position, byte value)
        let nbytes = self.code_bytes();
        let mut lut = vec![0.0f32; nbytes * 256];
        let mask = (levels - 1) as u8;
        for p in 0..nbytes {
            for v in 0..256usize {
                let mut s = 0.0;
                for k in 0..per {
                    let code = ((v as u8) >> (k * self.bits as usize)) & mask;
                    s += qr[p * per + k] * c[code as usize];
                }
                lut[p * 256 + v] = s;
            }
        }
        self.codes
            .chunks_exact(nbytes)
            .zip(&self.scales)
            .map(|(codes, &s)| {
                let mut acc = [0.0f32; 4];
                for (p, &b) in codes.iter().enumerate() {
                    acc[p & 3] += lut[p * 256 + b as usize];
                }
                s * (acc[0] + acc[1] + acc[2] + acc[3])
            })
            .collect()
    }

    /// Top-`k` (slot, score) pairs, best first; `allow` filters slots.
    pub fn search(&self, q: &[f32], k: usize, allow: impl Fn(usize) -> bool) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = self.scores(q).into_iter().enumerate().filter(|(i, _)| allow(*i)).collect();
        all.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    fn phi(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    #[test]
    fn codebook_satisfies_lloyd_max_conditions() {
        // each level must be the conditional mean of N(0,1) over its decision cell
        for bits in [2, 4] {
            let c = codebook(bits);
            for i in 0..c.len() {
                let lo = if i == 0 { -12.0 } else { 0.5 * (c[i - 1] + c[i]) as f64 };
                let hi = if i + 1 == c.len() { 12.0 } else { 0.5 * (c[i] + c[i + 1]) as f64 };
                let (mut m0, mut m1) = (0.0, 0.0);
                let steps = 20000;
                for s in 0..steps {
                    let x = lo + (hi - lo) * (s as f64 + 0.5) / steps as f64;
                    m0 += phi(x);
                    m1 += x * phi(x);
                }
                let mean = m1 / m0;
                assert!((mean - c[i] as f64).abs() < 2e-3, "bits {bits} level {i}: {} vs {mean}", c[i]);
            }
        }
    }

    #[test]
    fn rotation_is_orthogonal() {
        for d in [64, 96, 128, 80] {
            let r = Rotation::new(d, 7);
            let mut s = 1u64;
            let x: Vec<f32> = (0..d).map(|_| rng(&mut s)).collect();
            let mut y = x.clone();
            r.apply(&mut y);
            let n0: f32 = x.iter().map(|v| v * v).sum();
            let n1: f32 = y.iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() < 1e-4 * n0);
            r.apply_inverse(&mut y);
            for (a, b) in x.iter().zip(&y) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn four_bit_rows_are_accurate_even_with_outlier_channels() {
        let d = 128;
        let rot = Rotation::new(d, KV_SEED);
        let mut s = 3u64;
        let mut rel = 0.0;
        for _ in 0..50 {
            let mut x: Vec<f32> = (0..d).map(|_| rng(&mut s)).collect();
            x[5] = 20.0; // a RoPE-style outlier channel
            x[77] = -9.0;
            let mut row = vec![0u8; row_bytes_4(d)];
            let mut scratch = vec![0.0; d];
            quantize_4(&rot, &x, &mut scratch, &mut row);
            let mut y = vec![0.0; d];
            dequantize_4(&rot, &row, &mut y);
            let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
            let norm: f32 = x.iter().map(|a| a * a).sum();
            rel += (err / norm).sqrt();
            // dot in the rotated domain equals dot with the reconstruction
            let mut q: Vec<f32> = (0..d).map(|_| rng(&mut s)).collect();
            let direct: f32 = q.iter().zip(&y).map(|(a, b)| a * b).sum();
            rot.apply(&mut q);
            let mut qq = vec![BlockQ8_0::default(); d / 32];
            crate::quant::quantize_row_q8_0(&q, &mut qq);
            let deq: Vec<f32> = crate::quant::dequantize_q8_0(&qq);
            let mut yr = vec![0.0; d];
            axpy_rotated_4(&mut yr, 1.0, &row);
            let exact: f32 = deq.iter().zip(&yr).map(|(a, b)| a * b).sum();
            assert!((dot_rotated_4(&qq, &row) - exact).abs() < 1e-3 * (1.0 + exact.abs()));
            // int8 query quantization: error bounded relative to |q|·|y|
            let qn = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            let yn = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((exact - direct).abs() < 0.01 * qn * yn, "{exact} vs {direct}");
            // SIMD and scalar paths agree
            let codes = &row[2..];
            assert!((dot_chunks(&qq, codes) - dot_chunks_scalar(&qq, codes)).abs() < 1e-3 * (1.0 + exact.abs()));
        }
        // Lloyd–Max 4-bit Gaussian quantizer: relative RMS error ≈ sqrt(0.0095) ≈ 0.10
        assert!(rel / 50.0 < 0.12, "relative error {}", rel / 50.0);
        assert_eq!(row_bytes_4(128), 66);
        assert_eq!(row_bytes_4(96), 6 + 48);
    }

    #[test]
    fn index_recovers_nearest_neighbours() {
        let d = 64;
        for bits in [2, 4] {
            let mut idx = TurboIndex::new(d, bits, 11);
            let mut s = 9u64;
            let data: Vec<Vec<f32>> = (0..300).map(|_| (0..d).map(|_| rng(&mut s)).collect()).collect();
            for v in &data {
                idx.add(v);
            }
            let mut hits = 0;
            for (i, v) in data.iter().enumerate().take(50) {
                // a noisy copy of a stored vector must retrieve it
                let q: Vec<f32> = v.iter().map(|x| x + 0.1 * rng(&mut s)).collect();
                if idx.search(&q, 1, |_| true)[0].0 == i {
                    hits += 1;
                }
            }
            assert!(hits >= if bits == 4 { 49 } else { 40 }, "{bits}-bit recall {hits}/50");
            assert_eq!(idx.memory_bytes(), 300 * (4 + d * bits as usize / 8));
            idx.swap_remove(0);
            assert_eq!(idx.len(), 299);
            assert_eq!(idx.search(&data[299], 1, |_| true)[0].0, 0);
        }
    }
}
