//! Portable scalar reference implementations: block dequantization and int8 dot products.
//! These define the semantics the SIMD kernels must reproduce and serve as the fallback
//! on CPUs without NEON/AVX2.

use super::{f16_at, BlockQ8K, BlockQ8_0, GgmlType, QK, QK_K};

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Unpack the 12-byte scale/min field of Q4_K/Q5_K into 8 scales and 8 mins.
#[inline]
pub fn unpack_k4_scales(q: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        let (a, b) = get_scale_min_k4(j, q);
        sc[j] = a;
        m[j] = b;
    }
    (sc, m)
}

/// Unpack the 12-byte Q3_K scale field into 16 signed scales (already offset by -32).
#[inline]
pub fn unpack_q3k_scales(s: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes([s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut out = [0i8; 16];
    for i in 0..4 {
        let b = aux[i].to_le_bytes();
        for k in 0..4 {
            out[4 * i + k] = (b[k] as i8).wrapping_sub(32);
        }
    }
    out
}

pub fn dequantize_block(ty: GgmlType, b: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::Q4_0 => {
            let d = f16_at(b, 0);
            let qs = &b[2..18];
            for j in 0..16 {
                y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
                y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
            }
        }
        GgmlType::Q4_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            let qs = &b[4..20];
            for j in 0..16 {
                y[j] = (qs[j] & 0xF) as f32 * d + m;
                y[j + 16] = (qs[j] >> 4) as f32 * d + m;
            }
        }
        GgmlType::Q5_0 => {
            let d = f16_at(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let qs = &b[6..22];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = (((qs[j] as u32 & 0xF) | xh0) as i32 - 16) as f32 * d;
                y[j + 16] = (((qs[j] as u32 >> 4) | xh1) as i32 - 16) as f32 * d;
            }
        }
        GgmlType::Q5_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let qs = &b[8..24];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = ((qs[j] as u32 & 0xF) | xh0) as f32 * d + m;
                y[j + 16] = ((qs[j] as u32 >> 4) | xh1) as f32 * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = f16_at(b, 0);
            for j in 0..QK {
                y[j] = b[2 + j] as i8 as f32 * d;
            }
        }
        GgmlType::Q2_K => {
            let scales = &b[0..16];
            let q = &b[16..80];
            let d = f16_at(b, 80);
            let min = f16_at(b, 82);
            let mut yi = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &q[n * 32..];
                let mut shift = 0;
                for _ in 0..4 {
                    for half in 0..2 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for l in 0..16 {
                            y[yi] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                            yi += 1;
                        }
                    }
                    shift += 2;
                }
            }
        }
        GgmlType::Q3_K => {
            let hm = &b[0..32];
            let q = &b[32..96];
            let scales = unpack_q3k_scales(&b[96..108]);
            let d_all = f16_at(b, 108);
            let mut m = 1u8;
            let mut is = 0;
            let mut yi = 0;
            for n in 0..2 {
                let q = &q[n * 32..];
                let mut shift = 0;
                for _ in 0..4 {
                    for half in 0..2 {
                        let dl = d_all * scales[is] as f32;
                        is += 1;
                        for l in 0..16 {
                            let idx = l + 16 * half;
                            let v = ((q[idx] >> shift) & 3) as i32 - if hm[idx] & m != 0 { 0 } else { 4 };
                            y[yi] = dl * v as f32;
                            yi += 1;
                        }
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        GgmlType::Q4_K => {
            let d = f16_at(b, 0);
            let min = f16_at(b, 2);
            let (sc, mn) = unpack_k4_scales(&b[4..16]);
            let q = &b[16..144];
            for j in 0..4 {
                let d1 = d * sc[2 * j] as f32;
                let m1 = min * mn[2 * j] as f32;
                let d2 = d * sc[2 * j + 1] as f32;
                let m2 = min * mn[2 * j + 1] as f32;
                for l in 0..32 {
                    y[64 * j + l] = d1 * (q[32 * j + l] & 0xF) as f32 - m1;
                    y[64 * j + 32 + l] = d2 * (q[32 * j + l] >> 4) as f32 - m2;
                }
            }
        }
        GgmlType::Q5_K => {
            let d = f16_at(b, 0);
            let min = f16_at(b, 2);
            let (sc, mn) = unpack_k4_scales(&b[4..16]);
            let qh = &b[16..48];
            let ql = &b[48..176];
            for j in 0..4 {
                let d1 = d * sc[2 * j] as f32;
                let m1 = min * mn[2 * j] as f32;
                let d2 = d * sc[2 * j + 1] as f32;
                let m2 = min * mn[2 * j + 1] as f32;
                let u1 = 1u8 << (2 * j);
                let u2 = 2u8 << (2 * j);
                for l in 0..32 {
                    let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    y[64 * j + l] = d1 * ((ql[32 * j + l] & 0xF) + h1) as f32 - m1;
                    y[64 * j + 32 + l] = d2 * ((ql[32 * j + l] >> 4) + h2) as f32 - m2;
                }
            }
        }
        GgmlType::Q6_K => {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            let d = f16_at(b, 208);
            for n in 0..2 {
                let ql = &ql[64 * n..];
                let qh = &qh[32 * n..];
                let sc = &sc[8 * n..];
                let y = &mut y[128 * n..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => unreachable!("not a block type"),
    }
}

// ---------------------------------------------------------------------------------------
// Scalar int8 dot products: weight row (raw block bytes) · quantized activation row.
// ---------------------------------------------------------------------------------------

pub fn vec_dot_q4_0(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(18).zip(a) {
        let qs = &b[2..18];
        let mut s = 0i32;
        for j in 0..16 {
            s += ((qs[j] & 0xF) as i32 - 8) * ab.qs[j] as i32;
            s += ((qs[j] >> 4) as i32 - 8) * ab.qs[j + 16] as i32;
        }
        sum += f16_at(b, 0) * ab.d * s as f32;
    }
    sum
}

pub fn vec_dot_q4_1(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(20).zip(a) {
        let qs = &b[4..20];
        let mut s = 0i32;
        for j in 0..16 {
            s += (qs[j] & 0xF) as i32 * ab.qs[j] as i32;
            s += (qs[j] >> 4) as i32 * ab.qs[j + 16] as i32;
        }
        sum += f16_at(b, 0) * ab.d * s as f32 + f16_at(b, 2) * ab.s;
    }
    sum
}

pub fn vec_dot_q5_0(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(22).zip(a) {
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        let qs = &b[6..22];
        let mut s = 0i32;
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            s += (((qs[j] as u32 & 0xF) | xh0) as i32 - 16) * ab.qs[j] as i32;
            s += (((qs[j] as u32 >> 4) | xh1) as i32 - 16) * ab.qs[j + 16] as i32;
        }
        sum += f16_at(b, 0) * ab.d * s as f32;
    }
    sum
}

pub fn vec_dot_q5_1(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(24).zip(a) {
        let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let qs = &b[8..24];
        let mut s = 0i32;
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            s += ((qs[j] as u32 & 0xF) | xh0) as i32 * ab.qs[j] as i32;
            s += ((qs[j] as u32 >> 4) | xh1) as i32 * ab.qs[j + 16] as i32;
        }
        sum += f16_at(b, 0) * ab.d * s as f32 + f16_at(b, 2) * ab.s;
    }
    sum
}

pub fn vec_dot_q8_0(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(34).zip(a) {
        let mut s = 0i32;
        for j in 0..QK {
            s += b[2 + j] as i8 as i32 * ab.qs[j] as i32;
        }
        sum += f16_at(b, 0) * ab.d * s as f32;
    }
    sum
}

pub fn vec_dot_q4_k(w: &[u8], a: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(144).zip(a) {
        let d = f16_at(b, 0) * ab.d;
        let dmin = f16_at(b, 2) * ab.d;
        let (sc, mn) = unpack_k4_scales(&b[4..16]);
        let q = &b[16..144];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        for j in 0..4 {
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                s1 += (q[32 * j + l] & 0xF) as i32 * ab.qs[64 * j + l] as i32;
                s2 += (q[32 * j + l] >> 4) as i32 * ab.qs[64 * j + 32 + l] as i32;
            }
            sumi += s1 * sc[2 * j] as i32 + s2 * sc[2 * j + 1] as i32;
        }
        for j in 0..8 {
            summ += (ab.bsums[2 * j] as i32 + ab.bsums[2 * j + 1] as i32) * mn[j] as i32;
        }
        sum += d * sumi as f32 - dmin * summ as f32;
    }
    sum
}

pub fn vec_dot_q5_k(w: &[u8], a: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(176).zip(a) {
        let d = f16_at(b, 0) * ab.d;
        let dmin = f16_at(b, 2) * ab.d;
        let (sc, mn) = unpack_k4_scales(&b[4..16]);
        let qh = &b[16..48];
        let ql = &b[48..176];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        for j in 0..4 {
            let u1 = 1u8 << (2 * j);
            let u2 = 2u8 << (2 * j);
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                s1 += ((ql[32 * j + l] & 0xF) as i32 + h1) * ab.qs[64 * j + l] as i32;
                s2 += ((ql[32 * j + l] >> 4) as i32 + h2) * ab.qs[64 * j + 32 + l] as i32;
            }
            sumi += s1 * sc[2 * j] as i32 + s2 * sc[2 * j + 1] as i32;
        }
        for j in 0..8 {
            summ += (ab.bsums[2 * j] as i32 + ab.bsums[2 * j + 1] as i32) * mn[j] as i32;
        }
        sum += d * sumi as f32 - dmin * summ as f32;
    }
    sum
}

pub fn vec_dot_q6_k(w: &[u8], a: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(210).zip(a) {
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        let d = f16_at(b, 208) * ab.d;
        let mut sumi = 0i32;
        for n in 0..2 {
            let ql = &ql[64 * n..];
            let qh = &qh[32 * n..];
            let sc = &sc[8 * n..];
            let aq = &ab.qs[128 * n..];
            // accumulate per 16-element group
            let mut g = [0i32; 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                g[is] += q1 * aq[l] as i32;
                g[is + 2] += q2 * aq[l + 32] as i32;
                g[is + 4] += q3 * aq[l + 64] as i32;
                g[is + 6] += q4 * aq[l + 96] as i32;
            }
            for k in 0..8 {
                sumi += g[k] * sc[k] as i8 as i32;
            }
        }
        sum += d * sumi as f32;
    }
    sum
}

/// Generic (slow) path for Q2_K/Q3_K: dequantize the weight block and dot in f32.
pub fn vec_dot_k_generic(ty: GgmlType, w: &[u8], a: &[BlockQ8K]) -> f32 {
    let bb = ty.block_bytes();
    let mut tmp = [0.0f32; QK_K];
    let mut sum = 0.0f32;
    for (b, ab) in w.chunks_exact(bb).zip(a) {
        dequantize_block(ty, b, &mut tmp);
        let mut s = 0.0f32;
        for j in 0..QK_K {
            s += tmp[j] * ab.qs[j] as f32;
        }
        sum += s * ab.d;
    }
    sum
}
