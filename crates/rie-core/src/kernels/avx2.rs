//! x86_64 AVX2/FMA/F16C kernels (Android x86_64 emulators, ChromeOS, development hosts).

#![allow(clippy::missing_safety_doc)]

use crate::quant::{reference::unpack_k4_scales, BlockQ8K, BlockQ8_0, QK};
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn _cvtsh_ss(bits: u16) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

#[inline(always)]
unsafe fn ld_f16(p: *const u8) -> f32 {
    _cvtsh_ss(u16::from_le_bytes([*p, *p.add(1)]))
}

#[inline(always)]
unsafe fn hsum_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[inline(always)]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

/// Signed int8 × int8 dot of 32 lanes, returned as 8 f32 partial sums.
#[inline(always)]
unsafe fn mul_sum_i8(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q8_0(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (i, ab) in a.iter().enumerate() {
        let b = w.as_ptr().add(i * 34);
        let x = _mm256_loadu_si256(b.add(2) as *const __m256i);
        let y = _mm256_loadu_si256(ab.qs.as_ptr() as *const __m256i);
        let d = _mm256_set1_ps(ld_f16(b) * ab.d);
        acc = _mm256_fmadd_ps(d, mul_sum_i8(x, y), acc);
    }
    hsum_ps(acc)
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4_0(w: &[u8], a: &[BlockQ8_0]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let m4 = _mm256_set1_epi8(0x0F);
    let off = _mm256_set1_epi8(8);
    for (i, ab) in a.iter().enumerate() {
        let b = w.as_ptr().add(i * 18);
        let q = _mm_loadu_si128(b.add(2) as *const __m128i);
        // low nibbles → elements 0..15, high nibbles → 16..31
        let both = _mm256_set_m128i(_mm_srli_epi16(q, 4), q);
        let x = _mm256_sub_epi8(_mm256_and_si256(both, m4), off);
        let y = _mm256_loadu_si256(ab.qs.as_ptr() as *const __m256i);
        let d = _mm256_set1_ps(ld_f16(b) * ab.d);
        acc = _mm256_fmadd_ps(d, mul_sum_i8(x, y), acc);
    }
    hsum_ps(acc)
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4_k(w: &[u8], a: &[BlockQ8K]) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let mut sum = 0.0f32;
    for (i, ab) in a.iter().enumerate() {
        let b = w.as_ptr().add(i * 144);
        let d = ld_f16(b) * ab.d;
        let dmin = ld_f16(b.add(2)) * ab.d;
        let (sc, mn) = unpack_k4_scales(std::slice::from_raw_parts(b.add(4), 12));
        let mut summ = 0i32;
        for j in 0..8 {
            summ += (ab.bsums[2 * j] as i32 + ab.bsums[2 * j + 1] as i32) * mn[j] as i32;
        }
        let q = b.add(16);
        let aq = ab.qs.as_ptr();
        let mut acc = _mm256_setzero_si256();
        for j in 0..4 {
            let v = _mm256_loadu_si256(q.add(32 * j) as *const __m256i);
            let lo = _mm256_and_si256(v, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
            let alo = _mm256_loadu_si256(aq.add(64 * j) as *const __m256i);
            let ahi = _mm256_loadu_si256(aq.add(64 * j + 32) as *const __m256i);
            let plo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, alo), _mm256_set1_epi16(sc[2 * j] as i16));
            let phi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, ahi), _mm256_set1_epi16(sc[2 * j + 1] as i16));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(plo, phi));
        }
        sum += d * hsum_epi32(acc) as f32 - dmin * summ as f32;
    }
    sum
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q6_k(w: &[u8], a: &[BlockQ8K]) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(3);
    let mut sum = 0.0f32;
    for (i, ab) in a.iter().enumerate() {
        let b = w.as_ptr().add(i * 210);
        let d = ld_f16(b.add(208)) * ab.d;
        let scp = b.add(192) as *const i8;
        let mut acc = _mm256_setzero_si256();
        let mut corr = 0i32;
        for n in 0..2 {
            let ql = b.add(64 * n);
            let qh = _mm256_loadu_si256(b.add(128 + 32 * n) as *const __m256i);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql1 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            let v = [
                _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4)),
                _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4)),
            ];
            let aq = ab.qs.as_ptr().add(128 * n);
            for (k, vk) in v.iter().enumerate() {
                let s0 = *scp.add(8 * n + 2 * k) as i16;
                let s1 = *scp.add(8 * n + 2 * k + 1) as i16;
                let y = _mm256_loadu_si256(aq.add(32 * k) as *const __m256i);
                let p = _mm256_maddubs_epi16(*vk, y);
                let scales = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, scales));
                corr += s0 as i32 * ab.bsums[8 * n + 2 * k] as i32 + s1 as i32 * ab.bsums[8 * n + 2 * k + 1] as i32;
            }
        }
        sum += d * (hsum_epi32(acc) - 32 * corr) as f32;
    }
    sum
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn quantize_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    let sign = _mm256_set1_ps(-0.0);
    for (bi, ob) in out.iter_mut().enumerate() {
        let p = x.as_ptr().add(bi * QK);
        let mut m = _mm256_setzero_ps();
        for k in 0..4 {
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(p.add(8 * k))));
        }
        let lo = _mm256_castps256_ps128(m);
        let hi = _mm256_extractf128_ps(m, 1);
        let s = _mm_max_ps(lo, hi);
        let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_max_ss(s, _mm_movehdup_ps(s));
        let amax = _mm_cvtss_f32(s);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let vid = _mm256_set1_ps(id);
        let mut isum = _mm256_setzero_si256();
        for k in 0..4 {
            let q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(8 * k)), vid));
            isum = _mm256_add_epi32(isum, q);
            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, q);
            for l in 0..8 {
                ob.qs[8 * k + l] = tmp[l].clamp(-127, 127) as i8;
            }
        }
        ob.d = d;
        ob.s = d * hsum_epi32(isum) as f32;
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), acc1);
        i += 16;
    }
    let mut s = hsum_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16_f32(a: &[u16], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let h0 = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let h1 = _mm256_cvtph_ps(_mm_loadu_si128(a.as_ptr().add(i + 8) as *const __m128i));
        acc0 = _mm256_fmadd_ps(h0, _mm256_loadu_ps(b.as_ptr().add(i)), acc0);
        acc1 = _mm256_fmadd_ps(h1, _mm256_loadu_ps(b.as_ptr().add(i + 8)), acc1);
        i += 16;
    }
    let mut s = hsum_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        s += _cvtsh_ss(a[i]) * b[i];
        i += 1;
    }
    s
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn axpy_f16(y: &mut [f32], alpha: f32, x: &[u16]) {
    let n = x.len();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i) as *const __m128i));
        let p = y.as_mut_ptr().add(i);
        _mm256_storeu_ps(p, _mm256_fmadd_ps(h, va, _mm256_loadu_ps(p)));
        i += 8;
    }
    while i < n {
        y[i] += alpha * _cvtsh_ss(x[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(src.as_ptr().add(i)));
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += 8;
    }
    while i < n {
        dst[i] = half::f16::from_f32(src[i]).to_bits();
        i += 1;
    }
}
