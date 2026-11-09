//! AArch64 NEON kernels.
//!
//! `SDOT`/`SMMLA` are emitted with inline assembly because their Rust intrinsics are still
//! unstable; each kernel is monomorphised twice — once with `SDOT` (dotprod) and once with a
//! `SMULL`/`SADALP` fallback for Armv8.0 cores such as Cortex-A53/A73 — and chosen at runtime.

#![allow(clippy::missing_safety_doc)]

use super::CpuFeatures;
use crate::quant::{reference::unpack_k4_scales, BlockQ8K, BlockQ8_0, QK};
use std::arch::aarch64::*;
use std::arch::asm;

/// Hardware half → single conversion (FCVT is base Armv8 FP, no FEAT_FP16 needed).
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let r: f32;
    unsafe {
        asm!(
            "fmov {t:s}, {b:w}",
            "fcvt {r:s}, {t:h}",
            b = in(reg) bits as u32,
            t = out(vreg) _,
            r = out(vreg) r,
            options(pure, nomem, nostack)
        );
    }
    r
}

#[inline(always)]
unsafe fn ld_f16(p: *const u8) -> f32 {
    fp16_to_f32(u16::from_le_bytes([*p, *p.add(1)]))
}

pub(crate) trait I8Dot {
    unsafe fn dot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t;
}

pub(crate) struct Sdot;
pub(crate) struct Mull;

impl I8Dot for Sdot {
    #[inline(always)]
    unsafe fn dot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut r = acc;
        asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b,
             options(pure, nomem, nostack));
        r
    }
}

impl I8Dot for Mull {
    #[inline(always)]
    unsafe fn dot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let p1 = vmull_high_s8(a, b);
        vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)))
    }
}

#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    asm!("smmla {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b,
         options(pure, nomem, nostack));
    r
}

macro_rules! dispatch {
    ($f:expr, $name:ident, $($arg:expr),*) => {{
        if $f.dotprod { paste_dotprod::$name($($arg),*) } else { paste_neon::$name($($arg),*) }
    }};
}

macro_rules! gen_variants {
    ($($name:ident ( $($a:ident : $t:ty),* ) -> $r:ty ;)*) => {
        mod paste_dotprod {
            use super::*;
            $(
                #[target_feature(enable = "neon,dotprod")]
                pub unsafe fn $name($($a: $t),*) -> $r { super::generic::$name::<Sdot>($($a),*) }
            )*
        }
        mod paste_neon {
            use super::*;
            $(
                #[target_feature(enable = "neon")]
                pub unsafe fn $name($($a: $t),*) -> $r { super::generic::$name::<Mull>($($a),*) }
            )*
        }
        $(
            #[inline]
            pub unsafe fn $name(f: &CpuFeatures, $($a: $t),*) -> $r { dispatch!(f, $name, $($a),*) }
        )*
    };
}

gen_variants! {
    dot_q8_0(w: &[u8], a: &[BlockQ8_0]) -> f32;
    dot_q4_0(w: &[u8], a: &[BlockQ8_0]) -> f32;
    dot_q4_1(w: &[u8], a: &[BlockQ8_0]) -> f32;
    dot_q5_0(w: &[u8], a: &[BlockQ8_0]) -> f32;
    dot_q5_1(w: &[u8], a: &[BlockQ8_0]) -> f32;
    dot_q4_k(w: &[u8], a: &[BlockQ8K]) -> f32;
    dot_q5_k(w: &[u8], a: &[BlockQ8K]) -> f32;
    dot_q6_k(w: &[u8], a: &[BlockQ8K]) -> f32;
}

mod generic {
    use super::*;

    #[inline(always)]
    unsafe fn hsum(v: int32x4_t) -> i32 {
        vaddvq_s32(v)
    }

    #[inline(always)]
    pub unsafe fn dot_q8_0<D: I8Dot>(w: &[u8], a: &[BlockQ8_0]) -> f32 {
        let nb = a.len();
        debug_assert!(w.len() >= nb * 34);
        let wp = w.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 1 < nb {
            let b0 = wp.add(i * 34);
            let b1 = wp.add((i + 1) * 34);
            let a0 = &a[i];
            let a1 = &a[i + 1];
            let p0 = D::dot(
                D::dot(vdupq_n_s32(0), vld1q_s8(b0.add(2) as *const i8), vld1q_s8(a0.qs.as_ptr())),
                vld1q_s8(b0.add(18) as *const i8),
                vld1q_s8(a0.qs.as_ptr().add(16)),
            );
            let p1 = D::dot(
                D::dot(vdupq_n_s32(0), vld1q_s8(b1.add(2) as *const i8), vld1q_s8(a1.qs.as_ptr())),
                vld1q_s8(b1.add(18) as *const i8),
                vld1q_s8(a1.qs.as_ptr().add(16)),
            );
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ld_f16(b0) * a0.d);
            acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), ld_f16(b1) * a1.d);
            i += 2;
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        if i < nb {
            let b0 = wp.add(i * 34);
            let a0 = &a[i];
            let p0 = D::dot(
                D::dot(vdupq_n_s32(0), vld1q_s8(b0.add(2) as *const i8), vld1q_s8(a0.qs.as_ptr())),
                vld1q_s8(b0.add(18) as *const i8),
                vld1q_s8(a0.qs.as_ptr().add(16)),
            );
            sum += hsum(p0) as f32 * ld_f16(b0) * a0.d;
        }
        sum
    }

    #[inline(always)]
    unsafe fn q4_unpack(p: *const u8, offset8: bool) -> (int8x16_t, int8x16_t) {
        let v = vld1q_u8(p);
        let lo = vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0F)));
        let hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
        if offset8 {
            (vsubq_s8(lo, vdupq_n_s8(8)), vsubq_s8(hi, vdupq_n_s8(8)))
        } else {
            (lo, hi)
        }
    }

    #[inline(always)]
    pub unsafe fn dot_q4_0<D: I8Dot>(w: &[u8], a: &[BlockQ8_0]) -> f32 {
        let nb = a.len();
        debug_assert!(w.len() >= nb * 18);
        let wp = w.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 1 < nb {
            let b0 = wp.add(i * 18);
            let b1 = wp.add((i + 1) * 18);
            let (l0, h0) = q4_unpack(b0.add(2), true);
            let (l1, h1) = q4_unpack(b1.add(2), true);
            let a0 = &a[i];
            let a1 = &a[i + 1];
            let p0 = D::dot(D::dot(vdupq_n_s32(0), l0, vld1q_s8(a0.qs.as_ptr())), h0, vld1q_s8(a0.qs.as_ptr().add(16)));
            let p1 = D::dot(D::dot(vdupq_n_s32(0), l1, vld1q_s8(a1.qs.as_ptr())), h1, vld1q_s8(a1.qs.as_ptr().add(16)));
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ld_f16(b0) * a0.d);
            acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), ld_f16(b1) * a1.d);
            i += 2;
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        if i < nb {
            let b0 = wp.add(i * 18);
            let (l0, h0) = q4_unpack(b0.add(2), true);
            let a0 = &a[i];
            let p0 = D::dot(D::dot(vdupq_n_s32(0), l0, vld1q_s8(a0.qs.as_ptr())), h0, vld1q_s8(a0.qs.as_ptr().add(16)));
            sum += hsum(p0) as f32 * ld_f16(b0) * a0.d;
        }
        sum
    }

    #[inline(always)]
    pub unsafe fn dot_q4_1<D: I8Dot>(w: &[u8], a: &[BlockQ8_0]) -> f32 {
        let wp = w.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        let mut summ = 0.0f32;
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 20);
            let (l, h) = q4_unpack(b.add(4), false);
            let p = D::dot(D::dot(vdupq_n_s32(0), l, vld1q_s8(ab.qs.as_ptr())), h, vld1q_s8(ab.qs.as_ptr().add(16)));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ld_f16(b) * ab.d);
            summ += ld_f16(b.add(2)) * ab.s;
        }
        vaddvq_f32(acc) + summ
    }

    /// Expand the 32 high bits of a Q5 block into two byte vectors holding 0x10 or 0.
    #[inline(always)]
    unsafe fn q5_high(qh: u32) -> (uint8x16_t, uint8x16_t) {
        const MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let m = vld1q_u8(MASK.as_ptr());
        let lo = vcombine_u8(vdup_n_u8(qh as u8), vdup_n_u8((qh >> 8) as u8));
        let hi = vcombine_u8(vdup_n_u8((qh >> 16) as u8), vdup_n_u8((qh >> 24) as u8));
        let ten = vdupq_n_u8(0x10);
        (vandq_u8(vtstq_u8(lo, m), ten), vandq_u8(vtstq_u8(hi, m), ten))
    }

    #[inline(always)]
    pub unsafe fn dot_q5_0<D: I8Dot>(w: &[u8], a: &[BlockQ8_0]) -> f32 {
        let wp = w.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 22);
            let qh = u32::from_le_bytes([*b.add(2), *b.add(3), *b.add(4), *b.add(5)]);
            let (h0, h1) = q5_high(qh);
            let v = vld1q_u8(b.add(6));
            let lo = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, vdupq_n_u8(0xF)), h0)), vdupq_n_s8(16));
            let hi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), h1)), vdupq_n_s8(16));
            let p = D::dot(D::dot(vdupq_n_s32(0), lo, vld1q_s8(ab.qs.as_ptr())), hi, vld1q_s8(ab.qs.as_ptr().add(16)));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ld_f16(b) * ab.d);
        }
        vaddvq_f32(acc)
    }

    #[inline(always)]
    pub unsafe fn dot_q5_1<D: I8Dot>(w: &[u8], a: &[BlockQ8_0]) -> f32 {
        let wp = w.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        let mut summ = 0.0f32;
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 24);
            let qh = u32::from_le_bytes([*b.add(4), *b.add(5), *b.add(6), *b.add(7)]);
            let (h0, h1) = q5_high(qh);
            let v = vld1q_u8(b.add(8));
            let lo = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v, vdupq_n_u8(0xF)), h0));
            let hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v, 4), h1));
            let p = D::dot(D::dot(vdupq_n_s32(0), lo, vld1q_s8(ab.qs.as_ptr())), hi, vld1q_s8(ab.qs.as_ptr().add(16)));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ld_f16(b) * ab.d);
            summ += ld_f16(b.add(2)) * ab.s;
        }
        vaddvq_f32(acc) + summ
    }

    #[inline(always)]
    unsafe fn mins_dot(bsums: &[i16; 16], mn: &[u8; 8]) -> i32 {
        // pairwise-add the 16 bsums into 8 sub-block sums and dot with the mins
        let b0 = vld1q_s16(bsums.as_ptr());
        let b1 = vld1q_s16(bsums.as_ptr().add(8));
        let s = vpaddq_s16(b0, b1);
        let m = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn.as_ptr())));
        let p = vaddq_s32(vmull_s16(vget_low_s16(s), vget_low_s16(m)), vmull_high_s16(s, m));
        vaddvq_s32(p)
    }

    #[inline(always)]
    pub unsafe fn dot_q4_k<D: I8Dot>(w: &[u8], a: &[BlockQ8K]) -> f32 {
        let wp = w.as_ptr();
        let m4 = vdupq_n_u8(0xF);
        let mut sum = 0.0f32;
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 144);
            let d = ld_f16(b) * ab.d;
            let dmin = ld_f16(b.add(2)) * ab.d;
            let (sc, mn) = unpack_k4_scales(std::slice::from_raw_parts(b.add(4), 12));
            let summ = mins_dot(&ab.bsums, &mn);
            let q = b.add(16);
            let aq = ab.qs.as_ptr();
            let mut sumi = 0i32;
            for j in 0..4 {
                let q0 = vld1q_u8(q.add(32 * j));
                let q1 = vld1q_u8(q.add(32 * j + 16));
                let a0 = vld1q_s8(aq.add(64 * j));
                let a1 = vld1q_s8(aq.add(64 * j + 16));
                let a2 = vld1q_s8(aq.add(64 * j + 32));
                let a3 = vld1q_s8(aq.add(64 * j + 48));
                let p1 = D::dot(
                    D::dot(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q0, m4)), a0),
                    vreinterpretq_s8_u8(vandq_u8(q1, m4)),
                    a1,
                );
                let p2 = D::dot(
                    D::dot(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), a2),
                    vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
                    a3,
                );
                sumi += hsum(p1) * sc[2 * j] as i32 + hsum(p2) * sc[2 * j + 1] as i32;
            }
            sum += d * sumi as f32 - dmin * summ as f32;
        }
        sum
    }

    #[inline(always)]
    pub unsafe fn dot_q5_k<D: I8Dot>(w: &[u8], a: &[BlockQ8K]) -> f32 {
        let wp = w.as_ptr();
        let m4 = vdupq_n_u8(0xF);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let mut sum = 0.0f32;
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 176);
            let d = ld_f16(b) * ab.d;
            let dmin = ld_f16(b.add(2)) * ab.d;
            let (sc, mn) = unpack_k4_scales(std::slice::from_raw_parts(b.add(4), 12));
            let summ = mins_dot(&ab.bsums, &mn);
            let mut qh0 = vld1q_u8(b.add(16));
            let mut qh1 = vld1q_u8(b.add(32));
            let ql = b.add(48);
            let aq = ab.qs.as_ptr();
            let mut sumi = 0i32;
            for j in 0..4 {
                let h0 = vshlq_n_u8(vandq_u8(mone, qh0), 4);
                let h1 = vshlq_n_u8(vandq_u8(mone, qh1), 4);
                let h2 = vshlq_n_u8(vandq_u8(mtwo, qh0), 3);
                let h3 = vshlq_n_u8(vandq_u8(mtwo, qh1), 3);
                qh0 = vshrq_n_u8(qh0, 2);
                qh1 = vshrq_n_u8(qh1, 2);
                let q0 = vld1q_u8(ql.add(32 * j));
                let q1 = vld1q_u8(ql.add(32 * j + 16));
                let v0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4), h0));
                let v1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4), h1));
                let v2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q0, 4), h2));
                let v3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q1, 4), h3));
                let p1 = D::dot(D::dot(vdupq_n_s32(0), v0, vld1q_s8(aq.add(64 * j))), v1, vld1q_s8(aq.add(64 * j + 16)));
                let p2 = D::dot(D::dot(vdupq_n_s32(0), v2, vld1q_s8(aq.add(64 * j + 32))), v3, vld1q_s8(aq.add(64 * j + 48)));
                sumi += hsum(p1) * sc[2 * j] as i32 + hsum(p2) * sc[2 * j + 1] as i32;
            }
            sum += d * sumi as f32 - dmin * summ as f32;
        }
        sum
    }

    #[inline(always)]
    pub unsafe fn dot_q6_k<D: I8Dot>(w: &[u8], a: &[BlockQ8K]) -> f32 {
        let wp = w.as_ptr();
        let m4 = vdupq_n_u8(0xF);
        let m3 = vdupq_n_u8(3);
        let s32 = vdupq_n_s8(32);
        let mut sum = 0.0f32;
        for (i, ab) in a.iter().enumerate() {
            let b = wp.add(i * 210);
            let d = ld_f16(b.add(208)) * ab.d;
            let sc = b.add(192) as *const i8;
            let mut sumi = 0i32;
            for n in 0..2 {
                let ql = b.add(64 * n);
                let qh = b.add(128 + 32 * n);
                let aq = ab.qs.as_ptr().add(128 * n);
                let sc = sc.add(8 * n);
                let qh0 = vld1q_u8(qh);
                let qh1 = vld1q_u8(qh.add(16));
                let ql0 = vld1q_u8(ql);
                let ql1 = vld1q_u8(ql.add(16));
                let ql2 = vld1q_u8(ql.add(32));
                let ql3 = vld1q_u8(ql.add(48));
                let mk = |lo: uint8x16_t, hi: uint8x16_t| vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo, vshlq_n_u8(hi, 4))), s32);
                let v = [
                    mk(vandq_u8(ql0, m4), vandq_u8(qh0, m3)),
                    mk(vandq_u8(ql1, m4), vandq_u8(qh1, m3)),
                    mk(vandq_u8(ql2, m4), vandq_u8(vshrq_n_u8(qh0, 2), m3)),
                    mk(vandq_u8(ql3, m4), vandq_u8(vshrq_n_u8(qh1, 2), m3)),
                    mk(vshrq_n_u8(ql0, 4), vandq_u8(vshrq_n_u8(qh0, 4), m3)),
                    mk(vshrq_n_u8(ql1, 4), vandq_u8(vshrq_n_u8(qh1, 4), m3)),
                    mk(vshrq_n_u8(ql2, 4), vshrq_n_u8(qh0, 6)),
                    mk(vshrq_n_u8(ql3, 4), vshrq_n_u8(qh1, 6)),
                ];
                for (k, vk) in v.iter().enumerate() {
                    let p = D::dot(vdupq_n_s32(0), *vk, vld1q_s8(aq.add(16 * k)));
                    sumi += hsum(p) * (*sc.add(k)) as i32;
                }
            }
            sum += d * sumi as f32;
        }
        sum
    }
}

// ---------------------------------------------------------------------------------------
// i8mm 2x2 tiles (prompt processing).
// ---------------------------------------------------------------------------------------

#[inline(always)]
unsafe fn zip_lo(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
}
#[inline(always)]
unsafe fn zip_hi(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
}

#[inline(always)]
unsafe fn mmla_block(x0a: int8x16_t, x0b: int8x16_t, x1a: int8x16_t, x1b: int8x16_t, a0: &BlockQ8_0, a1: &BlockQ8_0) -> int32x4_t {
    let y0a = vld1q_s8(a0.qs.as_ptr());
    let y0b = vld1q_s8(a0.qs.as_ptr().add(16));
    let y1a = vld1q_s8(a1.qs.as_ptr());
    let y1b = vld1q_s8(a1.qs.as_ptr().add(16));
    let mut s = vdupq_n_s32(0);
    s = smmla(s, zip_lo(x0a, x1a), zip_lo(y0a, y1a));
    s = smmla(s, zip_hi(x0a, x1a), zip_hi(y0a, y1a));
    s = smmla(s, zip_lo(x0b, x1b), zip_lo(y0b, y1b));
    s = smmla(s, zip_hi(x0b, x1b), zip_hi(y0b, y1b));
    s
}

#[target_feature(enable = "neon,i8mm")]
pub unsafe fn mmla_q8_0(w0: &[u8], w1: &[u8], a0: &[BlockQ8_0], a1: &[BlockQ8_0]) -> [f32; 4] {
    let nb = a0.len();
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..nb {
        let p0 = w0.as_ptr().add(i * 34);
        let p1 = w1.as_ptr().add(i * 34);
        let s = mmla_block(
            vld1q_s8(p0.add(2) as *const i8),
            vld1q_s8(p0.add(18) as *const i8),
            vld1q_s8(p1.add(2) as *const i8),
            vld1q_s8(p1.add(18) as *const i8),
            &a0[i],
            &a1[i],
        );
        let d0 = ld_f16(p0);
        let d1 = ld_f16(p1);
        let scale = [d0 * a0[i].d, d0 * a1[i].d, d1 * a0[i].d, d1 * a1[i].d];
        acc = vmlaq_f32(acc, vcvtq_f32_s32(s), vld1q_f32(scale.as_ptr()));
    }
    let mut out = [0.0f32; 4];
    vst1q_f32(out.as_mut_ptr(), acc);
    out
}

#[target_feature(enable = "neon,i8mm")]
pub unsafe fn mmla_q4_0(w0: &[u8], w1: &[u8], a0: &[BlockQ8_0], a1: &[BlockQ8_0]) -> [f32; 4] {
    let nb = a0.len();
    let m4 = vdupq_n_u8(0xF);
    let s8 = vdupq_n_s8(8);
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..nb {
        let p0 = w0.as_ptr().add(i * 18);
        let p1 = w1.as_ptr().add(i * 18);
        let v0 = vld1q_u8(p0.add(2));
        let v1 = vld1q_u8(p1.add(2));
        let s = mmla_block(
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8),
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8),
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4)), s8),
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), s8),
            &a0[i],
            &a1[i],
        );
        let d0 = ld_f16(p0);
        let d1 = ld_f16(p1);
        let scale = [d0 * a0[i].d, d0 * a1[i].d, d1 * a0[i].d, d1 * a1[i].d];
        acc = vmlaq_f32(acc, vcvtq_f32_s32(s), vld1q_f32(scale.as_ptr()));
    }
    let mut out = [0.0f32; 4];
    vst1q_f32(out.as_mut_ptr(), acc);
    out
}

// ---------------------------------------------------------------------------------------
// Activation quantization and float kernels.
// ---------------------------------------------------------------------------------------

#[target_feature(enable = "neon")]
pub unsafe fn quantize_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    for (bi, ob) in out.iter_mut().enumerate() {
        let p = x.as_ptr().add(bi * QK);
        let mut v = [vdupq_n_f32(0.0); 8];
        for k in 0..8 {
            v[k] = vld1q_f32(p.add(4 * k));
        }
        let mut amax = vabsq_f32(v[0]);
        for k in 1..8 {
            amax = vmaxq_f32(amax, vabsq_f32(v[k]));
        }
        let amax = vmaxvq_f32(amax);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut isum = vdupq_n_s32(0);
        for k in 0..8 {
            let q = vcvtnq_s32_f32(vmulq_n_f32(v[k], id));
            isum = vaddq_s32(isum, q);
            let q16 = vqmovn_s32(q);
            let q8 = vqmovn_s16(vcombine_s16(q16, q16));
            let lane = vget_lane_s32::<0>(vreinterpret_s32_s8(q8));
            std::ptr::write_unaligned(ob.qs.as_mut_ptr().add(4 * k) as *mut i32, lane);
        }
        ob.d = d;
        ob.s = d * vaddvq_s32(isum) as f32;
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 16 <= n {
        for k in 0..4 {
            acc[k] = vfmaq_f32(acc[k], vld1q_f32(pa.add(i + 4 * k)), vld1q_f32(pb.add(i + 4 * k)));
        }
        i += 16;
    }
    while i + 4 <= n {
        acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }
    let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Convert 8 f16 values to two f32x4 (FCVTL/FCVTL2: base Armv8 AdvSIMD).
#[inline(always)]
unsafe fn cvt8(p: *const u16) -> (float32x4_t, float32x4_t) {
    let h = vld1q_u16(p);
    let lo: float32x4_t;
    let hi: float32x4_t;
    asm!("fcvtl {lo:v}.4s, {h:v}.4h", "fcvtl2 {hi:v}.4s, {h:v}.8h",
         h = in(vreg) h, lo = out(vreg) lo, hi = out(vreg) hi, options(pure, nomem, nostack));
    (lo, hi)
}

#[target_feature(enable = "neon")]
pub unsafe fn dot_f16_f32(a: &[u16], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 16 <= n {
        let (l0, h0) = cvt8(pa.add(i));
        let (l1, h1) = cvt8(pa.add(i + 8));
        acc0 = vfmaq_f32(acc0, l0, vld1q_f32(pb.add(i)));
        acc1 = vfmaq_f32(acc1, h0, vld1q_f32(pb.add(i + 4)));
        acc2 = vfmaq_f32(acc2, l1, vld1q_f32(pb.add(i + 8)));
        acc3 = vfmaq_f32(acc3, h1, vld1q_f32(pb.add(i + 12)));
        i += 16;
    }
    while i + 8 <= n {
        let (l0, h0) = cvt8(pa.add(i));
        acc0 = vfmaq_f32(acc0, l0, vld1q_f32(pb.add(i)));
        acc1 = vfmaq_f32(acc1, h0, vld1q_f32(pb.add(i + 4)));
        i += 8;
    }
    let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        s += fp16_to_f32(a[i]) * b[i];
        i += 1;
    }
    s
}

#[target_feature(enable = "neon")]
pub unsafe fn axpy_f16(y: &mut [f32], alpha: f32, x: &[u16]) {
    let n = x.len();
    let (px, py) = (x.as_ptr(), y.as_mut_ptr());
    let va = vdupq_n_f32(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let (l, h) = cvt8(px.add(i));
        vst1q_f32(py.add(i), vfmaq_f32(vld1q_f32(py.add(i)), l, va));
        vst1q_f32(py.add(i + 4), vfmaq_f32(vld1q_f32(py.add(i + 4)), h, va));
        i += 8;
    }
    while i < n {
        y[i] += alpha * fp16_to_f32(x[i]);
        i += 1;
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let lo = vld1q_f32(src.as_ptr().add(i));
        let hi = vld1q_f32(src.as_ptr().add(i + 4));
        let out: uint16x8_t;
        asm!("fcvtn {o:v}.4h, {lo:v}.4s", "fcvtn2 {o:v}.8h, {hi:v}.4s",
             lo = in(vreg) lo, hi = in(vreg) hi, o = out(vreg) out, options(pure, nomem, nostack));
        vst1q_u16(dst.as_mut_ptr().add(i), out);
        i += 8;
    }
    while i < n {
        dst[i] = half::f16::from_f32(src[i]).to_bits();
        i += 1;
    }
}
