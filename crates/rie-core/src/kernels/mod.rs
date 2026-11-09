//! Compute kernels with runtime CPU feature dispatch.
//!
//! * aarch64 (all current phones): NEON baseline, `SDOT` (Armv8.2 dotprod — Cortex-A55/A75
//!   and newer, every Apple A-series since A11) and `SMMLA` (Armv8.6 i8mm — Cortex-X2/A710
//!   and newer, Snapdragon 8 Gen 1+, Dimensity 9000+, Tensor G2+, Apple A15+) are detected
//!   at runtime so a single binary runs optimally on old and new devices.
//! * x86_64 (emulators, ChromeOS, development): AVX2+FMA+F16C.
//! * Everything else: portable scalar code.

pub mod matmul;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

use crate::quant::{reference, BlockQ8K, BlockQ8_0, GgmlType, QK, QK_K};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub fp16: bool,
    pub sve: bool,
    pub sme: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        let mut f = CpuFeatures::default();
        #[cfg(target_arch = "aarch64")]
        {
            f.neon = std::arch::is_aarch64_feature_detected!("neon");
            f.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
            f.i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
            f.fp16 = std::arch::is_aarch64_feature_detected!("fp16");
            f.sve = std::arch::is_aarch64_feature_detected!("sve");
            f.sme = false; // detection not yet stable in Rust
        }
        #[cfg(target_arch = "x86_64")]
        {
            f.avx2 = is_x86_feature_detected!("avx2");
            f.fma = is_x86_feature_detected!("fma");
            f.f16c = is_x86_feature_detected!("f16c");
        }
        // Allow forcing slower paths for testing / A-B benchmarking.
        if let Ok(v) = std::env::var("RIE_DISABLE") {
            for flag in v.split(',') {
                match flag.trim() {
                    "dotprod" => f.dotprod = false,
                    "i8mm" => f.i8mm = false,
                    "neon" => {
                        f.neon = false;
                        f.dotprod = false;
                        f.i8mm = false;
                    }
                    "avx2" => f.avx2 = false,
                    "simd" => {
                        f.neon = false;
                        f.dotprod = false;
                        f.i8mm = false;
                        f.avx2 = false;
                    }
                    _ => {}
                }
            }
        }
        f
    }

    #[inline]
    pub fn get() -> &'static CpuFeatures {
        static F: OnceLock<CpuFeatures> = OnceLock::new();
        F.get_or_init(CpuFeatures::detect)
    }

    #[inline]
    pub(crate) fn x86_simd(&self) -> bool {
        self.avx2 && self.fma && self.f16c
    }

    pub fn describe(&self) -> String {
        let mut v = Vec::new();
        for (on, name) in [
            (self.neon, "NEON"),
            (self.dotprod, "DOTPROD"),
            (self.i8mm, "I8MM"),
            (self.fp16, "FP16"),
            (self.sve, "SVE"),
            (self.sme, "SME"),
            (self.avx2, "AVX2"),
            (self.fma, "FMA"),
            (self.f16c, "F16C"),
        ] {
            if on {
                v.push(name);
            }
        }
        if v.is_empty() {
            "scalar".into()
        } else {
            v.join(" ")
        }
    }

    /// Name of the int8 kernel family that will be used.
    pub fn kernel_name(&self) -> &'static str {
        if self.i8mm && self.dotprod {
            "neon-i8mm"
        } else if self.dotprod {
            "neon-dotprod"
        } else if self.neon {
            "neon"
        } else if self.x86_simd() {
            "avx2"
        } else {
            "scalar"
        }
    }
}

/// Quantized activation row in whichever format the weight type needs.
#[derive(Clone, Copy)]
pub enum ActRow<'a> {
    F32(&'a [f32]),
    Q8_0(&'a [BlockQ8_0]),
    Q8K(&'a [BlockQ8K]),
}

/// Dot product of one weight row (`n` elements, raw ggml blocks) with an activation row.
#[inline]
pub fn vec_dot(ty: GgmlType, w: &[u8], a: ActRow<'_>, n: usize) -> f32 {
    let f = CpuFeatures::get();
    match (ty, a) {
        (GgmlType::F32, ActRow::F32(x)) => dot_f32_bytes(w, x, n),
        (GgmlType::F16, ActRow::F32(x)) => dot_f16_f32(as_u16(w, n), &x[..n]),
        (GgmlType::BF16, ActRow::F32(x)) => {
            let w = as_u16(w, n);
            let mut s = 0.0f32;
            for i in 0..n {
                s += f32::from_bits((w[i] as u32) << 16) * x[i];
            }
            s
        }
        (GgmlType::Q8_0, ActRow::Q8_0(a)) => {
            let a = &a[..n / QK];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q8_0(f, w, a) };
            }
            #[cfg(target_arch = "x86_64")]
            if f.x86_simd() {
                return unsafe { avx2::dot_q8_0(w, a) };
            }
            reference::vec_dot_q8_0(w, a)
        }
        (GgmlType::Q4_0, ActRow::Q8_0(a)) => {
            let a = &a[..n / QK];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q4_0(f, w, a) };
            }
            #[cfg(target_arch = "x86_64")]
            if f.x86_simd() {
                return unsafe { avx2::dot_q4_0(w, a) };
            }
            reference::vec_dot_q4_0(w, a)
        }
        (GgmlType::Q4_1, ActRow::Q8_0(a)) => {
            let a = &a[..n / QK];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q4_1(f, w, a) };
            }
            reference::vec_dot_q4_1(w, a)
        }
        (GgmlType::Q5_0, ActRow::Q8_0(a)) => {
            let a = &a[..n / QK];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q5_0(f, w, a) };
            }
            reference::vec_dot_q5_0(w, a)
        }
        (GgmlType::Q5_1, ActRow::Q8_0(a)) => {
            let a = &a[..n / QK];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q5_1(f, w, a) };
            }
            reference::vec_dot_q5_1(w, a)
        }
        (GgmlType::Q4_K, ActRow::Q8K(a)) => {
            let a = &a[..n / QK_K];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q4_k(f, w, a) };
            }
            #[cfg(target_arch = "x86_64")]
            if f.x86_simd() {
                return unsafe { avx2::dot_q4_k(w, a) };
            }
            reference::vec_dot_q4_k(w, a)
        }
        (GgmlType::Q5_K, ActRow::Q8K(a)) => {
            let a = &a[..n / QK_K];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q5_k(f, w, a) };
            }
            reference::vec_dot_q5_k(w, a)
        }
        (GgmlType::Q6_K, ActRow::Q8K(a)) => {
            let a = &a[..n / QK_K];
            #[cfg(target_arch = "aarch64")]
            if f.neon {
                return unsafe { neon::dot_q6_k(f, w, a) };
            }
            #[cfg(target_arch = "x86_64")]
            if f.x86_simd() {
                return unsafe { avx2::dot_q6_k(w, a) };
            }
            reference::vec_dot_q6_k(w, a)
        }
        (GgmlType::Q2_K | GgmlType::Q3_K, ActRow::Q8K(a)) => reference::vec_dot_k_generic(ty, w, &a[..n / QK_K]),
        _ => panic!("vec_dot: activation format does not match weight type {ty:?}"),
    }
}

/// 2x2 tile: two weight rows × two activation rows using `SMMLA`, returning
/// `[w0·a0, w0·a1, w1·a0, w1·a1]`. Only for Q4_0 / Q8_0 on i8mm hardware.
#[inline]
#[allow(unused_variables)]
pub(crate) fn mmla_2x2(ty: GgmlType, w0: &[u8], w1: &[u8], a0: &[BlockQ8_0], a1: &[BlockQ8_0]) -> Option<[f32; 4]> {
    #[cfg(target_arch = "aarch64")]
    {
        let f = CpuFeatures::get();
        if f.i8mm {
            return match ty {
                GgmlType::Q8_0 => Some(unsafe { neon::mmla_q8_0(w0, w1, a0, a1) }),
                GgmlType::Q4_0 => Some(unsafe { neon::mmla_q4_0(w0, w1, a0, a1) }),
                _ => None,
            };
        }
    }
    None
}

#[inline]
pub(crate) fn has_mmla(ty: GgmlType) -> bool {
    CpuFeatures::get().i8mm && matches!(ty, GgmlType::Q8_0 | GgmlType::Q4_0) && cfg!(target_arch = "aarch64")
}

#[inline]
fn as_u16(w: &[u8], n: usize) -> &[u16] {
    assert!(w.len() >= n * 2);
    // f16 data in GGUF is 32-byte aligned in the file and rows are even-sized: alignment holds.
    let (pre, mid, _) = unsafe { w[..n * 2].align_to::<u16>() };
    assert!(pre.is_empty(), "unaligned f16 tensor data");
    mid
}

#[inline]
fn dot_f32_bytes(w: &[u8], x: &[f32], n: usize) -> f32 {
    let (pre, mid, _) = unsafe { w[..n * 4].align_to::<f32>() };
    if pre.is_empty() {
        dot_f32(mid, &x[..n])
    } else {
        let mut s = 0.0;
        for i in 0..n {
            s += f32::from_le_bytes(w[i * 4..i * 4 + 4].try_into().unwrap()) * x[i];
        }
        s
    }
}

pub fn quantize_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    #[cfg(target_arch = "aarch64")]
    if CpuFeatures::get().neon {
        return unsafe { neon::quantize_q8_0(x, out) };
    }
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().x86_simd() {
        return unsafe { avx2::quantize_q8_0(x, out) };
    }
    crate::quant::quantize_row_q8_0_scalar(x, out)
}

// ---------------------------------------------------------------------------------------
// Float kernels used by attention, norms and the KV cache.
// ---------------------------------------------------------------------------------------

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if CpuFeatures::get().neon {
        return unsafe { neon::dot_f32(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().x86_simd() {
        return unsafe { avx2::dot_f32(a, b) };
    }
    // 4 independent accumulators let the compiler vectorise.
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        for k in 0..8 {
            acc[k] += a[i * 8 + k] * b[i * 8 + k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Dot of an f16 vector (raw bits) with an f32 vector.
#[inline]
pub fn dot_f16_f32(a: &[u16], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if CpuFeatures::get().neon {
        return unsafe { neon::dot_f16_f32(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().x86_simd() {
        return unsafe { avx2::dot_f16_f32(a, b) };
    }
    a.iter().zip(b).map(|(&h, &x)| half::f16::from_bits(h).to_f32() * x).sum()
}

/// `y += alpha * x` with `x` stored as f16.
#[inline]
pub fn axpy_f16(y: &mut [f32], alpha: f32, x: &[u16]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "aarch64")]
    if CpuFeatures::get().neon {
        return unsafe { neon::axpy_f16(y, alpha, x) };
    }
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().x86_simd() {
        return unsafe { avx2::axpy_f16(y, alpha, x) };
    }
    for (o, &h) in y.iter_mut().zip(x) {
        *o += alpha * half::f16::from_bits(h).to_f32();
    }
}

/// Convert f32 → f16 bits (round to nearest even).
#[inline]
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    if CpuFeatures::get().neon {
        return unsafe { neon::f32_to_f16(src, dst) };
    }
    #[cfg(target_arch = "x86_64")]
    if CpuFeatures::get().x86_simd() {
        return unsafe { avx2::f32_to_f16(src, dst) };
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_f32(s).to_bits();
    }
}

/// Scalar f16 → f32 using the hardware converter where available.
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::fp16_to_f32(bits)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        half::f16::from_bits(bits).to_f32()
    }
}
