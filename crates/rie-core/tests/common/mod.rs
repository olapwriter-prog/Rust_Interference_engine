#![allow(dead_code, unused_imports)]
pub use rie_core::quant::GgmlType;
pub use rie_core::testing::{scale_offsets, Rng};

pub fn random_weight(rng: &mut Rng, ty: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    rie_core::testing::random_weight(rng, ty, rows, cols, 1.0)
}

pub const ALL_TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
];

pub fn assert_close(a: &[f32], b: &[f32], rtol: f32, what: &str) {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= rtol * scale, "{what}: mismatch at {i}: {x} vs {y} (scale {scale}, rtol {rtol})");
    }
}

pub fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}
