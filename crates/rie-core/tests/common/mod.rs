#![allow(dead_code)]
use half::f16;
use rie_core::quant::GgmlType;

/// Tiny deterministic PRNG for tests.
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
    /// Uniform in [-1, 1).
    pub fn sym(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.sym() * scale).collect()
    }
}

/// Byte offsets of the f16 scale fields inside one block of each type.
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

/// Random but valid weight data for `rows × cols` of `ty`, with magnitudes typical of
/// real checkpoints (values roughly within ±0.1).
pub fn random_weight(rng: &mut Rng, ty: GgmlType, rows: usize, cols: usize) -> Vec<u8> {
    match ty {
        GgmlType::F32 => rng.vec(rows * cols, 0.1).iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => rng.vec(rows * cols, 0.1).iter().flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
        GgmlType::BF16 => rng
            .vec(rows * cols, 0.1)
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        _ => {
            let n_blocks = rows * cols / ty.block_size();
            let bb = ty.block_bytes();
            let mut data: Vec<u8> = (0..n_blocks * bb).map(|_| rng.next_u64() as u8).collect();
            let scale = match ty {
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
                    if i == 0 && matches!(ty, GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q3_K | GgmlType::Q6_K) && rng.f32() < 0.3 {
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
        assert!(
            (x - y).abs() <= rtol * scale,
            "{what}: mismatch at {i}: {x} vs {y} (scale {scale}, rtol {rtol})"
        );
    }
}
