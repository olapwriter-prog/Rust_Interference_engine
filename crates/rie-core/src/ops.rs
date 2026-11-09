//! Element-wise and normalisation ops (f32).

use crate::kernels::dot_f32;

/// `out = x / rms(x) * w`
#[inline]
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss = dot_f32(x, x) / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

#[inline]
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss = dot_f32(x, x) / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] *= scale * w[i];
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)).tanh())
}

/// In-place numerically stable softmax.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotary position embedding parameters for one family of layers.
#[derive(Debug, Clone)]
pub struct Rope {
    /// Number of rotated dimensions (≤ head_dim).
    pub dim: usize,
    /// NEOX style rotates (x[i], x[i + dim/2]); normal (GPT-J) style rotates (x[2i], x[2i+1]).
    pub neox: bool,
    /// Per pair inverse frequency, already including frequency factors and linear scaling.
    pub inv_freq: Vec<f32>,
    /// Magnitude scaling applied to cos/sin (YaRN attention factor; 1.0 otherwise).
    pub mscale: f32,
}

impl Rope {
    pub fn new(dim: usize, neox: bool, base: f32, freq_scale: f32, factors: Option<&[f32]>) -> Self {
        let half = dim / 2;
        let inv_freq = (0..half)
            .map(|i| {
                let mut f = base.powf(-2.0 * i as f32 / dim as f32) * freq_scale;
                if let Some(fac) = factors {
                    if i < fac.len() && fac[i] != 0.0 {
                        f /= fac[i];
                    }
                }
                f
            })
            .collect();
        Rope { dim, neox, inv_freq, mscale: 1.0 }
    }

    /// Fill `cos`/`sin` (length dim/2) for position `pos`.
    pub fn table(&self, pos: usize, cos: &mut [f32], sin: &mut [f32]) {
        for (i, f) in self.inv_freq.iter().enumerate() {
            let theta = pos as f32 * f;
            let (s, c) = theta.sin_cos();
            cos[i] = c * self.mscale;
            sin[i] = s * self.mscale;
        }
    }

    /// Rotate one head vector in place using a precomputed table.
    #[inline]
    pub fn apply(&self, x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = self.dim / 2;
        if self.neox {
            for i in 0..half {
                let (a, b) = (x[i], x[i + half]);
                x[i] = a * cos[i] - b * sin[i];
                x[i + half] = a * sin[i] + b * cos[i];
            }
        } else {
            for i in 0..half {
                let (a, b) = (x[2 * i], x[2 * i + 1]);
                x[2 * i] = a * cos[i] - b * sin[i];
                x[2 * i + 1] = a * sin[i] + b * cos[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_preserves_norm_and_matches_complex_rotation() {
        let r = Rope::new(8, false, 10000.0, 1.0, None);
        let mut cos = vec![0.0; 4];
        let mut sin = vec![0.0; 4];
        r.table(5, &mut cos, &mut sin);
        let x0: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut x = x0.clone();
        r.apply(&mut x, &cos, &sin);
        let n0: f32 = x0.iter().map(|v| v * v).sum();
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
        // pair 0 has frequency 1: rotation by 5 radians
        let (c, s) = (5.0f32.cos(), 5.0f32.sin());
        assert!((x[0] - (x0[0] * c - x0[1] * s)).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -1000.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[3] == 0.0);
    }
}
