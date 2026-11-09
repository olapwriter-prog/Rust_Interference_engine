//! On-device reinforcement learning for quick corrections.
//!
//! The base model is never modified. A small **mixture of correction experts** sits on
//! top of the output head and adds a learned bias to the logits:
//!
//! ```text
//! ĥ        = h / ‖h‖                         final hidden state (output-head input)
//! g        = softmax(W_g ĥ)                   router over K experts
//! Δlogits  = Σ_k g_k · A_k (B_k ĥ)            each expert is a rank-r map
//! ```
//!
//! Different kinds of request land in different regions of hidden-state space, so the
//! router lets, say, finance tool calls and browser tool calls learn different
//! corrections. Training is REINFORCE on the experts only: for a generated token `a`
//! with reward `R`, the logit gradient is `R·(onehot(a) − p)`, restricted to the
//! top-M candidates recorded at sampling time. Every update is closed-form (no autograd)
//! and touches only the vocabulary rows involved, so it costs milliseconds on a phone.
//!
//! `A` starts at zero (no effect until feedback arrives) and is stored sparsely by
//! vocabulary row. Each expert's contribution is clamped to ±`max_shift` logits so
//! corrections cannot overwhelm the base model.

use crate::error::{bail, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Number of correction experts.
    pub n_experts: usize,
    /// Rank of each expert.
    pub rank: usize,
    /// Learning rate for policy-gradient updates.
    pub lr: f32,
    /// Multiplicative decay applied to touched rows on each update.
    pub weight_decay: f32,
    /// Maximum absolute change the adapter may apply to any logit.
    pub max_shift: f32,
    /// Candidates recorded per generated token (the support of the policy gradient).
    pub top_m: usize,
    pub seed: u64,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        AdapterConfig { n_experts: 4, rank: 8, lr: 0.5, weight_decay: 1e-3, max_shift: 8.0, top_m: 32, seed: 0x5EED_CAFE }
    }
}

/// One generated (or teacher-forced) token: what the policy saw and what it did.
#[derive(Debug, Clone)]
pub struct Step {
    /// Final hidden state as f16 bits.
    pub h: Vec<u16>,
    /// Top-M (token, base logit) before the adapter.
    pub top: Vec<(u32, f32)>,
    pub token: u32,
}

impl Step {
    pub fn new(h: &[f32], base_logits: &[f32], token: u32, top_m: usize) -> Self {
        let mut idx: Vec<u32> = (0..base_logits.len() as u32).collect();
        let m = top_m.min(idx.len());
        if m < idx.len() {
            idx.select_nth_unstable_by(m - 1, |&a, &b| {
                base_logits[b as usize].partial_cmp(&base_logits[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(m);
        }
        let mut top: Vec<(u32, f32)> = idx.into_iter().map(|i| (i, base_logits[i as usize])).collect();
        if !top.iter().any(|&(i, _)| i == token) {
            top.push((token, base_logits[token as usize]));
        }
        Step { h: h.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect(), top, token }
    }
}

/// Statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct AdapterStats {
    pub updates: u64,
    pub rows: usize,
    pub reward_sum: f64,
    pub reward_count: u64,
}

pub struct CorrectionExperts {
    pub cfg: AdapterConfig,
    n_embd: usize,
    n_vocab: usize,
    /// per vocabulary row: K × r weights
    a: HashMap<u32, Box<[f32]>>,
    /// K × r × E
    b: Vec<f32>,
    /// K × E router
    wg: Vec<f32>,
    pub stats: AdapterStats,
}

struct Rng(u64);
impl Rng {
    fn normal(&mut self) -> f32 {
        // Box–Muller from xorshift
        let mut u = || {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
        };
        let (a, b) = (u(), u());
        ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
    }
}

impl CorrectionExperts {
    pub fn new(n_embd: usize, n_vocab: usize, cfg: AdapterConfig) -> Self {
        let (k, r) = (cfg.n_experts.max(1), cfg.rank.max(1));
        let mut rng = Rng(cfg.seed | 1);
        // B ~ N(0, 1) per entry so that B·ĥ (‖ĥ‖ = 1) has unit-variance components;
        // the router starts with small random weights to break the symmetry between experts.
        let b = (0..k * r * n_embd).map(|_| rng.normal()).collect();
        let wg = (0..k * n_embd).map(|_| rng.normal() * 1.0).collect();
        CorrectionExperts {
            cfg: AdapterConfig { n_experts: k, rank: r, ..cfg },
            n_embd,
            n_vocab,
            a: HashMap::new(),
            b,
            wg,
            stats: AdapterStats::default(),
        }
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    fn unit(h: &[f32]) -> Vec<f32> {
        let n = h.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        h.iter().map(|v| v / n).collect()
    }

    /// Router weights `g` and expert projections `u_k = B_k ĥ`.
    fn route(&self, hu: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (k, r, e) = (self.cfg.n_experts, self.cfg.rank, self.n_embd);
        let mut g: Vec<f32> = (0..k).map(|i| crate::kernels::dot_f32(&self.wg[i * e..(i + 1) * e], hu)).collect();
        crate::ops::softmax(&mut g);
        let mut u = vec![0.0f32; k * r];
        for i in 0..k * r {
            u[i] = crate::kernels::dot_f32(&self.b[i * e..(i + 1) * e], hu);
        }
        (g, u)
    }

    fn shift(&self, row: &[f32], g: &[f32], u: &[f32]) -> f32 {
        let r = self.cfg.rank;
        let mut s = 0.0;
        for k in 0..self.cfg.n_experts {
            let mut d = 0.0;
            for j in 0..r {
                d += row[k * r + j] * u[k * r + j];
            }
            s += g[k] * d;
        }
        s.clamp(-self.cfg.max_shift, self.cfg.max_shift)
    }

    /// Add the learned corrections to `logits` for hidden state `h`.
    pub fn apply(&self, h: &[f32], logits: &mut [f32]) {
        if self.a.is_empty() || h.len() != self.n_embd {
            return;
        }
        let hu = Self::unit(h);
        let (g, u) = self.route(&hu);
        for (&tok, row) in &self.a {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l += self.shift(row, &g, &u);
            }
        }
    }

    /// Router weights for a hidden state (which experts are active), for inspection.
    pub fn gate(&self, h: &[f32]) -> Vec<f32> {
        self.route(&Self::unit(h)).0
    }

    /// REINFORCE update on one step: increase (`reward > 0`) or decrease (`< 0`) the
    /// probability of the token that was produced.
    pub fn update(&mut self, step: &Step, reward: f32) {
        if reward == 0.0 || step.h.len() != self.n_embd {
            return;
        }
        let (k, r, e) = (self.cfg.n_experts, self.cfg.rank, self.n_embd);
        let h: Vec<f32> = step.h.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let hu = Self::unit(&h);
        let (g, u) = self.route(&hu);
        // current policy over the recorded support
        let zero = vec![0.0f32; k * r];
        let mut logits: Vec<f32> =
            step.top.iter().map(|&(t, base)| base + self.a.get(&t).map(|row| self.shift(row, &g, &u)).unwrap_or(0.0)).collect();
        crate::ops::softmax(&mut logits);
        let delta: Vec<f32> = step.top.iter().zip(&logits).map(|(&(t, _), &p)| reward * ((t == step.token) as u8 as f32 - p)).collect();
        // gradients for B and the router use A before this update
        let mut at_delta = vec![0.0f32; k * r]; // Σ_j δ_j A_k[j]
        let mut dg = vec![0.0f32; k]; // Σ_j δ_j z_kj
        for (&(t, _), &d) in step.top.iter().zip(&delta) {
            let row = self.a.get(&t).map(|x| &x[..]).unwrap_or(&zero);
            for kk in 0..k {
                let mut z = 0.0;
                for j in 0..r {
                    at_delta[kk * r + j] += d * row[kk * r + j];
                    z += row[kk * r + j] * u[kk * r + j];
                }
                dg[kk] += d * z;
            }
        }
        let lr = self.cfg.lr;
        // A rows
        for (&(t, _), &d) in step.top.iter().zip(&delta) {
            if d.abs() < 1e-6 {
                continue;
            }
            let row = self.a.entry(t).or_insert_with(|| vec![0.0; k * r].into_boxed_slice());
            for kk in 0..k {
                for j in 0..r {
                    let w = &mut row[kk * r + j];
                    *w = *w * (1.0 - self.cfg.weight_decay) + lr * g[kk] * d * u[kk * r + j];
                }
            }
        }
        // B
        for kk in 0..k {
            for j in 0..r {
                let c = lr * 0.1 * g[kk] * at_delta[kk * r + j];
                if c != 0.0 {
                    let bb = &mut self.b[(kk * r + j) * e..(kk * r + j + 1) * e];
                    bb.iter_mut().zip(&hu).for_each(|(w, x)| *w += c * x);
                }
            }
        }
        // router
        let mean: f32 = g.iter().zip(&dg).map(|(a, b)| a * b).sum();
        for kk in 0..k {
            let c = lr * 0.1 * g[kk] * (dg[kk] - mean);
            if c != 0.0 {
                self.wg[kk * e..(kk + 1) * e].iter_mut().zip(&hu).for_each(|(w, x)| *w += c * x);
            }
        }
        self.stats.updates += 1;
        self.stats.rows = self.a.len();
    }

    /// Apply `reward` to every step, `epochs` times (several passes make a single
    /// correction take effect immediately).
    pub fn reinforce(&mut self, steps: &[Step], reward: f32, epochs: usize) {
        for _ in 0..epochs.max(1) {
            for s in steps {
                self.update(s, reward);
            }
        }
        self.stats.reward_sum += reward as f64;
        self.stats.reward_count += 1;
    }

    pub fn reset(&mut self) {
        let cfg = self.cfg.clone();
        *self = Self::new(self.n_embd, self.n_vocab, cfg);
    }

    // ---- persistence ----

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIEA\x01\0\0\0");
        for v in [self.cfg.n_experts, self.cfg.rank, self.n_embd, self.n_vocab, self.a.len()] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        out.extend_from_slice(&self.cfg.max_shift.to_le_bytes());
        out.extend_from_slice(&self.stats.updates.to_le_bytes());
        for v in self.b.iter().chain(&self.wg) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut rows: Vec<_> = self.a.iter().collect();
        rows.sort_by_key(|(t, _)| **t);
        for (t, row) in rows {
            out.extend_from_slice(&t.to_le_bytes());
            for v in row.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        let tmp = path.as_ref().with_extension("tmp");
        std::fs::File::create(&tmp)?.write_all(&out)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>, cfg: AdapterConfig) -> Result<Self> {
        let mut buf = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut buf)?;
        if buf.len() < 8 + 40 + 12 || &buf[..4] != b"RIEA" {
            bail!(Format, "not an RIE adapter file");
        }
        let mut pos = 8;
        let u64r = |p: &mut usize| {
            let v = u64::from_le_bytes(buf[*p..*p + 8].try_into().unwrap());
            *p += 8;
            v as usize
        };
        let (k, r, e, v, n_rows) = (u64r(&mut pos), u64r(&mut pos), u64r(&mut pos), u64r(&mut pos), u64r(&mut pos));
        let max_shift = f32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let updates = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let need = pos + 4 * (k * r * e + k * e) + n_rows * (4 + 4 * k * r);
        if buf.len() != need {
            bail!(Format, "adapter file is truncated or corrupt");
        }
        let f32s = |p: &mut usize, n: usize| -> Vec<f32> {
            let v = buf[*p..*p + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            *p += 4 * n;
            v
        };
        let b = f32s(&mut pos, k * r * e);
        let wg = f32s(&mut pos, k * e);
        let mut a = HashMap::with_capacity(n_rows);
        for _ in 0..n_rows {
            let t = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            a.insert(t, f32s(&mut pos, k * r).into_boxed_slice());
        }
        Ok(CorrectionExperts {
            cfg: AdapterConfig { n_experts: k, rank: r, max_shift, ..cfg },
            n_embd: e,
            n_vocab: v,
            b,
            wg,
            stats: AdapterStats { updates, rows: a.len(), ..Default::default() },
            a,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hidden(seed: u64, e: usize) -> Vec<f32> {
        let mut r = Rng(seed * 7919 + 1);
        (0..e).map(|_| r.normal()).collect()
    }

    #[test]
    fn negative_then_positive_feedback_flips_the_choice() {
        let (e, v) = (64, 100);
        let mut ex = CorrectionExperts::new(e, v, AdapterConfig::default());
        let h = hidden(1, e);
        let mut base = vec![0.0f32; v];
        base[3] = 2.0; // model prefers 3
        base[7] = 1.5; // the right answer is 7
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let s_bad = Step::new(&h, &base, 3, 16);
        let s_good = Step::new(&h, &base, 7, 16);
        ex.reinforce(&[s_bad], -1.0, 3);
        ex.reinforce(&[s_good], 1.0, 3);
        let mut l = base.clone();
        ex.apply(&h, &mut l);
        assert_eq!(argmax(&l), 7, "correction must take effect immediately: {:?} {:?}", l[3], l[7]);
        // an unrelated context is affected much less (router + low rank)
        let h2 = hidden(99, e);
        let mut l2 = base.clone();
        ex.apply(&h2, &mut l2);
        let shift_same = (l[7] - base[7]) - (l[3] - base[3]);
        let shift_other = (l2[7] - base[7]) - (l2[3] - base[3]);
        assert!(shift_same > shift_other.abs(), "{shift_same} vs {shift_other}");
    }

    #[test]
    fn shifts_are_bounded_and_persist() {
        let (e, v) = (32, 50);
        let mut ex = CorrectionExperts::new(e, v, AdapterConfig { max_shift: 2.0, ..Default::default() });
        let h = hidden(3, e);
        let base = vec![0.0f32; v];
        let s = Step::new(&h, &base, 5, 8);
        ex.reinforce(&[s], 1.0, 200);
        let mut l = base.clone();
        ex.apply(&h, &mut l);
        assert!(l[5] <= 2.0 + 1e-4 && l[5] > 0.5, "{}", l[5]);
        let dir = std::env::temp_dir().join(format!("rie-adapter-{}.bin", std::process::id()));
        ex.save(&dir).unwrap();
        let ex2 = CorrectionExperts::load(&dir, AdapterConfig::default()).unwrap();
        let mut l2 = base.clone();
        ex2.apply(&h, &mut l2);
        assert!((l2[5] - l[5]).abs() < 1e-5);
        let _ = std::fs::remove_file(dir);
    }
}
