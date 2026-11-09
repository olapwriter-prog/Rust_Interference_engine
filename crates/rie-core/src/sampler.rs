//! Token sampling: repetition penalties, temperature, top-k, top-p, min-p.

#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// 0.0 = greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// How many recent tokens the penalties look at.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        SamplerParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: 0x5EED,
        }
    }
}

impl SamplerParams {
    pub fn greedy() -> Self {
        SamplerParams { temperature: 0.0, repeat_penalty: 1.0, ..Default::default() }
    }
}

pub struct Sampler {
    pub params: SamplerParams,
    rng: u64,
    cand: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(params: SamplerParams) -> Self {
        let rng = params.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        Sampler { params, rng, cand: Vec::new() }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token from `logits` (modified in place by penalties).
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let p = self.params.clone();
        // penalties
        if (p.repeat_penalty != 1.0 || p.frequency_penalty != 0.0 || p.presence_penalty != 0.0) && !history.is_empty() {
            let start = history.len().saturating_sub(p.repeat_last_n);
            let recent = &history[start..];
            let mut counts: Vec<(u32, u32)> = Vec::new();
            for &t in recent {
                match counts.iter_mut().find(|(id, _)| *id == t) {
                    Some((_, c)) => *c += 1,
                    None => counts.push((t, 1)),
                }
            }
            for (t, c) in counts {
                let Some(l) = logits.get_mut(t as usize) else { continue };
                if p.repeat_penalty != 1.0 {
                    *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
                }
                *l -= c as f32 * p.frequency_penalty + p.presence_penalty;
            }
        }
        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        // top-k (partial selection, O(n))
        self.cand.clear();
        self.cand.extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)));
        let k = if p.top_k == 0 { self.cand.len() } else { p.top_k.min(self.cand.len()) };
        if k < self.cand.len() {
            self.cand.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.cand.truncate(k);
        }
        self.cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // temperature softmax
        let max = self.cand[0].1;
        let mut sum = 0.0;
        for c in self.cand.iter_mut() {
            c.1 = ((c.1 - max) / p.temperature).exp();
            sum += c.1;
        }
        for c in self.cand.iter_mut() {
            c.1 /= sum;
        }
        // min-p
        if p.min_p > 0.0 {
            let thresh = self.cand[0].1 * p.min_p;
            let keep = self.cand.iter().take_while(|c| c.1 >= thresh).count().max(1);
            self.cand.truncate(keep);
        }
        // top-p
        if p.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = self.cand.len();
            for (i, c) in self.cand.iter().enumerate() {
                cum += c.1;
                if cum >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            self.cand.truncate(keep);
        }
        let total: f32 = self.cand.iter().map(|c| c.1).sum();
        let mut r = self.next_f32() * total;
        for c in &self.cand {
            r -= c.1;
            if r <= 0.0 {
                return c.0;
            }
        }
        self.cand.last().unwrap().0
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_topk1_agree() {
        let logits = vec![0.1, 3.0, -1.0, 2.9];
        let mut s = Sampler::new(SamplerParams::greedy());
        assert_eq!(s.sample(&mut logits.clone(), &[]), 1);
        let mut s = Sampler::new(SamplerParams { top_k: 1, repeat_penalty: 1.0, ..Default::default() });
        assert_eq!(s.sample(&mut logits.clone(), &[]), 1);
    }

    #[test]
    fn repetition_penalty_changes_choice() {
        let mut s = Sampler::new(SamplerParams { repeat_penalty: 2.0, ..SamplerParams::greedy() });
        assert_eq!(s.sample(&mut vec![0.1, 3.0, -1.0, 2.9], &[1]), 3);
    }

    #[test]
    fn sampling_distribution_is_reasonable() {
        let mut s = Sampler::new(SamplerParams { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, ..Default::default() });
        let mut hits = [0usize; 2];
        for _ in 0..4000 {
            let t = s.sample(&mut vec![0.0, (3.0f32).ln()], &[]);
            hits[t as usize] += 1;
        }
        let frac = hits[1] as f32 / 4000.0;
        assert!((frac - 0.75).abs() < 0.04, "frac={frac}");
    }
}
