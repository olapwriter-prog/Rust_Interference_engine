//! Parallel quantized matrix multiplication `Y = X · Wᵀ`.
//!
//! * `X`: `n_tok × k` activations (f32), quantized once per call to the int8 format each
//!   weight type needs and shared by all weights consuming the same input (e.g. Q, K, V).
//! * `W`: `rows × k` in its native ggml block format (never repacked on CPU).
//! * Work is split into row chunks handed out dynamically so heterogeneous cores share
//!   the load proportionally to their speed. Several weights can be computed in one
//!   parallel region (`matmul_multi`) to save fork/join barriers per layer.
//! * During prompt processing every weight row is loaded once and reused for all tokens
//!   of the batch (and with `SMMLA` 2×2 register tiles on i8mm CPUs).

use super::{has_mmla, mmla_2x2, vec_dot, ActRow};
use crate::quant::{quantize_row_q8_k, ActFormat, BlockQ8K, BlockQ8_0, QK, QK_K};
use crate::tensor::Weight;
use crate::threadpool::ThreadPool;

/// Reusable buffers holding one batch of quantized activations.
#[derive(Default)]
pub struct QuantActs {
    pub n_tok: usize,
    pub k: usize,
    f32: Vec<f32>,
    q8_0: Vec<BlockQ8_0>,
    q8k: Vec<BlockQ8K>,
    have: [bool; 3],
}

impl QuantActs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Quantize `x` (`n_tok × k`) into every format in `formats`.
    pub fn prepare(&mut self, pool: &ThreadPool, x: &[f32], n_tok: usize, k: usize, formats: &[ActFormat]) {
        assert!(x.len() >= n_tok * k);
        self.n_tok = n_tok;
        self.k = k;
        self.have = [false; 3];
        for &f in formats {
            match f {
                ActFormat::F32 if !self.have[0] => {
                    self.f32.clear();
                    self.f32.extend_from_slice(&x[..n_tok * k]);
                    self.have[0] = true;
                }
                ActFormat::Q8_0 if !self.have[1] => {
                    assert!(k % QK == 0, "row length {k} not a multiple of {QK}");
                    let nb = k / QK;
                    self.q8_0.resize(n_tok * nb, BlockQ8_0::default());
                    let out = SyncPtr(self.q8_0.as_mut_ptr());
                    pool.for_each(n_tok, 1, |t, _| {
                        let o = unsafe { std::slice::from_raw_parts_mut(out.get().add(t * nb), nb) };
                        crate::quant::quantize_row_q8_0(&x[t * k..(t + 1) * k], o);
                    });
                    self.have[1] = true;
                }
                ActFormat::Q8K if !self.have[2] => {
                    assert!(k % QK_K == 0, "row length {k} not a multiple of {QK_K}");
                    let nb = k / QK_K;
                    self.q8k.resize(n_tok * nb, BlockQ8K::default());
                    let out = SyncPtr(self.q8k.as_mut_ptr());
                    pool.for_each(n_tok, 1, |t, _| {
                        let o = unsafe { std::slice::from_raw_parts_mut(out.get().add(t * nb), nb) };
                        quantize_row_q8_k(&x[t * k..(t + 1) * k], o);
                    });
                    self.have[2] = true;
                }
                _ => {}
            }
        }
    }

    #[inline]
    pub fn row(&self, fmt: ActFormat, t: usize) -> ActRow<'_> {
        match fmt {
            ActFormat::F32 => {
                debug_assert!(self.have[0]);
                ActRow::F32(&self.f32[t * self.k..(t + 1) * self.k])
            }
            ActFormat::Q8_0 => {
                debug_assert!(self.have[1]);
                let nb = self.k / QK;
                ActRow::Q8_0(&self.q8_0[t * nb..(t + 1) * nb])
            }
            ActFormat::Q8K => {
                debug_assert!(self.have[2]);
                let nb = self.k / QK_K;
                ActRow::Q8K(&self.q8k[t * nb..(t + 1) * nb])
            }
        }
    }

    fn q8_0_row(&self, t: usize) -> &[BlockQ8_0] {
        let nb = self.k / QK;
        &self.q8_0[t * nb..(t + 1) * nb]
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// One matmul job: `out[t * ld + r] = W[r] · X[t]` for all rows/tokens.
pub struct MatmulJob<'a> {
    pub w: &'a Weight,
    pub out: &'a mut [f32],
    /// Leading dimension (stride between tokens) of `out`; usually `w.rows`.
    pub ld: usize,
}

/// Formats needed for a set of weights.
pub fn formats_for(ws: &[&Weight]) -> Vec<ActFormat> {
    let mut v = Vec::new();
    for w in ws {
        let f = w.ty.act_format();
        if !v.contains(&f) {
            v.push(f);
        }
    }
    v
}

pub fn matmul(pool: &ThreadPool, acts: &QuantActs, w: &Weight, out: &mut [f32]) {
    let ld = w.rows;
    matmul_multi(pool, acts, &mut [MatmulJob { w, out, ld }]);
}

/// Compute several matmuls sharing the same input activations in a single parallel region.
pub fn matmul_multi(pool: &ThreadPool, acts: &QuantActs, jobs: &mut [MatmulJob<'_>]) {
    let n_tok = acts.n_tok;
    let n_threads = pool.n_threads();
    // Chunking: aim for ~8 chunks per thread per matrix for load balance across
    // big/little cores, but keep chunks ≥ 16 rows to amortise scheduling.
    struct Plan {
        w: *const Weight,
        out: SyncPtr<f32>,
        ld: usize,
        chunk: usize,
        n_chunks: usize,
    }
    let mut plans = Vec::with_capacity(jobs.len());
    let mut total = 0usize;
    for j in jobs.iter_mut() {
        assert_eq!(j.w.cols, acts.k, "matmul inner dimension mismatch");
        assert!(j.out.len() >= (n_tok - 1) * j.ld + j.w.rows);
        let rows = j.w.rows;
        let target = (rows / (n_threads * 8)).max(16);
        let chunk = target.div_ceil(4) * 4;
        let n_chunks = rows.div_ceil(chunk);
        total += n_chunks;
        plans.push(Plan { w: j.w as *const Weight, out: SyncPtr(j.out.as_mut_ptr()), ld: j.ld, chunk, n_chunks });
    }
    struct Plans(Vec<Plan>);
    unsafe impl Sync for Plans {}
    let plans = &Plans(plans);

    pool.for_each(total, 1, |mut c, _tid| {
        let mut pi = 0;
        while c >= plans.0[pi].n_chunks {
            c -= plans.0[pi].n_chunks;
            pi += 1;
        }
        let p = &plans.0[pi];
        let w = unsafe { &*p.w };
        let r0 = c * p.chunk;
        let r1 = (r0 + p.chunk).min(w.rows);
        compute_rows(w, acts, r0, r1, p.out, p.ld);
    });
}

#[inline]
fn compute_rows(w: &Weight, acts: &QuantActs, r0: usize, r1: usize, out: SyncPtr<f32>, ld: usize) {
    let n_tok = acts.n_tok;
    let fmt = w.ty.act_format();
    let k = w.cols;
    let out = out.get();
    let mut r = r0;
    if n_tok >= 2 && has_mmla(w.ty) {
        // 2×2 SMMLA tiles.
        while r + 2 <= r1 {
            let w0 = w.row(r);
            let w1 = w.row(r + 1);
            let mut t = 0;
            while t + 2 <= n_tok {
                let res = mmla_2x2(w.ty, w0, w1, acts.q8_0_row(t), acts.q8_0_row(t + 1)).unwrap();
                unsafe {
                    *out.add(t * ld + r) = res[0];
                    *out.add((t + 1) * ld + r) = res[1];
                    *out.add(t * ld + r + 1) = res[2];
                    *out.add((t + 1) * ld + r + 1) = res[3];
                }
                t += 2;
            }
            if t < n_tok {
                let a = acts.row(fmt, t);
                unsafe {
                    *out.add(t * ld + r) = vec_dot(w.ty, w0, a, k);
                    *out.add(t * ld + r + 1) = vec_dot(w.ty, w1, a, k);
                }
            }
            r += 2;
        }
    }
    // Generic path: each weight row stays hot in L1 while it is dotted with all tokens.
    while r < r1 {
        let wr = w.row(r);
        for t in 0..n_tok {
            let v = vec_dot(w.ty, wr, acts.row(fmt, t), k);
            unsafe { *out.add(t * ld + r) = v };
        }
        r += 1;
    }
}
