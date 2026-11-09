mod common;
use common::*;
use rie_core::kernels::matmul::{formats_for, matmul, matmul_multi, MatmulJob, QuantActs};
use rie_core::kernels::{dot_f16_f32, dot_f32, vec_dot, axpy_f16, f32_to_f16_slice, ActRow, CpuFeatures};
use rie_core::quant::*;
use rie_core::tensor::Weight;
use rie_core::threadpool::{PoolConfig, ThreadPool};
use std::time::Duration;

fn reference_dot(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let mut wf = vec![0.0; n];
    dequantize_row(ty, w, &mut wf);
    // compare against the *quantized* activation the kernel actually sees
    let xa: Vec<f32> = match ty.act_format() {
        ActFormat::F32 => x.to_vec(),
        ActFormat::Q8_0 => {
            let mut b = vec![BlockQ8_0::default(); n / QK];
            quantize_row_q8_0(x, &mut b);
            dequantize_q8_0(&b)
        }
        ActFormat::Q8K => {
            let mut b = vec![BlockQ8K::default(); n / QK_K];
            quantize_row_q8_k(x, &mut b);
            dequantize_q8_k(&b)
        }
    };
    wf.iter().zip(&xa).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32
}

#[test]
fn print_features() {
    eprintln!("CPU features: {} -> kernels {}", CpuFeatures::get().describe(), CpuFeatures::get().kernel_name());
}

#[test]
fn vec_dot_matches_dequantized_reference_for_all_types() {
    let mut rng = Rng::new(7);
    for &ty in ALL_TYPES {
        for &n in &[256usize, 512, 768, 2048] {
            for trial in 0..4 {
                let w = random_weight(&mut rng, ty, 1, n);
                let x = rng.vec(n, 2.0);
                let expect = reference_dot(ty, &w, &x);
                let mut q0 = vec![BlockQ8_0::default(); n / QK];
                let mut qk = vec![BlockQ8K::default(); n / QK_K];
                quantize_row_q8_0(&x, &mut q0);
                quantize_row_q8_k(&x, &mut qk);
                let a = match ty.act_format() {
                    ActFormat::F32 => ActRow::F32(&x),
                    ActFormat::Q8_0 => ActRow::Q8_0(&q0),
                    ActFormat::Q8K => ActRow::Q8K(&qk),
                };
                let got = vec_dot(ty, &w, a, n);
                let mag: f32 = {
                    let mut wf = vec![0.0; n];
                    dequantize_row(ty, &w, &mut wf);
                    wf.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum()
                };
                assert!(
                    (got - expect).abs() <= 1e-4 * mag + 1e-5,
                    "{ty:?} n={n} trial={trial}: got {got}, expected {expect} (mag {mag})"
                );
            }
        }
    }
}

#[test]
fn scalar_reference_dots_agree_with_dequant() {
    // Validates the reference int8 dot implementations themselves.
    let mut rng = Rng::new(11);
    let n = 512;
    for &ty in &[GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0] {
        let w = random_weight(&mut rng, ty, 1, n);
        let x = rng.vec(n, 1.0);
        let mut q = vec![BlockQ8_0::default(); n / QK];
        quantize_row_q8_0_scalar(&x, &mut q);
        let got = match ty {
            GgmlType::Q4_0 => reference::vec_dot_q4_0(&w, &q),
            GgmlType::Q4_1 => reference::vec_dot_q4_1(&w, &q),
            GgmlType::Q5_0 => reference::vec_dot_q5_0(&w, &q),
            GgmlType::Q5_1 => reference::vec_dot_q5_1(&w, &q),
            _ => reference::vec_dot_q8_0(&w, &q),
        };
        let mut wf = vec![0.0; n];
        dequantize_row(ty, &w, &mut wf);
        let xa = dequantize_q8_0(&q);
        let expect: f32 = wf.iter().zip(&xa).map(|(a, b)| a * b).sum();
        assert!((got - expect).abs() < 1e-3 * expect.abs().max(1.0), "{ty:?}: {got} vs {expect}");
    }
    for &ty in &[GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        let w = random_weight(&mut rng, ty, 1, n);
        let x = rng.vec(n, 1.0);
        let mut q = vec![BlockQ8K::default(); n / QK_K];
        quantize_row_q8_k(&x, &mut q);
        let got = match ty {
            GgmlType::Q4_K => reference::vec_dot_q4_k(&w, &q),
            GgmlType::Q5_K => reference::vec_dot_q5_k(&w, &q),
            _ => reference::vec_dot_q6_k(&w, &q),
        };
        let mut wf = vec![0.0; n];
        dequantize_row(ty, &w, &mut wf);
        let xa = dequantize_q8_k(&q);
        let expect: f32 = wf.iter().zip(&xa).map(|(a, b)| a * b).sum();
        assert!((got - expect).abs() < 1e-3 * expect.abs().max(1.0), "{ty:?}: {got} vs {expect}");
    }
}

#[test]
fn quantizers_roundtrip() {
    let mut rng = Rng::new(3);
    let x = rng.vec(1024, 1.0);
    for (ty, data, tol) in [
        (GgmlType::Q8_0, quantize::q8_0(&x), 0.01),
        (GgmlType::Q4_0, quantize::q4_0(&x), 0.15),
        (GgmlType::F16, quantize::f16(&x), 0.001),
    ] {
        let mut y = vec![0.0; x.len()];
        dequantize_row(ty, &data, &mut y);
        assert_close(&y, &x, tol, ty.name());
    }
}

#[test]
fn float_kernels() {
    let mut rng = Rng::new(5);
    for n in [1usize, 7, 8, 15, 16, 33, 128, 1000] {
        let a = rng.vec(n, 1.0);
        let b = rng.vec(n, 1.0);
        let expect: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_f32(&a, &b) - expect).abs() < 1e-4 * n as f32);
        let mut h = vec![0u16; n];
        f32_to_f16_slice(&a, &mut h);
        for i in 0..n {
            assert_eq!(h[i], half::f16::from_f32(a[i]).to_bits(), "f32->f16 rounding at {i}");
        }
        let ah: Vec<f32> = h.iter().map(|&v| half::f16::from_bits(v).to_f32()).collect();
        let expect: f32 = ah.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot_f16_f32(&h, &b) - expect).abs() < 1e-4 * n as f32);
        let mut y = b.clone();
        axpy_f16(&mut y, 0.5, &h);
        for i in 0..n {
            assert!((y[i] - (b[i] + 0.5 * ah[i])).abs() < 1e-5);
        }
    }
}

#[test]
fn parallel_matmul_matches_naive_for_batches() {
    let pool = ThreadPool::new(PoolConfig { n_threads: 4, cores: vec![], spin: Duration::from_micros(20) });
    let mut rng = Rng::new(42);
    let (rows, k) = (200usize, 512usize);
    for &ty in &[GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::F16, GgmlType::Q5_0] {
        let w = Weight::owned(ty, rows, k, random_weight(&mut rng, ty, rows, k));
        for &n_tok in &[1usize, 2, 3, 8] {
            let x = rng.vec(n_tok * k, 1.0);
            let mut acts = QuantActs::new();
            acts.prepare(&pool, &x, n_tok, k, &formats_for(&[&w]));
            let mut out = vec![0.0; n_tok * rows];
            matmul(&pool, &acts, &w, &mut out);
            let mut expect = vec![0.0; n_tok * rows];
            for t in 0..n_tok {
                for r in 0..rows {
                    expect[t * rows + r] = reference_dot(ty, w.row(r), &x[t * k..(t + 1) * k]);
                }
            }
            assert_close(&out, &expect, 1e-4, &format!("{ty:?} n_tok={n_tok}"));
        }
    }
    // multi-job dispatch with mixed types sharing one activation batch
    let w1 = Weight::owned(GgmlType::Q4_K, 96, k, random_weight(&mut rng, GgmlType::Q4_K, 96, k));
    let w2 = Weight::owned(GgmlType::Q8_0, 64, k, random_weight(&mut rng, GgmlType::Q8_0, 64, k));
    let x = rng.vec(3 * k, 1.0);
    let mut acts = QuantActs::new();
    acts.prepare(&pool, &x, 3, k, &formats_for(&[&w1, &w2]));
    let mut o1 = vec![0.0; 3 * 96];
    let mut o2 = vec![0.0; 3 * 64];
    matmul_multi(&pool, &acts, &mut [MatmulJob { w: &w1, out: &mut o1, ld: 96 }, MatmulJob { w: &w2, out: &mut o2, ld: 64 }]);
    for t in 0..3 {
        for r in 0..96 {
            let e = reference_dot(GgmlType::Q4_K, w1.row(r), &x[t * k..(t + 1) * k]);
            assert!((o1[t * 96 + r] - e).abs() < 1e-3);
        }
        for r in 0..64 {
            let e = reference_dot(GgmlType::Q8_0, w2.row(r), &x[t * k..(t + 1) * k]);
            assert!((o2[t * 64 + r] - e).abs() < 1e-3);
        }
    }
}
