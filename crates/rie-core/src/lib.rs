//! # rie-core — Rust Inference Engine for mobile
//!
//! A from-scratch, dependency-light LLM inference runtime in the spirit of llama.cpp,
//! tuned for phone SoCs:
//!
//! * **GGUF** models (zero-copy mmap), all common ggml quantizations
//!   (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K…Q6_K, F16, BF16, F32).
//! * **ARM-first kernels**: NEON, `SDOT` (dotprod) and `SMMLA` (i8mm) selected at runtime;
//!   AVX2 for x86_64; scalar fallback everywhere else.
//! * **big.LITTLE-aware scheduling**: performance-core detection, thread pinning,
//!   dynamic work distribution, spin-then-sleep worker pool.
//! * **Architectures**: Llama 2/3.x, Mistral, Qwen2/2.5, Qwen3, Phi-3/3.5/4-mini,
//!   Gemma, Gemma 2, Gemma 3, SmolLM, and other Llama-compatible models.
//! * f16 KV cache, batched prompt processing, prompt-prefix reuse between turns,
//!   pluggable backends (see the `rie-gpu` crate for Vulkan/Metal via wgpu).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod chat;
pub mod constrain;
pub mod convert;
pub mod cpu;
pub mod engine;
pub mod error;
pub mod gguf;
pub mod kernels;
pub mod memory;
pub mod model;
pub mod ops;
pub mod quant;
pub mod rl;
pub mod sampler;
pub mod speech;
pub mod tensor;
#[doc(hidden)]
pub mod testing;
pub mod text;
pub mod threadpool;
pub mod tokenizer;
pub mod turbo;

pub use engine::{Engine, EngineConfig, GenerateParams, GenerationStats, StopReason};
pub use error::{Error, Result};
pub use kernels::CpuFeatures;
pub use model::cpu::KvType;
pub use model::{Backend, ModelConfig};
pub use sampler::{Sampler, SamplerParams};
pub use tokenizer::Tokenizer;
