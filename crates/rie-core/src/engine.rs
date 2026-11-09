//! High-level engine API: load a model, run prompts with streaming output, chat with
//! automatic KV-cache reuse across turns, benchmark.

use crate::chat::{ChatTemplate, Message};
use crate::cpu::CpuTopology;
use crate::error::{bail, Result};
use crate::gguf::GgufFile;
use crate::model::cpu::CpuBackend;
use crate::model::{Backend, Model};
use crate::sampler::{Sampler, SamplerParams};
use crate::threadpool::{PoolConfig, ThreadPool};
use crate::tokenizer::{Tokenizer, Utf8Stream};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Worker threads (default: number of performance cores, max 8).
    pub n_threads: Option<usize>,
    /// Context window (KV cache size). Default 4096; capped at the model's training context.
    pub n_ctx: usize,
    /// Max tokens per forward pass during prompt processing.
    pub n_batch: usize,
    /// Pin worker threads to performance cores (Linux/Android).
    pub pin_threads: bool,
    /// Idle spin time of workers before sleeping.
    pub spin: Duration,
    /// mlock() the weights (avoids page-out on memory pressure; may need permissions).
    pub mlock: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { n_threads: None, n_ctx: 4096, n_batch: 64, pin_threads: true, spin: Duration::from_micros(300), mlock: false }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub sampler: SamplerParams,
    pub stop_strings: Vec<String>,
    /// Map literal special-token text in the prompt (e.g. `<|im_start|>`) to token ids.
    pub parse_special: bool,
    /// Emit control tokens in the output text.
    pub show_special: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_tokens: 512,
            sampler: SamplerParams::default(),
            stop_strings: Vec::new(),
            parse_special: true,
            show_special: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfSequence,
    MaxTokens,
    StopString,
    ContextFull,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    /// Prompt tokens served from the KV cache (prefix shared with the previous request).
    pub reused_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_time: Duration,
    pub gen_time: Duration,
    pub stop: StopReason,
}

impl GenerationStats {
    pub fn prompt_tps(&self) -> f64 {
        let n = self.prompt_tokens - self.reused_tokens;
        if self.prompt_time.is_zero() {
            0.0
        } else {
            n as f64 / self.prompt_time.as_secs_f64()
        }
    }
    pub fn gen_tps(&self) -> f64 {
        if self.gen_time.is_zero() {
            0.0
        } else {
            self.generated_tokens as f64 / self.gen_time.as_secs_f64()
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub n_prompt: usize,
    pub n_gen: usize,
    pub prompt_tps: f64,
    pub gen_tps: f64,
}

pub struct Engine {
    pub model: Arc<Model>,
    pub tokenizer: Arc<Tokenizer>,
    backend: Box<dyn Backend>,
    /// Tokens whose K/V are currently in the cache, in order.
    cached: Vec<u32>,
    logits: Vec<f32>,
    pub template: ChatTemplate,
}

/// Load the GGUF file, model weights and tokenizer.
pub fn load_model(path: impl AsRef<Path>) -> Result<(Arc<Model>, Arc<Tokenizer>)> {
    let gguf = Arc::new(GgufFile::open(path)?);
    load_model_from(gguf)
}

pub fn load_model_from(gguf: Arc<GgufFile>) -> Result<(Arc<Model>, Arc<Tokenizer>)> {
    let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
    let model = Arc::new(Model::load(gguf)?);
    Ok((model, tokenizer))
}

/// Create the shared worker pool according to `cfg` and the detected SoC topology.
pub fn make_pool(cfg: &EngineConfig) -> Arc<ThreadPool> {
    let topo = CpuTopology::detect();
    let n_threads = cfg.n_threads.unwrap_or_else(|| topo.recommended_threads()).max(1);
    let mut cores = Vec::new();
    if cfg.pin_threads && topo.can_pin() {
        let perf = topo.performance_cores();
        cores = if n_threads <= perf.len() { perf } else { (0..topo.n_logical).collect() };
        crate::cpu::pin_current_thread(&cores);
    }
    crate::cpu::boost_current_thread();
    Arc::new(ThreadPool::new(PoolConfig { n_threads, cores, spin: cfg.spin }))
}

impl Engine {
    /// Load a GGUF model on the CPU backend.
    pub fn load(path: impl AsRef<Path>, cfg: EngineConfig) -> Result<Self> {
        let (model, tokenizer) = load_model(path)?;
        if cfg.mlock {
            model.gguf.lock_in_memory();
        }
        let n_ctx = cfg.n_ctx.min(model.cfg.ctx_train).max(16);
        let pool = make_pool(&cfg);
        let backend = CpuBackend::new(model.clone(), pool, n_ctx, cfg.n_batch);
        Ok(Self::with_backend(model, tokenizer, Box::new(backend)))
    }

    pub fn with_backend(model: Arc<Model>, tokenizer: Arc<Tokenizer>, backend: Box<dyn Backend>) -> Self {
        let template = ChatTemplate::detect(&tokenizer);
        let n_vocab = model.cfg.n_vocab;
        Engine { model, tokenizer, backend, cached: Vec::new(), logits: vec![0.0; n_vocab], template }
    }

    pub fn backend_name(&self) -> String {
        self.backend.name()
    }
    pub fn n_ctx(&self) -> usize {
        self.backend.n_ctx()
    }
    pub fn n_cached(&self) -> usize {
        self.cached.len()
    }

    /// Forget the conversation (the KV cache is simply overwritten later).
    pub fn reset(&mut self) {
        self.cached.clear();
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        self.tokenizer.encode(text, add_special, parse_special)
    }

    /// Make the KV cache hold exactly `tokens`, reusing the longest common prefix with the
    /// current cache, and leave the logits of the last token in `self.logits`.
    /// Returns the number of reused tokens.
    pub fn eval_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        if tokens.is_empty() {
            bail!(Runtime, "empty prompt");
        }
        if tokens.len() > self.backend.n_ctx() {
            bail!(Runtime, "prompt of {} tokens exceeds context window of {}", tokens.len(), self.backend.n_ctx());
        }
        let mut common = self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        // We need logits for the last token, so always re-evaluate at least one token.
        if common == tokens.len() {
            common -= 1;
        }
        self.cached.truncate(common);
        let nb = self.backend.max_batch();
        let mut pos = common;
        while pos < tokens.len() {
            let end = (pos + nb).min(tokens.len());
            self.backend.forward(&tokens[pos..end], pos, false, &mut self.logits)?;
            self.cached.extend_from_slice(&tokens[pos..end]);
            pos = end;
        }
        Ok(common)
    }

    /// Logits of the last evaluated token.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Append one token and evaluate it.
    pub fn eval_next(&mut self, token: u32) -> Result<()> {
        let pos = self.cached.len();
        if pos >= self.backend.n_ctx() {
            bail!(Runtime, "context window full");
        }
        self.backend.forward(&[token], pos, false, &mut self.logits)?;
        self.cached.push(token);
        Ok(())
    }

    /// Generate a continuation of `prompt` (raw text, BOS added automatically), streaming
    /// text pieces to `on_text`; return `false` from the callback to stop.
    pub fn generate(&mut self, prompt: &str, params: &GenerateParams, on_text: impl FnMut(&str) -> bool) -> Result<GenerationStats> {
        let tokens = self.tokenizer.encode(prompt, true, params.parse_special);
        self.generate_from_tokens(&tokens, params, on_text)
    }

    /// Chat completion: renders the conversation with the model's template. Shared prefixes
    /// with the previous turn are served from the KV cache, so follow-up turns only process
    /// the new message.
    pub fn chat(&mut self, messages: &[Message], params: &GenerateParams, on_text: impl FnMut(&str) -> bool) -> Result<GenerationStats> {
        let prompt = self.template.render(messages, true);
        let tokens = self.tokenizer.encode(&prompt, true, true);
        self.generate_from_tokens(&tokens, params, on_text)
    }

    pub fn generate_from_tokens(
        &mut self,
        tokens: &[u32],
        params: &GenerateParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<GenerationStats> {
        let t0 = Instant::now();
        let reused = self.eval_tokens(tokens)?;
        let prompt_time = t0.elapsed();
        let mut sampler = Sampler::new(params.sampler.clone());
        let mut utf8 = Utf8Stream::default();
        let mut pending = String::new(); // text held back while it may start a stop string
        let mut full = String::new();
        let max_stop = params.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);
        let t1 = Instant::now();
        let mut n_gen = 0;
        let mut stop = StopReason::MaxTokens;
        let mut logits = self.logits.clone();
        'gen: while n_gen < params.max_tokens {
            logits.copy_from_slice(&self.logits);
            let history_start = self.cached.len().saturating_sub(params.sampler.repeat_last_n);
            let tok = sampler.sample(&mut logits, &self.cached[history_start..]);
            n_gen += 1;
            if self.tokenizer.is_stop(tok) {
                // The stop token is not evaluated; the next chat turn re-renders it as part
                // of the template and the prefix match resumes from there.
                stop = StopReason::EndOfSequence;
                break;
            }
            let piece = utf8.push(&self.tokenizer.decode_token(tok, params.show_special));
            pending.push_str(&piece);
            full.push_str(&piece);
            if max_stop > 0 {
                if let Some((i, _)) =
                    params.stop_strings.iter().filter_map(|s| full.rfind(s.as_str()).map(|i| (i, s))).min_by_key(|(i, _)| *i)
                {
                    // emit text before the stop string
                    let emitted = full.len() - pending.len();
                    if i > emitted {
                        on_text(&full[emitted..i]);
                    }
                    stop = StopReason::StopString;
                    let _ = self.eval_next(tok);
                    break 'gen;
                }
                // flush all but the tail that could still begin a stop string
                let mut safe = pending.len().saturating_sub(max_stop - 1);
                while !pending.is_char_boundary(safe) {
                    safe -= 1;
                }
                if safe > 0 {
                    let out: String = pending.drain(..safe).collect();
                    if !on_text(&out) {
                        stop = StopReason::Cancelled;
                        break;
                    }
                }
            } else if !pending.is_empty() {
                let out = std::mem::take(&mut pending);
                if !on_text(&out) {
                    stop = StopReason::Cancelled;
                    break;
                }
            }
            if self.cached.len() >= self.backend.n_ctx() {
                stop = StopReason::ContextFull;
                break;
            }
            self.eval_next(tok)?;
        }
        if stop != StopReason::StopString {
            pending.push_str(&utf8.flush());
            if !pending.is_empty() {
                on_text(&pending);
            }
        }
        Ok(GenerationStats {
            prompt_tokens: tokens.len(),
            reused_tokens: reused,
            generated_tokens: n_gen,
            prompt_time,
            gen_time: t1.elapsed(),
            stop,
        })
    }

    /// llama-bench style throughput measurement (pp = prompt processing, tg = generation).
    pub fn bench(&mut self, n_prompt: usize, n_gen: usize) -> Result<BenchResult> {
        let n_vocab = self.model.cfg.n_vocab as u32;
        let toks: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i as u32 * 7919 + 13) % n_vocab).collect();
        // warm-up (page in weights, spin up cores)
        self.reset();
        self.eval_tokens(&toks[..toks.len().min(8)])?;
        self.reset();
        let t0 = Instant::now();
        self.eval_tokens(&toks)?;
        let pp = t0.elapsed();
        let t1 = Instant::now();
        let mut tok = crate::sampler::argmax(&self.logits);
        for _ in 0..n_gen {
            self.eval_next(tok)?;
            tok = crate::sampler::argmax(&self.logits);
        }
        let tg = t1.elapsed();
        self.reset();
        Ok(BenchResult {
            n_prompt: toks.len(),
            n_gen,
            prompt_tps: toks.len() as f64 / pp.as_secs_f64(),
            gen_tps: if n_gen > 0 { n_gen as f64 / tg.as_secs_f64() } else { 0.0 },
        })
    }

    /// Perplexity of `text` over windows of `n_ctx` tokens (model quality check).
    pub fn perplexity(&mut self, text: &str, max_windows: usize) -> Result<f64> {
        let toks = self.tokenizer.encode(text, true, false);
        let n_ctx = self.backend.n_ctx().min(512);
        let nb = self.backend.max_batch();
        let nv = self.model.cfg.n_vocab;
        let mut nll = 0.0f64;
        let mut count = 0usize;
        let mut logits = vec![0.0f32; nb * nv];
        for w in toks.chunks(n_ctx).take(max_windows) {
            if w.len() < 2 {
                break;
            }
            self.reset();
            let mut pos = 0;
            while pos < w.len() {
                let end = (pos + nb).min(w.len());
                self.backend.forward(&w[pos..end], pos, true, &mut logits)?;
                for t in pos..end {
                    // score the second half of each window, like llama.cpp
                    if t + 1 < w.len() && t >= w.len() / 2 {
                        let l = &logits[(t - pos) * nv..(t - pos + 1) * nv];
                        let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let lse = l.iter().map(|v| ((v - max) as f64).exp()).sum::<f64>().ln() + max as f64;
                        nll += lse - l[w[t + 1] as usize] as f64;
                        count += 1;
                    }
                }
                pos = end;
            }
        }
        self.reset();
        if count == 0 {
            bail!(Runtime, "text too short for perplexity");
        }
        Ok((nll / count as f64).exp())
    }
}
