//! `rie` — run GGUF language models on phones and desktops.

use clap::{Args, Parser, Subcommand};
use rie_core::chat::{Message, Role};
use rie_core::cpu::CpuTopology;
use rie_core::engine::{load_model_from, make_pool, Engine, EngineConfig, GenerateParams};
use rie_core::gguf::GgufFile;
use rie_core::model::cpu::CpuBackend;
use rie_core::quant::GgmlType;
use rie_core::testing::SyntheticSpec;
use rie_core::{CpuFeatures, SamplerParams};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "rie", version, about = "Rust Inference Engine — fast LLM inference on mobile CPUs and GPUs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate text from a prompt.
    Run(RunArgs),
    /// Interactive chat using the model's chat template.
    Chat(ChatArgs),
    /// Measure prompt-processing and generation speed (llama-bench style).
    Bench(BenchArgs),
    /// Show model metadata.
    Info { model: PathBuf },
    /// Tokenizer efficiency on text files (fertility = tokens per word); works with
    /// vocab-only GGUF files too. Compare models before picking one for a language.
    Tokstats {
        model: PathBuf,
        /// Text files to measure.
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
    /// Show CPU topology, SIMD features and GPUs.
    Device,
    /// Write a synthetic model (random weights, real shapes) to a GGUF file for testing.
    Synth {
        /// Preset name (see `bench --synthetic`) or "tiny-<arch>".
        preset: String,
        #[arg(long, default_value = "Q4_0")]
        wtype: String,
        #[arg(short, long)]
        out: PathBuf,
        /// Training context length to record in the file.
        #[arg(long)]
        ctx: Option<usize>,
        /// Chat template to embed: chatml or llama3 (default: none).
        #[arg(long)]
        chat_template: Option<String>,
    },
    /// Re-quantize a model (e.g. F32/F16 → 4-bit Q4_0 or 8-bit Q8_0).
    Quantize {
        input: PathBuf,
        output: PathBuf,
        /// Target type: Q4_0, Q8_0 or F16.
        #[arg(long = "type", default_value = "Q4_0")]
        ty: String,
    },
    /// Convert a dense model into a mixture of experts (only the top experts run per
    /// token). Measure quality with `perplexity` afterwards.
    Moefy {
        input: PathBuf,
        output: PathBuf,
        /// Experts per layer.
        #[arg(long, default_value_t = 8)]
        experts: usize,
        /// Experts evaluated per token.
        #[arg(long, default_value_t = 2)]
        used: usize,
        /// Calibration text (target-domain sample); without it, a data-free router is used.
        #[arg(long)]
        calib: Option<PathBuf>,
        /// Max calibration tokens.
        #[arg(long, default_value_t = 512)]
        calib_tokens: usize,
        #[arg(short, long, default_value_t = 4)]
        threads: usize,
    },
    /// Compute perplexity over a text file.
    Perplexity {
        #[command(flatten)]
        common: Common,
        #[arg(short, long)]
        file: PathBuf,
        #[arg(long, default_value_t = 16)]
        windows: usize,
    },
}

#[derive(Args, Clone)]
struct Common {
    /// Path to a GGUF model.
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Worker threads (default: performance cores).
    #[arg(short, long)]
    threads: Option<usize>,
    /// Context window.
    #[arg(short = 'c', long, default_value_t = 4096)]
    ctx: usize,
    /// Prompt batch size.
    #[arg(short = 'b', long, default_value_t = 64)]
    batch: usize,
    /// Do not pin threads to performance cores.
    #[arg(long)]
    no_pin: bool,
    /// Run on the GPU (Vulkan / Metal) — requires building with `--features gpu`.
    #[arg(long)]
    gpu: bool,
    /// Lock model weights in RAM.
    #[arg(long)]
    mlock: bool,
    /// KV-cache format: f16, q8_0 (~47% less KV memory) or turbo4 (TurboQuant 4-bit,
    /// ~74% less on CPU; the GPU backend falls back to q8_0).
    #[arg(long, default_value = "f16")]
    kv: String,
    /// Stream layers from flash: run models larger than free RAM (slower, flash-bound).
    #[arg(long)]
    stream: bool,
}

#[derive(Args)]
struct SampleArgs {
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

impl SampleArgs {
    fn params(&self) -> SamplerParams {
        SamplerParams {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            seed: if self.seed == 0 {
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64
            } else {
                self.seed
            },
            ..Default::default()
        }
    }
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    sample: SampleArgs,
    /// Prompt text.
    #[arg(short, long)]
    prompt: String,
    /// Max tokens to generate.
    #[arg(short = 'n', long, default_value_t = 256)]
    n_predict: usize,
    /// Wrap the prompt in the model's chat template as a user message.
    #[arg(long)]
    chat: bool,
    /// Optional system prompt (with --chat).
    #[arg(long)]
    system: Option<String>,
}

#[derive(Args)]
struct ChatArgs {
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    sample: SampleArgs,
    #[arg(long)]
    system: Option<String>,
    #[arg(short = 'n', long, default_value_t = 1024)]
    n_predict: usize,
    /// Override the chat template (llama3, chatml, gemma, phi3, mistral, granite, raw).
    #[arg(long)]
    template: Option<String>,
}

#[derive(Args)]
struct BenchArgs {
    #[command(flatten)]
    common: Common,
    /// Benchmark a synthetic model with the shape of a popular model instead of a file.
    /// One of: smollm2-360m, qwen2.5-0.5b, qwen3-0.6b, gemma-3-1b, llama-3.2-1b,
    /// qwen2.5-1.5b, qwen3-1.7b, llama-3.2-3b, qwen3-4b, phi-4-mini.
    #[arg(long)]
    synthetic: Option<String>,
    /// Weight type for --synthetic (Q4_0, Q4_K, Q8_0, F16, …).
    #[arg(long, default_value = "Q4_0")]
    wtype: String,
    /// Prompt tokens.
    #[arg(short = 'p', long, default_value_t = 128)]
    n_prompt: usize,
    /// Generated tokens.
    #[arg(short = 'n', long, default_value_t = 64)]
    n_gen: usize,
    /// Repetitions.
    #[arg(short = 'r', long, default_value_t = 1)]
    reps: usize,
}

fn engine_config(c: &Common) -> EngineConfig {
    EngineConfig {
        n_threads: c.threads,
        n_ctx: c.ctx,
        n_batch: c.batch,
        pin_threads: !c.no_pin,
        mlock: c.mlock,
        kv_type: rie_core::KvType::parse(&c.kv).unwrap_or_else(|| {
            eprintln!("unknown --kv '{}', using f16", c.kv);
            rie_core::KvType::F16
        }),
        stream_layers: c.stream,
        ..Default::default()
    }
}

fn build_engine(c: &Common, gguf: Arc<GgufFile>) -> Result<Engine, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let (model, tok) = load_model_from(gguf)?;
    let cfg = engine_config(c);
    if cfg.mlock {
        model.gguf.lock_in_memory();
    }
    let n_ctx = cfg.n_ctx.min(model.cfg.ctx_train).max(16);
    let backend: Box<dyn rie_core::Backend> = if c.gpu {
        #[cfg(feature = "gpu")]
        {
            Box::new(rie_gpu::GpuBackend::new(
                model.clone(),
                rie_gpu::GpuConfig { n_ctx, max_batch: c.batch.min(64), kv_q8: cfg.kv_type != rie_core::KvType::F16, ..Default::default() },
            )?)
        }
        #[cfg(not(feature = "gpu"))]
        {
            return Err("this binary was built without GPU support (rebuild with --features gpu)".into());
        }
    } else {
        let mut b = CpuBackend::with_kv_type(model.clone(), make_pool(&cfg), n_ctx, cfg.n_batch, cfg.kv_type);
        b.set_streaming(cfg.stream_layers);
        Box::new(b)
    };
    let e = Engine::with_backend(model, tok, backend);
    eprintln!(
        "loaded {} ({}, {} layers, {:.2} GB weights, main type {}) in {:.2}s on {}",
        e.model.cfg.name.as_deref().unwrap_or("model"),
        e.model.cfg.arch_name,
        e.model.cfg.n_layer,
        e.model.weight_bytes() as f64 / 1e9,
        e.model.main_type().name(),
        t0.elapsed().as_secs_f64(),
        e.backend_name()
    );
    Ok(e)
}

fn open_model(c: &Common) -> Result<Arc<GgufFile>, Box<dyn std::error::Error>> {
    let path = c.model.as_ref().ok_or("--model is required")?;
    Ok(Arc::new(GgufFile::open(path)?))
}

fn print_stats(s: &rie_core::GenerationStats) {
    eprintln!(
        "\n[prompt: {} tok ({} cached) {:.1} tok/s | generation: {} tok {:.1} tok/s | stop: {:?}]",
        s.prompt_tokens,
        s.reused_tokens,
        s.prompt_tps(),
        s.generated_tokens,
        s.gen_tps(),
        s.stop
    );
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Device => {
            let topo = CpuTopology::detect();
            print!("{topo}");
            println!("SIMD: {}  ->  kernels: {}", CpuFeatures::get().describe(), CpuFeatures::get().kernel_name());
            println!("default threads: {}", topo.recommended_threads());
            #[cfg(feature = "gpu")]
            for a in rie_gpu::list_adapters() {
                println!("GPU: {a}");
            }
        }
        Cmd::Tokstats { model, files } => {
            let g = GgufFile::open(&model)?;
            let tok = rie_core::Tokenizer::from_gguf(&g)?;
            println!(
                "{:<28} {:>7} {:>7} {:>9} {:>11} {:>9} {:>10}",
                "file", "words", "tokens", "fertility", "chars/token", "partial", "round-trip"
            );
            for f in files {
                let text = std::fs::read_to_string(&f)?;
                let s = tok.stats(&text);
                let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                println!(
                    "{:<28} {:>7} {:>7} {:>9.2} {:>11.2} {:>8.0}% {:>10}",
                    name,
                    s.words,
                    s.tokens,
                    s.fertility(),
                    s.chars_per_token(),
                    100.0 * s.partial_tokens as f32 / s.tokens.max(1) as f32,
                    if s.round_trip { "ok" } else { "MISMATCH" }
                );
            }
        }
        Cmd::Info { model } => {
            let g = GgufFile::open(&model)?;
            let cfg = rie_core::ModelConfig::from_gguf(&g)?;
            println!("{cfg:#?}");
            println!("GGUF v{}, {} tensors", g.version, g.tensors.len());
            let mut types: Vec<(GgmlType, u64)> = Vec::new();
            for t in &g.tensors {
                match types.iter_mut().find(|(ty, _)| *ty == t.ty) {
                    Some((_, n)) => *n += t.size_bytes,
                    None => types.push((t.ty, t.size_bytes)),
                }
            }
            for (ty, n) in types {
                println!("  {:6} {:8.1} MB", ty.name(), n as f64 / 1e6);
            }
            if let Some(t) = g.get_str("tokenizer.ggml.model") {
                println!("tokenizer: {t} (pre: {})", g.get_str("tokenizer.ggml.pre").unwrap_or("-"));
            }
        }
        Cmd::Run(a) => {
            let mut e = build_engine(&a.common, open_model(&a.common)?)?;
            let params = GenerateParams { max_tokens: a.n_predict, sampler: a.sample.params(), ..Default::default() };
            let mut out = std::io::stdout();
            let cb = |s: &str| {
                print!("{s}");
                let _ = out.flush();
                true
            };
            let stats = if a.chat {
                let mut msgs = Vec::new();
                if let Some(s) = a.system {
                    msgs.push(Message::new(Role::System, s));
                }
                msgs.push(Message::new(Role::User, a.prompt));
                e.chat(&msgs, &params, cb)?
            } else {
                print!("{}", a.prompt);
                e.generate(&a.prompt, &params, cb)?
            };
            println!();
            print_stats(&stats);
        }
        Cmd::Chat(a) => {
            let mut e = build_engine(&a.common, open_model(&a.common)?)?;
            if let Some(t) = &a.template {
                e.template = rie_core::chat::ChatTemplate::from_name(t).ok_or("unknown template")?;
            }
            eprintln!("chat template: {:?}. Type /reset to clear, /quit to exit.", e.template);
            let params = GenerateParams { max_tokens: a.n_predict, sampler: a.sample.params(), ..Default::default() };
            let mut msgs = Vec::new();
            if let Some(s) = &a.system {
                msgs.push(Message::new(Role::System, s.clone()));
            }
            let stdin = std::io::stdin();
            loop {
                eprint!("\n> ");
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 {
                    break;
                }
                let line = line.trim();
                match line {
                    "" => continue,
                    "/quit" | "/exit" => break,
                    "/reset" => {
                        msgs.retain(|m: &Message| m.role == Role::System);
                        e.reset();
                        continue;
                    }
                    _ => {}
                }
                msgs.push(Message::new(Role::User, line));
                let mut reply = String::new();
                let stats = e.chat(&msgs, &params, |s| {
                    print!("{s}");
                    let _ = std::io::stdout().flush();
                    reply.push_str(s);
                    true
                })?;
                msgs.push(Message::new(Role::Assistant, reply));
                print_stats(&stats);
            }
        }
        Cmd::Bench(a) => {
            let gguf = match &a.synthetic {
                Some(name) => {
                    let wtype = parse_type(&a.wtype)?;
                    let spec = SyntheticSpec::preset(name, wtype)
                        .ok_or_else(|| format!("unknown preset '{name}' (try one of {:?})", SyntheticSpec::PRESETS))?;
                    eprintln!("building synthetic {name} ({:?} weights)…", spec.wtype);
                    Arc::new(GgufFile::from_bytes(spec.build())?)
                }
                None => open_model(&a.common)?,
            };
            let mut e = build_engine(&a.common, gguf)?;
            println!("| model | backend | test | t/s |");
            println!("|---|---|---|---|");
            let name = a.synthetic.clone().unwrap_or_else(|| e.model.cfg.name.clone().unwrap_or_default());
            let (mut pp, mut tg) = (Vec::new(), Vec::new());
            for _ in 0..a.reps.max(1) {
                let r = e.bench(a.n_prompt, a.n_gen)?;
                pp.push(r.prompt_tps);
                tg.push(r.gen_tps);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!("| {name} {} | {} | pp{} | {:.2} |", e.model.main_type().name(), e.backend_name(), a.n_prompt, mean(&pp));
            println!("| {name} {} | {} | tg{} | {:.2} |", e.model.main_type().name(), e.backend_name(), a.n_gen, mean(&tg));
        }
        Cmd::Synth { preset, wtype, out, ctx, chat_template } => {
            let wtype = parse_type(&wtype)?;
            let spec = match preset.strip_prefix("tiny-") {
                Some(arch) => SyntheticSpec::tiny(arch, wtype),
                None => SyntheticSpec::preset(&preset, wtype).ok_or("unknown preset")?,
            };
            let mut spec = spec;
            if let Some(c) = ctx {
                spec.ctx = c;
            }
            spec.chat_template = chat_template;
            std::fs::write(&out, spec.build())?;
            println!("wrote {}", out.display());
        }
        Cmd::Quantize { input, output, ty } => {
            let target = parse_type(&ty)?;
            let src = GgufFile::open(&input)?;
            let before = std::fs::metadata(&input)?.len();
            let img = rie_core::convert::quantize_model(&src, target)?;
            std::fs::write(&output, &img)?;
            println!("wrote {} ({:.1} MB → {:.1} MB, {})", output.display(), before as f64 / 1e6, img.len() as f64 / 1e6, target.name());
        }
        Cmd::Moefy { input, output, experts, used, calib, calib_tokens, threads } => {
            let g = Arc::new(GgufFile::open(&input)?);
            let calib = match calib {
                Some(p) => {
                    let (_, tok) = load_model_from(g.clone())?;
                    let mut t = tok.encode(&std::fs::read_to_string(p)?, true, false);
                    t.truncate(calib_tokens);
                    Some(t)
                }
                None => None,
            };
            let opts = rie_core::convert::MoefyOptions { n_expert: experts, n_used: used, calib, threads, ..Default::default() };
            let (img, rep) = rie_core::convert::moefy(g, &opts)?;
            std::fs::write(&output, &img)?;
            for r in &rep {
                match r.routed_mass {
                    Some(m) => println!(
                        "layer {:3}: {experts} experts × {} neurons, router keeps {:.1}% of activation mass",
                        r.layer,
                        r.expert_size,
                        100.0 * m
                    ),
                    None => println!("layer {:3}: {experts} experts × {} neurons (data-free router)", r.layer, r.expert_size),
                }
            }
            println!("wrote {}: feed-forward compute per token is now {used}/{experts} of the dense model", output.display());
        }
        Cmd::Perplexity { common, file, windows } => {
            let mut e = build_engine(&common, open_model(&common)?)?;
            let text = std::fs::read_to_string(file)?;
            let t0 = Instant::now();
            let ppl = e.perplexity(&text, windows)?;
            println!("perplexity: {ppl:.4} ({:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
    let _ = Duration::ZERO;
    Ok(())
}

fn parse_type(s: &str) -> Result<GgmlType, String> {
    let up = s.to_uppercase();
    for t in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        if t.name() == up {
            return Ok(t);
        }
    }
    Err(format!("unsupported weight type {s}"))
}
