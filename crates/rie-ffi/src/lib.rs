//! C ABI for embedding the engine in Android (via JNI, see `jni.rs`) and iOS (Swift via
//! `include/rie.h`) apps. All functions are thread-safe with respect to `rie_cancel`;
//! other calls on one engine must not run concurrently.

#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "jni")]
mod jni_bridge;

use rie_core::chat::{Message, Role};
use rie_core::engine::{load_model, make_pool, Engine, EngineConfig, GenerateParams};
use rie_core::model::cpu::CpuBackend;
use rie_core::{GenerationStats, SamplerParams, StopReason};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct RieEngine {
    pub(crate) engine: Engine,
    pub(crate) cancel: Arc<AtomicBool>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RieConfig {
    /// 0 = auto (performance cores).
    pub n_threads: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    /// Non-zero: use the GPU backend (Vulkan/Metal) if compiled in and available.
    pub use_gpu: i32,
    pub pin_threads: i32,
    pub mlock: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RieSampling {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
    pub max_tokens: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RieStats {
    pub prompt_tokens: i32,
    pub reused_tokens: i32,
    pub generated_tokens: i32,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    /// 0 end-of-sequence, 1 max tokens, 2 stop string, 3 context full, 4 cancelled.
    pub stop_reason: i32,
}

/// Streaming callback: receives UTF-8 text pieces; return 0 to stop generation.
pub type RieTokenCallback = Option<unsafe extern "C" fn(piece: *const c_char, user: *mut c_void) -> i32>;

#[no_mangle]
pub extern "C" fn rie_default_config() -> RieConfig {
    RieConfig { n_threads: 0, n_ctx: 4096, n_batch: 64, use_gpu: 0, pin_threads: 1, mlock: 0 }
}

#[no_mangle]
pub extern "C" fn rie_default_sampling() -> RieSampling {
    let d = SamplerParams::default();
    RieSampling {
        temperature: d.temperature,
        top_k: d.top_k as i32,
        top_p: d.top_p,
        min_p: d.min_p,
        repeat_penalty: d.repeat_penalty,
        seed: 0,
        max_tokens: 512,
    }
}

fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

pub(crate) fn load_engine(path: &str, cfg: &RieConfig) -> Result<RieEngine, String> {
    let (model, tok) = load_model(path).map_err(|e| e.to_string())?;
    let ecfg = EngineConfig {
        n_threads: (cfg.n_threads > 0).then_some(cfg.n_threads as usize),
        n_ctx: if cfg.n_ctx > 0 { cfg.n_ctx as usize } else { 4096 },
        n_batch: if cfg.n_batch > 0 { cfg.n_batch as usize } else { 64 },
        pin_threads: cfg.pin_threads != 0,
        mlock: cfg.mlock != 0,
        ..Default::default()
    };
    if ecfg.mlock {
        model.gguf.lock_in_memory();
    }
    let n_ctx = ecfg.n_ctx.min(model.cfg.ctx_train).max(16);
    #[cfg(feature = "gpu")]
    if cfg.use_gpu != 0 {
        match rie_gpu::GpuBackend::new(model.clone(), rie_gpu::GpuConfig { n_ctx, max_batch: ecfg.n_batch.min(32), ..Default::default() }) {
            Ok(b) => {
                return Ok(RieEngine { engine: Engine::with_backend(model, tok, Box::new(b)), cancel: Arc::new(AtomicBool::new(false)) })
            }
            Err(e) => eprintln!("rie: GPU backend unavailable ({e}), falling back to CPU"),
        }
    }
    let backend = CpuBackend::new(model.clone(), make_pool(&ecfg), n_ctx, ecfg.n_batch);
    Ok(RieEngine { engine: Engine::with_backend(model, tok, Box::new(backend)), cancel: Arc::new(AtomicBool::new(false)) })
}

/// Load a GGUF model. Returns NULL on failure and writes a message into `err` (if given).
#[no_mangle]
pub unsafe extern "C" fn rie_load(path: *const c_char, cfg: *const RieConfig, err: *mut c_char, err_len: usize) -> *mut RieEngine {
    if path.is_null() {
        write_err(err, err_len, "null path");
        return std::ptr::null_mut();
    }
    let path = match CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(_) => {
            write_err(err, err_len, "path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    let cfg = if cfg.is_null() { rie_default_config() } else { *cfg };
    match std::panic::catch_unwind(|| load_engine(path, &cfg)) {
        Ok(Ok(e)) => Box::into_raw(Box::new(e)),
        Ok(Err(m)) => {
            write_err(err, err_len, &m);
            std::ptr::null_mut()
        }
        Err(_) => {
            write_err(err, err_len, "internal error while loading");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn rie_free(e: *mut RieEngine) {
    if !e.is_null() {
        drop(Box::from_raw(e));
    }
}

/// Request cancellation of a running generation (callable from any thread).
#[no_mangle]
pub unsafe extern "C" fn rie_cancel(e: *const RieEngine) {
    if let Some(e) = e.as_ref() {
        e.cancel.store(true, Ordering::SeqCst);
    }
}

/// Clear the conversation / KV cache.
#[no_mangle]
pub unsafe extern "C" fn rie_reset(e: *mut RieEngine) {
    if let Some(e) = e.as_mut() {
        e.engine.reset();
    }
}

pub(crate) fn to_params(s: &RieSampling) -> GenerateParams {
    let seed = if s.seed == 0 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    } else {
        s.seed
    };
    GenerateParams {
        max_tokens: s.max_tokens.max(1) as usize,
        sampler: SamplerParams {
            temperature: s.temperature,
            top_k: s.top_k.max(0) as usize,
            top_p: s.top_p,
            min_p: s.min_p,
            repeat_penalty: s.repeat_penalty,
            seed,
            ..Default::default()
        },
        ..Default::default()
    }
}

pub(crate) fn to_stats(s: &GenerationStats) -> RieStats {
    RieStats {
        prompt_tokens: s.prompt_tokens as i32,
        reused_tokens: s.reused_tokens as i32,
        generated_tokens: s.generated_tokens as i32,
        prompt_tps: s.prompt_tps(),
        gen_tps: s.gen_tps(),
        stop_reason: match s.stop {
            StopReason::EndOfSequence => 0,
            StopReason::MaxTokens => 1,
            StopReason::StopString => 2,
            StopReason::ContextFull => 3,
            StopReason::Cancelled => 4,
        },
    }
}

fn emit(cb: RieTokenCallback, user: *mut c_void, cancel: &AtomicBool, s: &str) -> bool {
    if cancel.load(Ordering::Relaxed) {
        return false;
    }
    match cb {
        Some(f) => match CString::new(s.replace('\0', "")) {
            Ok(c) => unsafe { f(c.as_ptr(), user) != 0 },
            Err(_) => true,
        },
        None => true,
    }
}

/// Generate from a raw prompt. Returns 0 on success, -1 on error (message in `err`).
#[no_mangle]
pub unsafe extern "C" fn rie_generate(
    e: *mut RieEngine,
    prompt: *const c_char,
    sampling: *const RieSampling,
    cb: RieTokenCallback,
    user: *mut c_void,
    stats: *mut RieStats,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let Some(e) = e.as_mut() else { return -1 };
    let Ok(prompt) = CStr::from_ptr(prompt).to_str() else {
        write_err(err, err_len, "prompt is not valid UTF-8");
        return -1;
    };
    let params = to_params(&if sampling.is_null() { rie_default_sampling() } else { *sampling });
    e.cancel.store(false, Ordering::SeqCst);
    let cancel = e.cancel.clone();
    match e.engine.generate(prompt, &params, |s| emit(cb, user, &cancel, s)) {
        Ok(s) => {
            if !stats.is_null() {
                *stats = to_stats(&s);
            }
            0
        }
        Err(m) => {
            write_err(err, err_len, &m.to_string());
            -1
        }
    }
}

pub(crate) fn parse_role(s: &str) -> Role {
    Role::parse(s).unwrap_or(Role::User)
}

/// Chat completion over `n` messages (`roles[i]` is "system", "user" or "assistant").
/// The KV cache is reused for the shared prefix with the previous call.
#[no_mangle]
pub unsafe extern "C" fn rie_chat(
    e: *mut RieEngine,
    roles: *const *const c_char,
    contents: *const *const c_char,
    n: i32,
    sampling: *const RieSampling,
    cb: RieTokenCallback,
    user: *mut c_void,
    stats: *mut RieStats,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let Some(e) = e.as_mut() else { return -1 };
    let mut msgs = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n.max(0) as usize {
        let r = CStr::from_ptr(*roles.add(i)).to_string_lossy();
        let c = CStr::from_ptr(*contents.add(i)).to_string_lossy().into_owned();
        msgs.push(Message::new(parse_role(&r), c));
    }
    let params = to_params(&if sampling.is_null() { rie_default_sampling() } else { *sampling });
    e.cancel.store(false, Ordering::SeqCst);
    let cancel = e.cancel.clone();
    match e.engine.chat(&msgs, &params, |s| emit(cb, user, &cancel, s)) {
        Ok(s) => {
            if !stats.is_null() {
                *stats = to_stats(&s);
            }
            0
        }
        Err(m) => {
            write_err(err, err_len, &m.to_string());
            -1
        }
    }
}

/// Throughput benchmark. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn rie_bench(e: *mut RieEngine, n_prompt: i32, n_gen: i32, pp_tps: *mut f64, tg_tps: *mut f64) -> i32 {
    let Some(e) = e.as_mut() else { return -1 };
    match e.engine.bench(n_prompt.max(1) as usize, n_gen.max(0) as usize) {
        Ok(r) => {
            if !pp_tps.is_null() {
                *pp_tps = r.prompt_tps;
            }
            if !tg_tps.is_null() {
                *tg_tps = r.gen_tps;
            }
            0
        }
        Err(_) => -1,
    }
}

pub(crate) fn device_info_string() -> String {
    let topo = rie_core::cpu::CpuTopology::detect();
    #[allow(unused_mut)]
    let mut s =
        format!("{topo}SIMD: {} (kernels: {})\n", rie_core::CpuFeatures::get().describe(), rie_core::CpuFeatures::get().kernel_name());
    #[cfg(feature = "gpu")]
    for a in rie_gpu::list_adapters() {
        s += &format!("GPU: {a}\n");
    }
    s
}

/// Human-readable SoC / SIMD / GPU summary. Free with `rie_string_free`.
#[no_mangle]
pub extern "C" fn rie_device_info() -> *mut c_char {
    CString::new(device_info_string()).unwrap_or_default().into_raw()
}

/// Description of the loaded model and backend. Free with `rie_string_free`.
#[no_mangle]
pub unsafe extern "C" fn rie_model_info(e: *const RieEngine) -> *mut c_char {
    let Some(e) = e.as_ref() else { return std::ptr::null_mut() };
    let m = &e.engine.model;
    let s = format!(
        "{} | arch {} | {} layers | {:.2} GB {} | ctx {} | {}",
        m.cfg.name.as_deref().unwrap_or("model"),
        m.cfg.arch_name,
        m.cfg.n_layer,
        m.weight_bytes() as f64 / 1e9,
        m.main_type().name(),
        e.engine.n_ctx(),
        e.engine.backend_name()
    );
    CString::new(s).unwrap_or_default().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn rie_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
