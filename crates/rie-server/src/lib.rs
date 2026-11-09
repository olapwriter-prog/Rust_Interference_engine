//! # rie-server
//!
//! OpenAI-compatible HTTP server for the Rust Inference Engine, built to be driven by
//! tool-using agent harnesses (stitaP tool-harness, or anything that speaks
//! `/v1/chat/completions` with `tools`):
//!
//! * **Native tool calling** in each model family's own format (Hermes/Qwen `<tool_call>`,
//!   Llama 3 JSON), with **grammar-constrained decoding**: once a call starts, only tokens
//!   that keep it a valid call of a declared tool with declared argument names and types
//!   can be sampled. `tool_choice: "required"` / a named function force a call.
//! * **Online reinforcement learning** (`--rl`, `--adapter FILE`): a mixture of low-rank
//!   correction experts on the output head learns from `POST /v1/feedback`
//!   (rewards), `POST /v1/corrections` (the right answer) and, with `--auto-reward`,
//!   from tool results the harness sends back (failed tool → negative reward).
//! * **Drop-in for llama-server**: accepts llama.cpp's command-line flags and serves
//!   `/health`, `/props` and llama.cpp-style `timings`, so harnesses that launch
//!   `llama-server` themselves can launch `rie-server` instead.

pub mod http;
pub mod toolbook;
pub mod tools;

use rie_core::constrain::{ArgType, ToolSig};
use rie_core::engine::{load_model, make_pool, Engine, EngineConfig, GenerateParams};
use rie_core::memory::{Kind, MemoryStore};
use rie_core::model::cpu::{CpuBackend, KvType};
use rie_core::rl::{AdapterConfig, Step};
use rie_core::SamplerParams;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use toolbook::ToolBook;
use tools::{extract, parse_messages, parse_tools, render, ApiMessage, CallFormat, ToolCallFilter};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model: PathBuf,
    pub alias: Option<String>,
    pub host: String,
    pub port: u16,
    pub n_ctx: usize,
    pub n_batch: usize,
    pub threads: Option<usize>,
    pub kv: KvType,
    pub mlock: bool,
    /// Stream layers from flash (models larger than RAM).
    pub stream: bool,
    pub gpu: bool,
    /// Grammar-constrain tool calls (default on).
    pub constrain: bool,
    /// Enable online learning.
    pub rl: bool,
    /// Persist the learned adapter here (implies `rl`).
    pub adapter: Option<PathBuf>,
    /// Derive rewards from tool results sent back by the client.
    pub auto_reward: bool,
    pub rl_cfg: AdapterConfig,
    /// Reward for a tool result that looks successful (auto-reward).
    pub success_reward: f32,
    /// Maximum bytes of one string value inside a constrained tool call / JSON reply.
    pub max_string: usize,
    /// Treat `tool_choice: "auto"` (or absent) as "required" when tools are offered.
    /// For very small models that never decide to call a tool on their own (and tests).
    pub force_tools: bool,
    /// Soft byte budget for one constrained tool call / JSON reply (then it winds down).
    pub max_call: usize,
    /// On-device memory + tool book directory (observations journal, tool cards).
    pub memory: Option<PathBuf>,
    /// Observations kept before the least-retained are forgotten.
    pub memory_size: usize,
    /// Observations recalled into the context per request (0 = only via the tool).
    pub recall: usize,
    /// Offer the built-in `recall_memory` tool (executed inside the server).
    pub memory_tool: bool,
    /// Show only the K tools most relevant to the request (0 = all).
    pub tool_recall: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model: PathBuf::new(),
            alias: None,
            host: "127.0.0.1".into(),
            port: 8080,
            n_ctx: 4096,
            n_batch: 64,
            threads: None,
            kv: KvType::F16,
            mlock: false,
            stream: false,
            gpu: false,
            constrain: true,
            rl: false,
            adapter: None,
            auto_reward: false,
            rl_cfg: AdapterConfig::default(),
            success_reward: 0.3,
            max_string: 4096,
            force_tools: false,
            max_call: 16384,
            memory: None,
            memory_size: 20000,
            recall: 3,
            memory_tool: true,
            tool_recall: 0,
        }
    }
}

/// What we remember about a completion so feedback can be applied later.
struct Record {
    prompt: Vec<u32>,
    steps: Vec<Step>,
    /// first step belonging to the tool call (if any)
    call_from: Option<usize>,
    tool_call_id: Option<String>,
    format: CallFormat,
    rewarded: bool,
}

pub struct State {
    pub engine: Engine,
    pub cfg: ServerConfig,
    alias: String,
    records: HashMap<String, Record>,
    order: VecDeque<String>,
    /// tool_call_id → completion id
    call_index: HashMap<String, String>,
    counter: u64,
    /// on-device episodic memory (with `--memory DIR`)
    pub memory: Option<MemoryStore>,
    /// harvested tools and what was learned about them (persisted with `--memory DIR`)
    pub toolbook: ToolBook,
    /// tool_call_ids already learned from (conversations are resent with every request)
    observed: HashSet<String>,
    observed_order: VecDeque<String>,
}

/// Name of the built-in memory tool.
pub const RECALL_TOOL: &str = "recall_memory";
const MAX_OBSERVED: usize = 4096;

fn recall_tool() -> (Value, ToolSig) {
    let schema = json!({"type": "function", "function": {"name": RECALL_TOOL,
        "description": "Search your long-term memory on this device: earlier requests, actions you took, tool results and errors, and notes. Use it when something may have come up before.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look for"}}, "required": ["query"]}}});
    (
        schema,
        ToolSig { name: RECALL_TOOL.into(), args: vec![("query".into(), ArgType::String)], required: vec!["query".into()], enums: vec![] },
    )
}

fn clip(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

const MAX_RECORDS: usize = 16;
const MAX_RECORD_STEPS: usize = 1024;

impl State {
    pub fn new(engine: Engine, cfg: ServerConfig) -> Self {
        let alias = cfg
            .alias
            .clone()
            .unwrap_or_else(|| cfg.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "rie".into()));
        let memory = cfg.memory.as_ref().and_then(|d| match MemoryStore::open(d, cfg.memory_size) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("rie-server: memory disabled, cannot open {}: {e}", d.display());
                None
            }
        });
        let toolbook = cfg.memory.as_ref().map(|d| ToolBook::open(d.clone())).unwrap_or_else(ToolBook::in_memory);
        State {
            engine,
            cfg,
            alias,
            records: HashMap::new(),
            order: VecDeque::new(),
            call_index: HashMap::new(),
            counter: 0,
            memory,
            toolbook,
            observed: HashSet::new(),
            observed_order: VecDeque::new(),
        }
    }

    /// Learn from an incoming conversation: harvest tool definitions, attribute every new
    /// tool result to the call that produced it, and record observations.
    fn learn(&mut self, msgs: &[ApiMessage], schemas: &[Value]) {
        let mut changed = self.toolbook.harvest(schemas);
        let mut calls: HashMap<&str, (&str, &str)> = HashMap::new();
        for m in msgs.iter().filter(|m| m.role == "assistant") {
            for c in &m.tool_calls {
                calls.insert(c.id.as_str(), (c.name.as_str(), c.arguments.as_str()));
            }
        }
        for m in msgs.iter().filter(|m| m.role == "tool") {
            let Some(id) = &m.tool_call_id else { continue };
            if self.observed.contains(id) {
                continue;
            }
            let Some((name, args)) = calls.get(id.as_str()).copied().or_else(|| m.name.as_deref().map(|n| (n, ""))) else { continue };
            if name == RECALL_TOOL {
                continue;
            }
            self.observed.insert(id.clone());
            self.observed_order.push_back(id.clone());
            if self.observed_order.len() > MAX_OBSERVED {
                if let Some(old) = self.observed_order.pop_front() {
                    self.observed.remove(&old);
                }
            }
            let ok = !tool_result_failed(&m.content);
            self.toolbook.observe(name, args, &m.content, ok);
            changed = true;
            if let Some(mem) = &mut self.memory {
                let text = format!("{name}({}) → {}", clip(args, 300), clip(m.content.trim(), 600));
                mem.record(Kind::Result, name, Some(ok), if ok { 0.3 } else { 0.7 }, &text);
            }
        }
        if let (Some(mem), Some(u)) = (&mut self.memory, msgs.iter().rev().find(|m| m.role == "user")) {
            mem.record(Kind::User, "", None, 0.3, &clip(&u.content, 1500));
        }
        if changed {
            self.toolbook.save();
        }
    }

    /// Context recalled for this request: relevant observations and learned tool notes.
    fn recalled_context(&mut self, msgs: &[ApiMessage], query: &str, offered: &[String]) -> Option<String> {
        let mut parts = Vec::new();
        if let (Some(mem), true) = (&mut self.memory, self.cfg.recall > 0) {
            let in_context = |t: &str| {
                let key: String = t.chars().take(60).collect();
                msgs.iter().any(|m| m.content.contains(&key))
            };
            let hits: Vec<String> = mem
                .recall(query, self.cfg.recall + 4, None, 0.3)
                .into_iter()
                .filter(|r| !in_context(&r.obs.text))
                .take(self.cfg.recall)
                .map(|r| {
                    let ok = match r.obs.ok {
                        Some(true) => " ok",
                        Some(false) => " FAILED",
                        None => "",
                    };
                    format!("- [{}{ok}] {}", r.obs.kind.name(), clip(&r.obs.text.replace('\n', " "), 300))
                })
                .collect();
            if !hits.is_empty() {
                parts.push(format!("Recalled from earlier sessions:\n{}", hits.join("\n")));
            }
        }
        let names: Vec<String> = offered.iter().filter(|n| n.as_str() != RECALL_TOOL).cloned().collect();
        let notes = self.toolbook.notes(&names);
        if !notes.is_empty() {
            parts.push(format!("What you learned about these tools:\n{notes}"));
        }
        (!parts.is_empty()).then(|| parts.join("\n\n"))
    }

    fn next_id(&mut self, prefix: &str) -> String {
        self.counter += 1;
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
        format!("{prefix}{t:x}{:x}", self.counter)
    }

    fn remember(&mut self, id: String, rec: Record) {
        if let Some(c) = &rec.tool_call_id {
            self.call_index.insert(c.clone(), id.clone());
        }
        self.records.insert(id.clone(), rec);
        self.order.push_back(id);
        while self.order.len() > MAX_RECORDS {
            if let Some(old) = self.order.pop_front() {
                if let Some(r) = self.records.remove(&old) {
                    if let Some(c) = r.tool_call_id {
                        self.call_index.remove(&c);
                    }
                }
            }
        }
    }

    fn save_adapter(&self) {
        if let (Some(path), Some(a)) = (&self.cfg.adapter, &self.engine.adapter) {
            if let Err(e) = a.save(path) {
                eprintln!("rie-server: could not save adapter: {e}");
            }
        }
    }

    /// Apply `reward` to a stored completion (`scope`: all | tool_call | content).
    fn reinforce(&mut self, id: &str, reward: f32, scope: &str, epochs: usize) -> Result<usize, String> {
        let rec = self.records.get_mut(id).ok_or_else(|| format!("unknown completion id {id} (only the last {MAX_RECORDS} are kept)"))?;
        let steps: &[Step] = match (scope, rec.call_from) {
            ("tool_call", Some(i)) => &rec.steps[i.min(rec.steps.len())..],
            ("tool_call", None) => return Err("completion has no tool call".into()),
            ("content", Some(i)) => &rec.steps[..i.min(rec.steps.len())],
            _ => &rec.steps,
        };
        let n = steps.len();
        let adapter = self.engine.adapter.as_mut().ok_or("online learning is disabled (start with --rl or --adapter FILE)")?;
        adapter.reinforce(steps, reward.clamp(-1.0, 1.0), epochs);
        rec.rewarded = true;
        Ok(n)
    }

    /// Auto-reward: look at tool results in an incoming conversation.
    fn auto_reward(&mut self, msgs: &[ApiMessage]) {
        if !self.cfg.auto_reward || self.engine.adapter.is_none() {
            return;
        }
        for m in msgs.iter().filter(|m| m.role == "tool") {
            let Some(call) = &m.tool_call_id else { continue };
            let Some(id) = self.call_index.get(call).cloned() else { continue };
            if self.records.get(&id).is_none_or(|r| r.rewarded) {
                continue;
            }
            let reward = if tool_result_failed(&m.content) { -1.0 } else { self.cfg.success_reward };
            if let Ok(n) = self.reinforce(&id, reward, "tool_call", 2) {
                eprintln!("rie-server: auto-reward {reward:+.1} for {call} ({n} steps)");
            }
        }
        self.save_adapter();
    }
}

/// Heuristic: does a tool result report a failure?
pub fn tool_result_failed(content: &str) -> bool {
    let head: String = content.chars().take(300).collect::<String>().to_lowercase();
    let t = head.trim_start();
    t.starts_with("error")
        || t.starts_with("{\"error\"")
        || t.starts_with("{\"ok\":false")
        || t.starts_with("{\"success\":false")
        || head.contains("traceback (most recent call last)")
        || head.contains("exception:")
        || head.contains("unknown tool")
        || head.contains("invalid argument")
        || head.contains("missing required")
        || head.contains("command not found")
        || head.contains("no such file")
}

/// Load the model and build the engine according to `cfg`.
pub fn build_engine(cfg: &ServerConfig) -> Result<Engine, String> {
    let (model, tok) = load_model(&cfg.model).map_err(|e| e.to_string())?;
    if cfg.mlock {
        model.gguf.lock_in_memory();
    }
    let n_ctx = cfg.n_ctx.min(model.cfg.ctx_train).max(64);
    let wants_rl = cfg.rl || cfg.adapter.is_some();
    #[cfg(feature = "gpu")]
    if cfg.gpu && !wants_rl {
        match rie_gpu::GpuBackend::new(
            model.clone(),
            rie_gpu::GpuConfig { n_ctx, max_batch: cfg.n_batch.min(32), kv_q8: cfg.kv != KvType::F16, ..Default::default() },
        ) {
            Ok(b) => return Ok(Engine::with_backend(model, tok, Box::new(b))),
            Err(e) => eprintln!("rie-server: GPU unavailable ({e}); using the CPU"),
        }
    }
    if cfg.gpu && wants_rl {
        eprintln!("rie-server: online learning needs the CPU backend; ignoring --gpu");
    }
    let ecfg = EngineConfig {
        n_threads: cfg.threads,
        n_ctx,
        n_batch: cfg.n_batch,
        kv_type: cfg.kv,
        stream_layers: cfg.stream,
        ..Default::default()
    };
    let mut backend = CpuBackend::with_kv_type(model.clone(), make_pool(&ecfg), n_ctx, ecfg.n_batch, ecfg.kv_type);
    backend.set_streaming(ecfg.stream_layers);
    let mut engine = Engine::with_backend(model, tok, Box::new(backend));
    if let Some(p) = &cfg.adapter {
        engine.load_adapter(p, cfg.rl_cfg.clone()).map_err(|e| e.to_string())?;
    } else if cfg.rl {
        engine.enable_adapter(cfg.rl_cfg.clone()).map_err(|e| e.to_string())?;
    }
    Ok(engine)
}

/// Run the server forever.
pub fn serve(cfg: ServerConfig) -> Result<(), String> {
    let t0 = Instant::now();
    let engine = build_engine(&cfg)?;
    eprintln!(
        "rie-server: loaded {} in {:.1}s on {} | ctx {} | tools: {} | online learning: {}",
        cfg.model.display(),
        t0.elapsed().as_secs_f64(),
        engine.backend_name(),
        engine.n_ctx(),
        if cfg.constrain { "grammar-constrained" } else { "unconstrained" },
        match (&engine.adapter, cfg.auto_reward) {
            (Some(_), true) => "on (+auto-reward)",
            (Some(_), false) => "on",
            _ => "off",
        }
    );
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port)).map_err(|e| format!("bind {}:{}: {e}", cfg.host, cfg.port))?;
    eprintln!("rie-server: listening on http://{}:{}", cfg.host, listener.local_addr().map(|a| a.port()).unwrap_or(cfg.port));
    serve_on(listener, Arc::new(Mutex::new(State::new(engine, cfg))));
    Ok(())
}

/// Serve on an existing listener (used by tests).
pub fn serve_on(listener: TcpListener, state: Arc<Mutex<State>>) {
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let st = state.clone();
        std::thread::spawn(move || {
            let _ = handle_conn(stream, st);
        });
    }
}

fn handle_conn(stream: TcpStream, state: Arc<Mutex<State>>) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut w = stream;
    while let Some(req) = http::read_request(&mut reader)? {
        let close = req.header("connection").is_some_and(|v| v.eq_ignore_ascii_case("close"));
        let keep = route(&req, &mut w, &state)?;
        if !keep || close {
            let _ = w.shutdown(std::net::Shutdown::Both);
            break;
        }
    }
    Ok(())
}

/// Lock the shared state, recovering from a poisoned mutex (a panic in one request must
/// not take the whole server down).
fn lock(state: &Arc<Mutex<State>>) -> std::sync::MutexGuard<'_, State> {
    state.lock().unwrap_or_else(|e| e.into_inner())
}

fn err_json(msg: &str, kind: &str) -> String {
    json!({"error": {"message": msg, "type": kind}}).to_string()
}

/// Returns whether the connection can be kept alive.
fn route(req: &http::Request, w: &mut TcpStream, state: &Arc<Mutex<State>>) -> std::io::Result<bool> {
    let path = req.path.split('?').next().unwrap_or("");
    let body: Value = if req.body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => {
                http::write_json(w, 400, &err_json(&format!("invalid JSON body: {e}"), "invalid_request_error"))?;
                return Ok(true);
            }
        }
    };
    match (req.method.as_str(), path) {
        ("GET", "/health") | ("GET", "/v1/health") => http::write_json(w, 200, r#"{"status":"ok"}"#)?,
        ("GET", "/props") => {
            let st = lock(state);
            let v = json!({
                "default_generation_settings": {"n_ctx": st.engine.n_ctx(), "model": st.alias},
                "total_slots": 1,
                "model_path": st.cfg.model.display().to_string(),
                "chat_template": format!("{:?}", st.engine.template),
                "build_info": format!("rie-server {}", env!("CARGO_PKG_VERSION")),
            });
            http::write_json(w, 200, &v.to_string())?
        }
        ("GET", "/v1/models") | ("GET", "/models") => {
            let st = lock(state);
            let v = json!({"object": "list", "data": [{"id": st.alias, "object": "model", "owned_by": "rie", "meta": {"n_ctx_train": st.engine.model.cfg.ctx_train, "n_vocab": st.engine.model.cfg.n_vocab}}]});
            http::write_json(w, 200, &v.to_string())?
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => return chat(&body, w, state),
        ("POST", "/v1/completions") | ("POST", "/completion") => return completion(&body, w, state),
        ("POST", "/v1/feedback") => {
            let id = body.get("id").and_then(Value::as_str).unwrap_or("");
            let reward = body.get("reward").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let scope = body.get("scope").and_then(Value::as_str).unwrap_or("all");
            let epochs = body.get("epochs").and_then(Value::as_u64).unwrap_or(2) as usize;
            let mut st = lock(state);
            match st.reinforce(id, reward, scope, epochs.clamp(1, 16)) {
                Ok(n) => {
                    st.save_adapter();
                    http::write_json(w, 200, &json!({"ok": true, "steps": n}).to_string())?
                }
                Err(e) => http::write_json(w, 400, &err_json(&e, "invalid_request_error"))?,
            }
        }
        ("POST", "/v1/corrections") => {
            let mut st = lock(state);
            match correction(&mut st, &body) {
                Ok(v) => {
                    st.save_adapter();
                    http::write_json(w, 200, &v.to_string())?
                }
                Err(e) => http::write_json(w, 400, &err_json(&e, "invalid_request_error"))?,
            }
        }
        ("GET", "/v1/rl") => {
            let st = lock(state);
            let v = match &st.engine.adapter {
                Some(a) => {
                    json!({"enabled": true, "experts": a.cfg.n_experts, "rank": a.cfg.rank, "updates": a.stats.updates, "rows": a.stats.rows,
                    "feedback_events": a.stats.reward_count, "mean_reward": if a.stats.reward_count > 0 { a.stats.reward_sum / a.stats.reward_count as f64 } else { 0.0 },
                    "auto_reward": st.cfg.auto_reward, "adapter_file": st.cfg.adapter.as_ref().map(|p| p.display().to_string())})
                }
                None => json!({"enabled": false}),
            };
            http::write_json(w, 200, &v.to_string())?
        }
        ("POST", "/v1/rl/reset") => {
            let mut st = lock(state);
            if let Some(a) = st.engine.adapter.as_mut() {
                a.reset();
            }
            st.save_adapter();
            http::write_json(w, 200, r#"{"ok":true}"#)?
        }
        ("GET", "/v1/memory") => {
            let mut st = lock(state);
            let Some(mem) = st.memory.as_mut() else {
                http::write_json(w, 400, &err_json("memory is disabled (start with --memory DIR)", "invalid_request_error"))?;
                return Ok(true);
            };
            match http::query_param(&req.path, "q") {
                Some(q) => {
                    let k = http::query_param(&req.path, "k").and_then(|v| v.parse().ok()).unwrap_or(5usize).min(50);
                    let hits: Vec<Value> = mem
                        .recall(&q, k, None, 0.0)
                        .into_iter()
                        .map(|r| json!({"id": r.obs.id, "score": r.score, "kind": r.obs.kind.name(), "tool": r.obs.tool, "ok": r.obs.ok, "ts": r.obs.ts, "recalls": r.obs.recalls, "text": r.obs.text}))
                        .collect();
                    http::write_json(w, 200, &json!({"query": q, "results": hits}).to_string())?
                }
                None => {
                    let n = http::query_param(&req.path, "n").and_then(|v| v.parse().ok()).unwrap_or(200usize);
                    http::write_text(w, 200, "text/markdown; charset=utf-8", &mem.to_markdown(n))?
                }
            }
        }
        ("POST", "/v1/memory") => {
            let mut st = lock(state);
            let Some(mem) = st.memory.as_mut() else {
                http::write_json(w, 400, &err_json("memory is disabled (start with --memory DIR)", "invalid_request_error"))?;
                return Ok(true);
            };
            let text = body.get("text").and_then(Value::as_str).unwrap_or("");
            let kind = body.get("kind").and_then(Value::as_str).and_then(Kind::parse).unwrap_or(Kind::Note);
            let tool = body.get("tool").and_then(Value::as_str).unwrap_or("");
            let importance = body.get("importance").and_then(Value::as_f64).unwrap_or(0.8) as f32;
            let id = mem.record(kind, tool, body.get("ok").and_then(Value::as_bool), importance, text);
            http::write_json(w, 200, &json!({"ok": true, "id": id, "stored": id.is_some(), "count": mem.len()}).to_string())?
        }
        ("GET", "/v1/tools") => {
            let st = lock(state);
            http::write_json(w, 200, &json!({"tools": st.toolbook.to_json()}).to_string())?
        }
        ("GET", p) if p.starts_with("/v1/tools/") => {
            let st = lock(state);
            match st.toolbook.tools.get(&p["/v1/tools/".len()..]) {
                Some(e) => http::write_text(w, 200, "text/markdown; charset=utf-8", &e.card())?,
                None => http::write_json(w, 404, &err_json("unknown tool", "not_found"))?,
            }
        }
        ("OPTIONS", _) => {
            use std::io::Write;
            write!(w, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\n\r\n")?;
        }
        _ => http::write_json(w, 404, &err_json(&format!("no route for {} {}", req.method, path), "not_found"))?,
    }
    Ok(true)
}

fn sampling(body: &Value) -> SamplerParams {
    let d = SamplerParams::default();
    let f = |k: &str, def: f32| body.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(def);
    SamplerParams {
        temperature: f("temperature", 0.7),
        top_k: body.get("top_k").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d.top_k),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        repeat_penalty: f("repeat_penalty", 1.0),
        frequency_penalty: f("frequency_penalty", 0.0),
        presence_penalty: f("presence_penalty", 0.0),
        seed: body
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)),
        ..d
    }
}

fn stops(body: &Value) -> Vec<String> {
    match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        _ => Vec::new(),
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn timings(s: &rie_core::GenerationStats) -> Value {
    let pn = s.prompt_tokens - s.reused_tokens;
    json!({
        "cache_n": s.reused_tokens,
        "prompt_n": pn,
        "prompt_ms": s.prompt_time.as_secs_f64() * 1000.0,
        "prompt_per_second": s.prompt_tps(),
        "predicted_n": s.generated_tokens,
        "predicted_ms": s.gen_time.as_secs_f64() * 1000.0,
        "predicted_per_second": s.gen_tps(),
    })
}

/// Visible part of streamed text: everything before a tool call starts, minus a tail that
/// could still turn into the call trigger.
fn visible_prefix(text: &str, format: CallFormat, tools: bool) -> &str {
    if !tools {
        return text;
    }
    match format {
        CallFormat::Hermes => {
            const T: &str = "<tool_call>";
            if let Some(i) = text.find(T) {
                return &text[..i];
            }
            for k in (1..T.len()).rev() {
                if text.ends_with(&T[..k]) {
                    return &text[..text.len() - k];
                }
            }
            text
        }
        CallFormat::Llama3 => {
            let t = text.trim_start();
            if t.is_empty() || t.starts_with('{') {
                ""
            } else {
                text
            }
        }
    }
}

fn chat(body: &Value, w: &mut TcpStream, state: &Arc<Mutex<State>>) -> std::io::Result<bool> {
    let mut msgs = parse_messages(body.get("messages").unwrap_or(&Value::Null));
    if msgs.is_empty() {
        http::write_json(w, 400, &err_json("messages must be a non-empty array", "invalid_request_error"))?;
        return Ok(true);
    }
    let (mut schemas, mut sigs) = parse_tools(body.get("tools"));
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let json_mode = body.pointer("/response_format/type").and_then(Value::as_str).is_some_and(|t| t == "json_object" || t == "json_schema");
    // tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name":…}}
    let mut named = false;
    let (use_tools, forced) = match body.get("tool_choice") {
        Some(Value::String(s)) if s == "none" => (false, false),
        Some(Value::String(s)) if s == "required" => (true, true),
        Some(Value::Object(o)) => {
            if let Some(n) = o.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) {
                sigs.retain(|t| t.name == n);
                named = true;
            }
            (true, true)
        }
        _ => (true, false),
    };
    let use_tools = use_tools && !sigs.is_empty();

    let mut st = lock(state);
    let forced = forced || (use_tools && st.cfg.force_tools);
    st.auto_reward(&msgs);
    st.learn(&msgs, &schemas);
    // what this request is about: the latest user message and the latest tool result
    let query = {
        let u = msgs.iter().rev().find(|m| m.role == "user").map(|m| m.content.as_str()).unwrap_or("");
        let t = msgs.last().filter(|m| m.role == "tool").map(|m| m.content.as_str()).unwrap_or("");
        format!("{} {}", clip(u, 1000), clip(t, 500))
    };
    // tool recall: only show the tools relevant to this request (plus those already used)
    if use_tools && !named && st.cfg.tool_recall > 0 && sigs.len() > st.cfg.tool_recall {
        let names: Vec<String> = sigs.iter().map(|t| t.name.clone()).collect();
        let mut keep: HashSet<String> = st.toolbook.relevant(&query, &names, st.cfg.tool_recall).into_iter().collect();
        msgs.iter().flat_map(|m| &m.tool_calls).for_each(|c| {
            keep.insert(c.name.clone());
        });
        sigs.retain(|t| keep.contains(&t.name));
        schemas.retain(|s| s.pointer("/function/name").and_then(Value::as_str).is_some_and(|n| keep.contains(n)));
    }
    let memory_tool = use_tools && !named && st.cfg.memory_tool && st.memory.is_some();
    if memory_tool {
        let (sc, sg) = recall_tool();
        schemas.push(sc);
        sigs.push(sg);
    }
    let offered: Vec<String> = if use_tools { sigs.iter().map(|t| t.name.clone()).collect() } else { Vec::new() };
    if let Some(ctx) = st.recalled_context(&msgs, &query, &offered) {
        if let Some(m) = msgs.iter_mut().rev().find(|m| m.role == "user" || m.role == "tool") {
            m.content.push_str(&format!("\n\n<context>\n{ctx}\n</context>"));
        }
    }
    let template = st.engine.template;
    let format = CallFormat::for_template(template);
    let forced = forced && use_tools && st.cfg.constrain;
    let params = GenerateParams {
        max_tokens: body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(1024),
        sampler: sampling(body),
        stop_strings: stops(body),
        ..Default::default()
    };
    let id = st.next_id("chatcmpl-");
    let model = st.alias.clone();
    let created = now();

    if stream {
        http::start_sse(w)?;
        let first = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": Value::Null}]});
        http::sse(w, &first.to_string())?;
    }
    // One internal round for the built-in memory tool, which the server executes
    // itself; the client only sees the final answer or tool call.
    let mut round = 0;
    let (stats, content, call, filter, tokens, emitted) = loop {
        round += 1;
        let last_round = round >= 2;
        let (r_schemas, r_sigs): (Vec<Value>, Vec<ToolSig>) = if memory_tool && last_round {
            (
                schemas.iter().filter(|s| s.pointer("/function/name").and_then(Value::as_str) != Some(RECALL_TOOL)).cloned().collect(),
                sigs.iter().filter(|t| t.name != RECALL_TOOL).cloned().collect(),
            )
        } else {
            (schemas.clone(), sigs.clone())
        };
        let mut prompt = render(template, &msgs, if use_tools { &r_schemas } else { &[] }, true);
        if forced {
            prompt.push_str(format.call_prefix());
        }
        let tokens = st.engine.tokenize(&prompt, true, true);
        let mut filter = ToolCallFilter::with_limits(
            format,
            if use_tools { r_sigs } else { Vec::new() },
            forced,
            st.cfg.constrain && use_tools,
            st.cfg.max_string,
            st.cfg.max_call,
        );
        let mut json_filter = JsonModeFilter::new(json_mode && !use_tools, st.cfg.max_string, st.cfg.max_call);
        let mut text = String::new();
        let mut emitted = 0usize;
        let mut broken = false;
        let res = {
            let wref = &mut *w;
            let mut on_text = |piece: &str| -> bool {
                text.push_str(piece);
                if stream {
                    let vis = visible_prefix(&text, format, use_tools);
                    if vis.len() > emitted && vis.is_char_boundary(emitted) {
                        let delta = &vis[emitted..];
                        let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": Value::Null}]});
                        if http::sse(wref, &chunk.to_string()).is_err() {
                            broken = true;
                            return false;
                        }
                        emitted = vis.len();
                    }
                }
                true
            };
            if json_filter.enabled {
                st.engine.generate_hooked(&tokens, &params, &mut json_filter, &mut on_text)
            } else {
                st.engine.generate_hooked(&tokens, &params, &mut filter, &mut on_text)
            }
        };
        if let Ok(path) = std::env::var("RIE_SERVER_LOG") {
            // debugging aid: request summary + what was generated
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let req = json!({"tool_choice": body.get("tool_choice"), "n_tools": sigs.len(), "max_tokens": params.max_tokens, "stop": params.stop_strings,
                    "temperature": body.get("temperature"), "n_messages": msgs.len(), "forced": forced, "prompt_tokens": tokens.len(), "round": round});
                let out = match &res {
                    Ok(s) => {
                        json!({"stop": format!("{:?}", s.stop), "generated": s.generated_tokens, "text": String::from_utf8_lossy(&filter.raw).chars().take(300).collect::<String>()})
                    }
                    Err(e) => json!({"error": e.to_string()}),
                };
                let _ = writeln!(f, "{}", json!({"request": req, "result": out}));
            }
        }
        let stats = match res {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                let status = if msg.contains("context") { 400 } else { 500 };
                if stream {
                    let _ = http::sse(w, &err_json(&msg, "server_error"));
                    return Ok(false);
                }
                http::write_json(w, status, &err_json(&msg, if status == 400 { "invalid_request_error" } else { "server_error" }))?;
                return Ok(true);
            }
        };
        if broken {
            return Ok(false);
        }
        let (content, call) = if use_tools { extract(&filter) } else { (text.trim().to_string(), None) };
        if let Some(c) = call.as_ref().filter(|c| memory_tool && !last_round && c.name == RECALL_TOOL) {
            let q = c.arguments.get("query").and_then(Value::as_str).unwrap_or("").to_string();
            let found = st.memory.as_mut().map(|m| m.recall(&q, 5, None, 0.1)).unwrap_or_default();
            let result = if found.is_empty() {
                "No matching memories.".to_string()
            } else {
                found
                    .iter()
                    .map(|r| {
                        let ok = match r.obs.ok {
                            Some(true) => " ok",
                            Some(false) => " FAILED",
                            None => "",
                        };
                        format!("- [{}{ok}] {}", r.obs.kind.name(), clip(&r.obs.text.replace('\n', " "), 400))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let cid = st.next_id("mem_");
            msgs.push(ApiMessage {
                role: "assistant".into(),
                content: content.clone(),
                tool_calls: vec![tools::ApiToolCall { id: cid.clone(), name: RECALL_TOOL.into(), arguments: c.arguments.to_string() }],
                ..Default::default()
            });
            msgs.push(ApiMessage {
                role: "tool".into(),
                content: result,
                tool_call_id: Some(cid),
                name: Some(RECALL_TOOL.into()),
                ..Default::default()
            });
            continue;
        }
        break (stats, content, call, filter, tokens, emitted);
    };
    if let Some(mem) = &mut st.memory {
        match &call {
            Some(c) => mem.record(Kind::Action, &c.name, None, 0.2, &format!("called {}({})", c.name, clip(&c.arguments.to_string(), 300))),
            None => mem.record(Kind::Action, "", None, 0.2, &format!("answered: {}", clip(&content, 400))),
        };
    }
    let finish = match (&call, stats.stop) {
        (Some(_), _) => "tool_calls",
        (None, rie_core::StopReason::MaxTokens) | (None, rie_core::StopReason::ContextFull) => "length",
        _ => "stop",
    };
    let call_id = call.as_ref().map(|_| st.next_id("call_"));
    // remember the trajectory for feedback
    let mut steps = st.engine.take_trajectory();
    steps.truncate(MAX_RECORD_STEPS);
    st.remember(
        id.clone(),
        Record {
            prompt: tokens.clone(),
            steps,
            call_from: call.as_ref().map(|_| filter.call_first_token.unwrap_or(0)),
            tool_call_id: call_id.clone(),
            format,
            rewarded: false,
        },
    );
    let tool_calls: Vec<Value> = call
        .iter()
        .map(|c| json!({"index": 0, "id": call_id.clone().unwrap_or_default(), "type": "function", "function": {"name": c.name, "arguments": c.arguments.to_string()}}))
        .collect();
    let usage = json!({"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.generated_tokens, "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        "prompt_tokens_details": {"cached_tokens": stats.reused_tokens}});
    drop(st);
    if stream {
        // flush any held-back content (e.g. a partial trigger that never became a call)
        if call.is_none() && content.len() > emitted && content.is_char_boundary(emitted) {
            let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": &content[emitted.min(content.len())..]}, "finish_reason": Value::Null}]});
            http::sse(w, &chunk.to_string())?;
        }
        if !tool_calls.is_empty() {
            let chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": Value::Null}]});
            http::sse(w, &chunk.to_string())?;
        }
        let mut last = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "timings": timings(&stats)});
        if include_usage {
            last["usage"] = usage;
        }
        http::sse(w, &last.to_string())?;
        http::sse(w, "[DONE]")?;
        return Ok(false);
    }
    let mut message = json!({"role": "assistant", "content": if content.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(content) }});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(
            tool_calls
                .into_iter()
                .map(|mut c| {
                    c.as_object_mut().map(|o| o.remove("index"));
                    c
                })
                .collect(),
        );
    }
    let resp = json!({"id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage, "timings": timings(&stats)});
    http::write_json(w, 200, &resp.to_string())?;
    Ok(true)
}

fn completion(body: &Value, w: &mut TcpStream, state: &Arc<Mutex<State>>) -> std::io::Result<bool> {
    let prompt = body.get("prompt").map(tools::content_text).unwrap_or_default();
    let mut st = lock(state);
    let params = GenerateParams {
        max_tokens: body.get("max_tokens").or_else(|| body.get("n_predict")).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(256),
        sampler: sampling(body),
        stop_strings: stops(body),
        ..Default::default()
    };
    let mut text = String::new();
    let res = st.engine.generate(&prompt, &params, |p| {
        text.push_str(p);
        true
    });
    match res {
        Ok(s) => {
            let id = st.next_id("cmpl-");
            let steps = st.engine.take_trajectory();
            let tokens = st.engine.tokenize(&prompt, true, true);
            st.remember(
                id.clone(),
                Record { prompt: tokens, steps, call_from: None, tool_call_id: None, format: CallFormat::Hermes, rewarded: false },
            );
            let v = json!({"id": id, "object": "text_completion", "created": now(), "model": st.alias, "content": text,
                "choices": [{"index": 0, "text": text, "finish_reason": if s.stop == rie_core::StopReason::MaxTokens { "length" } else { "stop" }}],
                "usage": {"prompt_tokens": s.prompt_tokens, "completion_tokens": s.generated_tokens, "total_tokens": s.prompt_tokens + s.generated_tokens},
                "timings": timings(&s)});
            http::write_json(w, 200, &v.to_string())?
        }
        Err(e) => http::write_json(w, 400, &err_json(&e.to_string(), "invalid_request_error"))?,
    }
    Ok(true)
}

/// `POST /v1/corrections`: teach the right answer for a stored completion.
/// Body: `{"id": …, "content": "the right reply"}` or
/// `{"id": …, "tool_call": {"name": …, "arguments": {…}}}`; optional `"epochs"`.
fn correction(st: &mut State, body: &Value) -> Result<Value, String> {
    let id = body.get("id").and_then(Value::as_str).ok_or("missing id")?;
    if st.engine.adapter.is_none() {
        return Err("online learning is disabled (start with --rl or --adapter FILE)".into());
    }
    let epochs = body.get("epochs").and_then(Value::as_u64).unwrap_or(4).clamp(1, 32) as usize;
    let (prompt, format) = {
        let rec = st.records.get(id).ok_or_else(|| format!("unknown completion id {id}"))?;
        (rec.prompt.clone(), rec.format)
    };
    let target = if let Some(tc) = body.get("tool_call") {
        let name = tc.get("name").and_then(Value::as_str).ok_or("tool_call.name missing")?;
        let args = match tc.get("arguments") {
            Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| format!("tool_call.arguments: {e}"))?,
            Some(v) => v.clone(),
            None => json!({}),
        };
        format.render_call(name, &args)
    } else {
        body.get("content").and_then(Value::as_str).ok_or("give `content` or `tool_call`")?.to_string()
    };
    let mut cont = st.engine.tokenize(&target, false, true);
    if let Some(e) = st.engine.tokenizer.eot.or(st.engine.tokenizer.eos) {
        if body.get("tool_call").is_none() {
            cont.push(e); // also learn to stop after the corrected reply
        }
    }
    let good = st.engine.teacher_force(&prompt, &cont).map_err(|e| e.to_string())?;
    // First-error credit assignment: only the first token where the rejected reply left the
    // corrected one is pushed down (later tokens were conditioned on an already-wrong
    // context, and pushing them all down just spreads probability onto random tokens);
    // every token of the correction is pushed up.
    let (diverge, bad_n) = {
        let rec = st.records.get(id).unwrap();
        let d = rec.steps.iter().zip(&good).position(|(b, g)| b.token != g.token);
        (d, rec.steps.len())
    };
    {
        let rec = st.records.get_mut(id).unwrap();
        let adapter = st.engine.adapter.as_mut().unwrap();
        let bad: &[Step] = match diverge {
            Some(i) => &rec.steps[i..i + 1],
            None => &[],
        };
        for _ in 0..epochs {
            adapter.correct(bad, -1.0, 1);
            adapter.correct(&good, 1.0, 1);
        }
        rec.rewarded = true;
    }
    Ok(json!({"ok": true, "rejected_steps": bad_n, "diverged_at": diverge, "corrected_steps": good.len(), "epochs": epochs}))
}

/// Constrain the whole reply to a JSON object (`response_format: json_object`).
struct JsonModeFilter {
    enabled: bool,
    c: rie_core::constrain::JsonConstraint,
}

impl JsonModeFilter {
    fn new(enabled: bool, max_string: usize, max_call: usize) -> Self {
        JsonModeFilter {
            enabled,
            c: rie_core::constrain::JsonConstraint::any_object().with_max_string(max_string).with_max_total(max_call),
        }
    }
}

impl rie_core::engine::TokenFilter for JsonModeFilter {
    fn active(&self) -> bool {
        self.enabled && !self.c.is_complete()
    }
    fn allows(&self, _: u32, bytes: &[u8]) -> bool {
        self.c.accepts(bytes)
    }
    fn accept(&mut self, _: u32, bytes: &[u8]) -> bool {
        if self.enabled {
            self.c.feed(bytes);
            return self.c.is_complete();
        }
        false
    }
}

/// Parse a llama-server-compatible command line.
pub fn parse_args(args: &[String]) -> Result<ServerConfig, String> {
    let mut c = ServerConfig::default();
    let mut i = 0;
    let val = |i: &mut usize| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{} needs a value", args[*i - 1]))
    };
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-m" | "--model" => c.model = PathBuf::from(val(&mut i)?),
            "-c" | "--ctx-size" => c.n_ctx = val(&mut i)?.parse().map_err(|_| "bad -c")?,
            "-b" | "--batch-size" | "-ub" | "--ubatch-size" => c.n_batch = val(&mut i)?.parse().map_err(|_| "bad -b")?,
            "-t" | "--threads" => c.threads = Some(val(&mut i)?.parse().map_err(|_| "bad -t")?),
            "--host" => c.host = val(&mut i)?,
            "--port" => c.port = val(&mut i)?.parse().map_err(|_| "bad --port")?,
            "-a" | "--alias" => c.alias = Some(val(&mut i)?),
            "-ctk" | "--cache-type-k" | "-ctv" | "--cache-type-v" => {
                let v = val(&mut i)?;
                // llama.cpp's 4-bit KV types map to TurboQuant-4, 5/8-bit ones to Q8_0
                if v.starts_with("q4") || v.starts_with("turbo") || v.starts_with("tq") || v.starts_with("iq4") {
                    c.kv = KvType::Turbo4;
                } else if v.starts_with("q8") || v.starts_with("q5") {
                    c.kv = KvType::Q8_0;
                }
            }
            "--kv" => {
                let v = val(&mut i)?;
                c.kv = KvType::parse(&v).ok_or_else(|| format!("unknown --kv {v} (f16, q8_0, turbo4)"))?;
            }
            "--stream" => c.stream = true,
            "--memory" => c.memory = Some(PathBuf::from(val(&mut i)?)),
            "--memory-size" => c.memory_size = val(&mut i)?.parse().map_err(|_| "bad --memory-size")?,
            "--recall" => c.recall = val(&mut i)?.parse().map_err(|_| "bad --recall")?,
            "--no-memory-tool" => c.memory_tool = false,
            "--tool-recall" => c.tool_recall = val(&mut i)?.parse().map_err(|_| "bad --tool-recall")?,
            "--mlock" => c.mlock = true,
            "--gpu" => c.gpu = true,
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                let v = val(&mut i)?;
                if std::env::var("RIE_SERVER_HONOR_NGL").is_ok() && v != "0" {
                    c.gpu = true;
                }
            }
            "--rl" => c.rl = true,
            "--adapter" => c.adapter = Some(PathBuf::from(val(&mut i)?)),
            "--auto-reward" => {
                c.auto_reward = true;
                c.rl = true;
            }
            "--rl-lr" => c.rl_cfg.lr = val(&mut i)?.parse().map_err(|_| "bad --rl-lr")?,
            "--experts" => c.rl_cfg.n_experts = val(&mut i)?.parse().map_err(|_| "bad --experts")?,
            "--rank" => c.rl_cfg.rank = val(&mut i)?.parse().map_err(|_| "bad --rank")?,
            "--no-constrain" => c.constrain = false,
            "--max-call" => c.max_call = val(&mut i)?.parse().map_err(|_| "bad --max-call")?,
            "--tool-choice" => c.force_tools = val(&mut i)? == "required",
            "--max-string" => c.max_string = val(&mut i)?.parse().map_err(|_| "bad --max-string")?,
            // llama.cpp flags that do not apply here (accepted so harnesses can launch us unchanged)
            "--fit" | "-fa" | "--flash-attn" => {
                if args.get(i + 1).is_some_and(|v| matches!(v.as_str(), "on" | "off" | "auto")) {
                    i += 1;
                }
            }
            "--fit-target"
            | "-np"
            | "--parallel"
            | "--metrics-port"
            | "--chat-template"
            | "--chat-template-file"
            | "-n"
            | "--n-predict"
            | "--keep"
            | "-s"
            | "--seed"
            | "--temp"
            | "--top-k"
            | "--top-p"
            | "--min-p"
            | "--rope-freq-base"
            | "--reasoning-format"
            | "--reasoning-budget"
            | "-dev"
            | "--device"
            | "-sm"
            | "--split-mode"
            | "-mg"
            | "--main-gpu"
            | "--slot-save-path"
            | "--api-key" => {
                val(&mut i)?;
            }
            "--jinja" | "--metrics" | "--no-mmap" | "--cont-batching" | "-cb" | "--no-webui" | "--verbose" | "-v" | "--log-disable"
            | "--embedding" | "--slots" | "--no-warmup" => {}
            "-h" | "--help" => return Err(USAGE.into()),
            other if other.starts_with('-') => eprintln!("rie-server: ignoring unknown flag {other}"),
            other if c.model.as_os_str().is_empty() => c.model = PathBuf::from(other),
            other => eprintln!("rie-server: ignoring argument {other}"),
        }
        i += 1;
    }
    if c.model.as_os_str().is_empty() {
        return Err(format!("no model given\n\n{USAGE}"));
    }
    Ok(c)
}

pub const USAGE: &str = "\
rie-server — OpenAI-compatible server (drop-in for llama-server) with constrained tool calling and online RL

usage: rie-server -m MODEL.gguf [options]

  -m, --model FILE        GGUF model
  -c, --ctx-size N        context window (default 4096)
  -t, --threads N         CPU threads (default: performance cores)
  -b, --batch-size N      prompt batch (default 64)
      --host H            bind address (default 127.0.0.1)
      --port P            port (default 8080)
  -a, --alias NAME        model name reported to clients
  -ctk/-ctv q8_0          8-bit KV cache (q4_0 / turbo4: TurboQuant 4-bit KV, 3.9x smaller than f16)
      --kv f16|q8_0|turbo4  KV cache format
      --stream            stream layers from flash (models larger than RAM)
      --gpu               Vulkan/Metal backend (build with --features gpu; disables RL)
      --mlock             lock weights in RAM
      --no-constrain      do not grammar-constrain tool calls
      --max-string N      max bytes per string value in a constrained call (default 4096)
      --max-call N        soft byte budget per tool call, then wound down to a valid end (default 16384)
      --tool-choice required  force a tool call whenever tools are offered with tool_choice auto

memory and tool knowledge (on the device):
      --memory DIR        record every request, action and tool result in DIR (compact journal
                          + TurboQuant 4-bit index), recall relevant ones into the context,
                          and keep learned tool cards in DIR/tools/*.md
      --memory-size N     observations kept before the weakest are forgotten (default 20000)
      --recall N          observations recalled into each request (default 3, 0 = tool only)
      --no-memory-tool    do not offer the built-in recall_memory tool
      --tool-recall K     show the model only the K tools most relevant to the request

online learning (reinforcement learning from feedback):
      --rl                enable correction experts (in memory)
      --adapter FILE      enable and persist them in FILE
      --auto-reward       learn from tool results sent back by the client
      --experts K         number of correction experts (default 4)
      --rank R            rank per expert (default 8)
      --rl-lr X           learning rate (default 0.5)

llama.cpp flags such as -ngl, -fa, --fit, --jinja, -np, --metrics are accepted and ignored.

endpoints: /v1/chat/completions (tools, tool_choice, stream, response_format), /v1/completions,
           /v1/models, /health, /props, POST /v1/feedback, POST /v1/corrections, GET /v1/rl, POST /v1/rl/reset,
           GET /v1/memory[?q=..&k=..] (Markdown journal or search), POST /v1/memory (add a note),
           GET /v1/tools, GET /v1/tools/NAME (learned tool card, Markdown)";
