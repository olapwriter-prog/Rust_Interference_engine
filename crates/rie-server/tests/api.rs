//! End-to-end HTTP tests with a synthetic model (random weights). Because the weights are
//! random, any schema-valid tool call is produced by the grammar constraint, not by luck.
//! (A random model rarely samples the closing quote of a free-text string, hence the large
//! `max_tokens` budgets; real models close strings in a few tokens.)

use rie_core::quant::GgmlType;
use rie_core::testing::SyntheticSpec;
use rie_server::{build_engine, serve_on, ServerConfig, State};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, OnceLock};

fn server() -> u16 {
    static PORT: OnceLock<u16> = OnceLock::new();
    *PORT.get_or_init(|| {
        let mut spec = SyntheticSpec::tiny("qwen2", GgmlType::Q8_0);
        spec.ctx = 4096;
        let path = std::env::temp_dir().join(format!("rie-server-test-{}.gguf", std::process::id()));
        std::fs::write(&path, spec.build()).unwrap();
        let cfg = ServerConfig {
            model: path,
            n_ctx: 4096,
            threads: Some(2),
            rl: true,
            auto_reward: true,
            alias: Some("synthetic".into()),
            max_string: 48,
            ..Default::default()
        };
        let engine = build_engine(&cfg).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(Mutex::new(State::new(engine, cfg)));
        std::thread::spawn(move || serve_on(listener, state));
        port
    })
}

fn request(method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", server())).unwrap();
    let b = body.map(|v| v.to_string()).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let status = out[9..12].parse().unwrap();
    let body = out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, body)
}

fn post(path: &str, body: Value) -> Value {
    let (st, b) = request("POST", path, Some(&body));
    assert_eq!(st, 200, "{path}: {b}");
    serde_json::from_str(&b).unwrap()
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "fin.loan_emi", "description": "Monthly EMI for a loan",
            "parameters": {"type": "object", "properties": {"principal": {"type": "number"}, "annualRate": {"type": "number"}, "months": {"type": "integer"}},
                "required": ["principal", "annualRate", "months"]}}},
        {"type": "function", "function": {"name": "weather.get", "description": "Current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
    ])
}

#[test]
fn health_props_models() {
    let (s, b) = request("GET", "/health", None);
    assert_eq!((s, b.as_str()), (200, r#"{"status":"ok"}"#));
    let (_, b) = request("GET", "/props", None);
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["default_generation_settings"]["n_ctx"], 4096);
    let (_, b) = request("GET", "/v1/models", None);
    assert!(b.contains("\"synthetic\""));
}

#[test]
fn forced_tool_calls_are_always_schema_valid() {
    for seed in 0..6u64 {
        let r = post(
            "/v1/chat/completions",
            json!({
                "messages": [{"role": "user", "content": "What is the EMI on 10 lakh at 8.5% for 20 years?"}],
                "tools": tools(), "tool_choice": "required", "temperature": 1.0, "seed": seed, "max_tokens": 1500
            }),
        );
        let ch = &r["choices"][0];
        assert_eq!(ch["finish_reason"], "tool_calls", "seed {seed}: {r}");
        let call = &ch["message"]["tool_calls"][0];
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        let name = call["function"]["name"].as_str().unwrap();
        let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        match name {
            "fin.loan_emi" => {
                assert!(args["principal"].is_number() && args["annualRate"].is_number(), "{args}");
                assert!(args["months"].is_i64() || args["months"].is_u64(), "{args}");
            }
            "weather.get" => {
                assert!(args["city"].is_string(), "{args}");
                if let Some(u) = args.get("unit") {
                    assert!(u == "c" || u == "f");
                }
            }
            other => panic!("unknown tool {other}"),
        }
    }
    // a named function forces that tool
    let r = post(
        "/v1/chat/completions",
        json!({
            "messages": [{"role": "user", "content": "weather?"}], "tools": tools(),
            "tool_choice": {"type": "function", "function": {"name": "weather.get"}}, "seed": 3, "max_tokens": 1500
        }),
    );
    assert_eq!(r["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "weather.get");
}

#[test]
fn streaming_and_json_mode() {
    let (s, b) = request(
        "POST",
        "/v1/chat/completions",
        Some(&json!({
            "messages": [{"role": "user", "content": "hi"}], "stream": true, "max_tokens": 8, "stream_options": {"include_usage": true}
        })),
    );
    assert_eq!(s, 200);
    let events: Vec<&str> = b.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(*events.last().unwrap(), "[DONE]");
    let last: Value = serde_json::from_str(events[events.len() - 2]).unwrap();
    assert!(last["choices"][0]["finish_reason"].is_string());
    assert_eq!(last["usage"]["completion_tokens"], 8);
    assert!(last["timings"]["predicted_per_second"].as_f64().unwrap() > 0.0);
    // streamed tool call arrives as a tool_calls delta
    let (_, b) = request(
        "POST",
        "/v1/chat/completions",
        Some(&json!({
            "messages": [{"role": "user", "content": "weather in Pune"}], "tools": tools(), "tool_choice": "required", "stream": true, "max_tokens": 1500, "seed": 1
        })),
    );
    assert!(b.contains("\"tool_calls\"") && b.contains("\"finish_reason\":\"tool_calls\""), "{b}");
    // response_format json_object
    let r = post(
        "/v1/chat/completions",
        json!({
            "messages": [{"role": "user", "content": "give json"}], "response_format": {"type": "json_object"}, "max_tokens": 1500, "seed": 5
        }),
    );
    let content = r["choices"][0]["message"]["content"].as_str().unwrap();
    let v: Value = serde_json::from_str(content).unwrap_or_else(|e| panic!("not JSON ({e}): {content}"));
    assert!(v.is_object());
}

#[test]
fn corrections_change_the_next_answer() {
    let ask = || {
        post(
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "Name a device."}], "temperature": 0, "max_tokens": 12}),
        )
    };
    let r1 = ask();
    let id = r1["id"].as_str().unwrap().to_string();
    let before = r1["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    let c = post("/v1/corrections", json!({"id": id, "content": "phone"}));
    assert!(c["corrected_steps"].as_u64().unwrap() >= 1, "{c}");
    let after = ask()["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert!(after.starts_with("phone"), "before {before:?}, after correction {after:?}");
    let (_, b) = request("GET", "/v1/rl", None);
    let v: Value = serde_json::from_str(&b).unwrap();
    assert!(v["updates"].as_u64().unwrap() > 0);
}

#[test]
fn feedback_and_auto_reward_from_tool_results() {
    let r = post(
        "/v1/chat/completions",
        json!({
            "messages": [{"role": "user", "content": "weather in Pune"}], "tools": tools(), "tool_choice": "required", "seed": 9, "max_tokens": 1500
        }),
    );
    let id = r["id"].as_str().unwrap();
    let call = r["choices"][0]["message"]["tool_calls"][0].clone();
    let fb = post("/v1/feedback", json!({"id": id, "reward": -1.0, "scope": "tool_call"}));
    assert!(fb["steps"].as_u64().unwrap() > 0);
    let (_, b) = request("GET", "/v1/rl", None);
    let before: Value = serde_json::from_str(&b).unwrap();
    // the harness sends the tool result back in the next turn: a failure triggers a negative reward
    post(
        "/v1/chat/completions",
        json!({
            "messages": [
                {"role": "user", "content": "weather in Pune"},
                {"role": "assistant", "content": null, "tool_calls": [call.clone()]},
                {"role": "tool", "tool_call_id": call["id"], "content": "Error: unknown city"}
            ],
            "tools": tools(), "max_tokens": 4
        }),
    );
    let (_, b) = request("GET", "/v1/rl", None);
    let after: Value = serde_json::from_str(&b).unwrap();
    assert!(after["feedback_events"].as_u64() >= before["feedback_events"].as_u64(), "{after}");
    let (s, _) = request("POST", "/v1/feedback", Some(&json!({"id": "nope", "reward": 1})));
    assert_eq!(s, 400);
}

#[test]
fn llama_server_compatible_flags() {
    let args: Vec<String> = "-m /x.gguf -c 8192 -ngl 99 --fit on --fit-target 1024 -np 1 -fa on -ctk q8_0 -ctv q8_0 -t 4 --mlock --jinja --metrics --host 127.0.0.1 --port 9123 --alias qwen"
        .split_whitespace()
        .map(String::from)
        .collect();
    let c = rie_server::parse_args(&args).unwrap();
    assert_eq!((c.n_ctx, c.port, c.threads, c.alias.as_deref()), (8192, 9123, Some(4), Some("qwen")));
    assert_eq!(c.kv, rie_core::KvType::Q8_0);
    assert!(c.mlock && !c.gpu);
}
