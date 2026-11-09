//! Scenario tests for "ask the phone": `/v1/ask` picks the relevant headless tools, the
//! model fills their arguments, the tools run in parallel and the result is rendered as a
//! categorized page in one fixed style.
//!
//! Uses the deterministic demo tools in `examples/phone-tools` (needs `python3`) plus
//! HTTP tools served by the test itself. The model has random weights: arguments are
//! schema-valid because of the grammar constraint, not because the model understands.

use rie_core::quant::GgmlType;
use rie_core::testing::SyntheticSpec;
use rie_server::{build_engine, serve_on, ServerConfig, State};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn demo_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/phone-tools/demo.json")
}

fn have_python() -> bool {
    std::process::Command::new("python3").arg("-c").arg("pass").status().map(|s| s.success()).unwrap_or(false)
}

fn start(headless: Option<PathBuf>, memory: Option<PathBuf>) -> u16 {
    let mut spec = SyntheticSpec::tiny("qwen2", GgmlType::Q8_0);
    spec.ctx = 4096;
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("rie-scenario-{}-{n}.gguf", std::process::id()));
    std::fs::write(&path, spec.build()).unwrap();
    let cfg = ServerConfig { model: path, n_ctx: 4096, threads: Some(2), headless, memory, max_string: 24, ..Default::default() };
    let engine = build_engine(&cfg).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(Mutex::new(State::new(engine, cfg)));
    std::thread::spawn(move || serve_on(listener, state));
    port
}

fn memory_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("rie-scenario-mem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// One shared server with the demo tools and an on-device memory.
fn demo() -> u16 {
    static P: OnceLock<u16> = OnceLock::new();
    *P.get_or_init(|| start(Some(demo_manifest()), Some(memory_dir())))
}

fn req(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
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
    (status, out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default())
}

fn ask(port: u16, body: Value) -> Value {
    let (st, b) = req(port, "POST", "/v1/ask", Some(&body));
    assert_eq!(st, 200, "{b}");
    serde_json::from_str(&b).unwrap()
}

fn cards(page: &Value) -> Vec<&Value> {
    page["ui"]["sections"].as_array().unwrap().iter().flat_map(|s| s["cards"].as_array().unwrap()).collect()
}

fn card<'a>(page: &'a Value, tool: &str) -> &'a Value {
    cards(page).into_iter().find(|c| c["tool"] == tool).unwrap_or_else(|| panic!("no card for {tool}: {page}"))
}

fn categories(page: &Value) -> Vec<String> {
    page["ui"]["sections"].as_array().unwrap().iter().map(|s| s["category"].as_str().unwrap().to_string()).collect()
}

macro_rules! need_python {
    () => {
        if !have_python() {
            eprintln!("skipped: python3 is needed for the demo tools");
            return;
        }
    };
}

#[test]
fn s01_single_category_question_calls_the_matching_tool() {
    need_python!();
    let p = ask(demo(), json!({"question": "What's the weather in Paris today?"}));
    assert_eq!(categories(&p)[0], "weather", "{p}");
    let c = card(&p, "weather.current");
    assert_eq!(c["status"], "ok");
    let fields = &c["items"][0];
    assert_eq!(fields["type"], "fields");
    assert!(fields["rows"].to_string().contains("Temperature c"), "{fields}");
    assert!(c["items"].to_string().contains("\"type\":\"table\""), "forecast table: {c}");
    // required argument filled with a string (schema-valid even with random weights)
    assert!(c["args"]["city"].is_string(), "{c}");
}

#[test]
fn s02_broad_question_fans_out_to_every_relevant_category() {
    need_python!();
    let q = "Morning briefing: weather, calendar events, tasks due, bank balance, news headlines, battery and steps";
    let p = ask(demo(), json!({"question": q, "max_tools": 10}));
    let cats = categories(&p);
    for want in ["weather", "calendar", "finance", "web", "device", "health"] {
        assert!(cats.iter().any(|c| c == want), "missing {want} in {cats:?}");
    }
    // sections always follow the fixed category order
    let order: Vec<usize> = cats.iter().map(|c| rie_server::ui::CATEGORIES.iter().position(|k| k.key == c).unwrap()).collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{cats:?}");
    // and cards are sorted by tool name inside a section
    for s in p["ui"]["sections"].as_array().unwrap() {
        let names: Vec<&str> = s["cards"].as_array().unwrap().iter().map(|c| c["tool"].as_str().unwrap()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }
    assert_eq!(p["ui"]["summary"]["failed"], 0, "{p}");
}

#[test]
fn s03_failures_timeouts_and_odd_outputs_are_shown_not_fatal() {
    need_python!();
    let t0 = std::time::Instant::now();
    let tools = ["transit.status", "traffic.live", "notes.raw", "web.preview", "logs.dump", "bank.balance"];
    let p = ask(demo(), json!({"question": "status check", "tools": tools}));
    // the slow tool is cut off at its 800 ms timeout; tools run in parallel
    assert!(t0.elapsed().as_secs_f32() < 30.0, "{:?}", t0.elapsed());
    let fail = card(&p, "transit.status");
    assert_eq!(fail["status"], "error");
    assert_eq!(fail["items"][0]["type"], "error");
    assert!(fail["items"][0]["text"].as_str().unwrap().contains("service unavailable"));
    let slow = card(&p, "traffic.live");
    assert_eq!(slow["status"], "timeout");
    assert!(slow["ms"].as_u64().unwrap() < 5000);
    let text = card(&p, "notes.raw");
    assert_eq!(text["items"][0]["type"], "text");
    let big = card(&p, "logs.dump");
    assert_eq!(big["truncated"], true);
    assert_eq!(card(&p, "bank.balance")["status"], "ok");
    assert_eq!(p["ui"]["summary"]["failed"], 2);
    // hostile tool output is escaped in the HTML page
    let (st, html) = req(demo(), "GET", p["ui_url"].as_str().unwrap(), None);
    assert_eq!(st, 200);
    assert!(!html.contains("<script>alert") && html.contains("&lt;script&gt;"), "unescaped tool output");
    assert!(html.contains("badge b-timeout") && html.contains("badge b-error") && html.contains("Output truncated."));
    assert!(html.contains("card st-error") && html.contains("<summary>Show all</summary>"));
}

#[test]
fn s04_unrelated_question_gives_an_empty_state_not_random_tools() {
    need_python!();
    let p = ask(demo(), json!({"question": "zebra quantum origami"}));
    assert_eq!(p["ui"]["summary"]["tools"], 0, "{p}");
    let (_, html) = req(demo(), "GET", p["ui_url"].as_str().unwrap(), None);
    assert!(html.contains("No relevant tools were found"));
    assert!(p["markdown"].as_str().unwrap().contains("No relevant tools were found"));
}

#[test]
fn s05_model_filled_arguments_always_match_the_schema() {
    need_python!();
    for _ in 0..4 {
        let p = ask(demo(), json!({"question": "echo test", "tools": ["debug.echo"]}));
        let got = &card(&p, "debug.echo")["items"].to_string();
        let args = &card(&p, "debug.echo")["args"];
        assert!(args["unit"] == "c" || args["unit"] == "f", "enum respected: {args}");
        assert!(args["count"].is_i64(), "integer: {args}");
        assert!(got.contains("Unit") || got.contains("unit"), "tool received the args: {got}");
    }
}

/// Remove the parts that legitimately change between runs (ids, times, durations).
fn stable(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("ui_") {
        out.push_str(&rest[..i]);
        let end = rest[i..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map(|e| i + e).unwrap_or(rest.len());
        out.push_str("ID");
        rest = &rest[end..];
    }
    out.push_str(rest);
    let re_ms = |t: &str| {
        let mut o = String::new();
        let b: Vec<char> = t.chars().collect();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_digit() {
                let j = (i..b.len()).find(|&j| !b[j].is_ascii_digit()).unwrap_or(b.len());
                o.push('N');
                i = j;
            } else {
                o.push(b[i]);
                i += 1;
            }
        }
        o
    };
    re_ms(&out)
}

#[test]
fn s06_same_question_same_layout_every_time() {
    need_python!();
    let q = json!({"question": "battery, steps and bank balance", "tools": ["device.battery", "health.steps", "bank.balance", "calendar.today"]});
    let a = ask(demo(), q.clone());
    let b = ask(demo(), q);
    assert_ne!(a["id"], b["id"]);
    let (_, ha) = req(demo(), "GET", a["ui_url"].as_str().unwrap(), None);
    let (_, hb) = req(demo(), "GET", b["ui_url"].as_str().unwrap(), None);
    assert_eq!(stable(&ha), stable(&hb), "HTML differs between identical requests");
    assert_eq!(stable(a["markdown"].as_str().unwrap()), stable(b["markdown"].as_str().unwrap()));
    assert_eq!(a["ui"]["layout"], 1);
    // metric formatting: steps is a bare number → metric with thousands separator
    assert_eq!(card(&a, "health.steps")["items"][0], json!({"type": "metric", "label": "Steps", "value": "8,432"}));
}

#[test]
fn s07_page_formats_and_routes() {
    need_python!();
    let p = ask(demo(), json!({"question": "my calendar", "tools": ["calendar.today"]}));
    let url = p["ui_url"].as_str().unwrap();
    let (s1, html) = req(demo(), "GET", url, None);
    let (s2, js) = req(demo(), "GET", &format!("{url}.json"), None);
    let (s3, md) = req(demo(), "GET", &format!("{url}.md"), None);
    assert_eq!((s1, s2, s3), (200, 200, 200));
    assert!(html.starts_with("<!doctype html>") && html.contains("prefers-color-scheme:dark") && html.contains("width=device-width"));
    assert_eq!(serde_json::from_str::<Value>(&js).unwrap()["id"], p["id"]);
    assert!(md.contains("### ▦ Calendar & Tasks") && md.contains("| Time | Title | Place |"), "{md}");
    let (s4, _) = req(demo(), "GET", "/ui/ui_doesnotexist", None);
    assert_eq!(s4, 404);
    let (_, index) = req(demo(), "GET", "/ui", None);
    assert!(index.contains(url));
}

#[test]
fn s08_outcomes_are_learned_and_remembered() {
    need_python!();
    ask(demo(), json!({"question": "transit", "tools": ["transit.status"]}));
    let (st, card_md) = req(demo(), "GET", "/v1/tools/transit.status", None);
    assert_eq!(st, 200);
    assert!(card_md.contains("service unavailable"), "limitation learned: {card_md}");
    let (_, found) = req(demo(), "GET", "/v1/memory?q=transit%20service%20unavailable&k=3", None);
    assert!(found.contains("transit.status"), "{found}");
    let (_, j) = req(demo(), "GET", "/v1/memory?q=answered%20page&k=5", None);
    assert!(j.contains("answered"), "answer recorded: {j}");
}

#[test]
fn s09_concurrent_questions() {
    need_python!();
    let port = demo();
    let hs: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                ask(port, json!({"question": format!("battery and weather {i}"), "tools": ["device.battery", "weather.current"]}))
            })
        })
        .collect();
    let ids: std::collections::HashSet<String> = hs.into_iter().map(|h| h.join().unwrap()["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids.len(), 4);
}

#[test]
fn s10_invalid_requests_and_unconfigured_server() {
    need_python!();
    assert_eq!(req(demo(), "POST", "/v1/ask", Some(&json!({"question": "  "}))).0, 400);
    let (st, b) = req(demo(), "POST", "/v1/ask", Some(&json!({"question": "x", "tools": ["nope.tool"]})));
    assert!(st == 400 && b.contains("unknown tool nope.tool"));
    let bare = start(None, None);
    let (st, b) = req(bare, "POST", "/v1/ask", Some(&json!({"question": "weather"})));
    assert!(st == 400 && b.contains("--headless-tools"), "{b}");
}

#[test]
fn s11_everything_mode_runs_all_tools() {
    need_python!();
    let p = ask(demo(), json!({"question": "everything", "all": true}));
    assert_eq!(p["ui"]["summary"]["tools"], 17, "{}", p["ui"]["summary"]);
    assert!(categories(&p).len() >= 8);
}

/// Minimal HTTP server for HTTP-backed headless tools.
fn http_tools_server() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let mut s = s;
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
            let resp = match path.as_str() {
                "/ok" => {
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 27\r\n\r\n{\"aqi\": 42, \"level\": \"good\"}"
                        .to_string()
                }
                "/chunked" => "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n[1, 2, \r\n2\r\n3]\r\n0\r\n\r\n".to_string(),
                _ => "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\n\r\noops".to_string(),
            };
            let _ = s.write_all(resp.as_bytes());
        }
    });
    port
}

#[test]
fn s12_http_tools_success_chunked_errors_and_unreachable() {
    let hp = http_tools_server();
    let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port(); // closed right away
    let dir = std::env::temp_dir().join(format!("rie-scenario-http-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = dir.join("tools.json");
    let t =
        |name: &str, url: String, cat: &str| json!({"name": name, "description": name, "category": cat, "http": url, "timeout_ms": 2000});
    std::fs::write(
        &manifest,
        json!({"tools": [
            t("air.quality", format!("http://127.0.0.1:{hp}/ok"), "health"),
            t("air.series", format!("http://127.0.0.1:{hp}/chunked"), "health"),
            t("air.broken", format!("http://127.0.0.1:{hp}/fail"), "health"),
            t("air.secure", "https://example.com/x".into(), "health"),
            t("air.down", format!("http://127.0.0.1:{dead}/"), "health"),
            {"name": "missing.binary", "description": "x", "command": ["/nonexistent/tool-binary"]},
        ]})
        .to_string(),
    )
    .unwrap();
    let port = start(Some(manifest), None);
    let p = ask(port, json!({"question": "air", "all": true}));
    assert_eq!(card(&p, "air.quality")["items"][0]["rows"], json!([["Aqi", "42"], ["Level", "good"]]));
    assert_eq!(card(&p, "air.series")["items"][0], json!({"type": "list", "title": null, "items": ["1", "2", "3"], "more": 0}));
    assert!(card(&p, "air.broken")["items"][0]["text"].as_str().unwrap().contains("HTTP 500"));
    assert!(card(&p, "air.secure")["items"][0]["text"].as_str().unwrap().contains("only http://"));
    assert_eq!(card(&p, "air.down")["status"], "error");
    let mb = card(&p, "missing.binary");
    assert!(mb["status"] == "error", "{mb}");
    assert_eq!(p["ui"]["summary"]["ok"], 2);
}

#[test]
fn s13_bad_manifests_are_rejected_with_a_reason() {
    use rie_server::headless::load_manifest;
    let dir = std::env::temp_dir().join(format!("rie-scenario-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (body, why) in [
        ("not json", "expected"),
        ("{}", "\"tools\" array"),
        (r#"{"tools":[{"description":"x","command":"a"}]}"#, "without a name"),
        (r#"{"tools":[{"name":"a"}]}"#, "needs \"command\" or \"http\""),
        (r#"{"tools":[{"name":"a","command":"x"},{"name":"a","command":"y"}]}"#, "duplicate"),
    ] {
        let f = dir.join("m.json");
        std::fs::write(&f, body).unwrap();
        let e = load_manifest(&f).unwrap_err();
        assert!(e.contains(why), "{body}: {e}");
    }
    assert!(load_manifest(&demo_manifest()).unwrap().len() == 17);
    assert!(load_manifest(&demo_manifest().with_file_name("termux.json")).unwrap().len() == 10);
}

#[test]
fn s14_tool_is_not_run_when_its_arguments_cannot_be_prepared() {
    need_python!();
    // a context window too small for the tool prompt
    let spec = SyntheticSpec::tiny("qwen2", GgmlType::Q8_0);
    let path = std::env::temp_dir().join(format!("rie-scenario-small-{}.gguf", std::process::id()));
    std::fs::write(&path, spec.build()).unwrap();
    let cfg = ServerConfig { model: path, n_ctx: 256, threads: Some(2), headless: Some(demo_manifest()), ..Default::default() };
    let engine = build_engine(&cfg).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(Mutex::new(State::new(engine, cfg)));
    std::thread::spawn(move || serve_on(listener, state));
    let p = ask(port, json!({"question": "x", "tools": ["debug.echo", "device.battery"]}));
    let echo = card(&p, "debug.echo");
    assert_eq!(echo["status"], "error");
    assert!(echo["items"][0]["text"].as_str().unwrap().contains("could not prepare arguments"), "{echo}");
    assert_eq!(echo["items"].as_array().unwrap().len(), 1, "the tool itself did not run");
    // a tool without parameters needs no model call and still works
    assert_eq!(card(&p, "device.battery")["status"], "ok");
}
