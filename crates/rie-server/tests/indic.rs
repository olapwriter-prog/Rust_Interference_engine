//! Indian-language scenarios (Telugu first, Hindi too): tool selection from Telugu/Hindi
//! questions, localized answer pages, speech text, the TTS endpoints (with a fake engine
//! that records what it was asked to say), Telugu inside constrained tool calls and
//! Telugu memory recall. The model has random weights: these check the plumbing, not
//! the model's Telugu.

use rie_core::quant::GgmlType;
use rie_core::testing::SyntheticSpec;
use rie_server::tts::Backend;
use rie_server::{build_engine, serve_on, ServerConfig, State};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn have_python() -> bool {
    std::process::Command::new("python3").arg("-c").arg("pass").status().map(|s| s.success()).unwrap_or(false)
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rie-indic-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn start(tweak: impl FnOnce(&mut ServerConfig)) -> u16 {
    let mut spec = SyntheticSpec::tiny("qwen2", GgmlType::Q8_0);
    spec.ctx = 4096;
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("rie-indic-{}-{n}.gguf", std::process::id()));
    std::fs::write(&path, spec.build()).unwrap();
    let mut cfg = ServerConfig {
        model: path,
        n_ctx: 4096,
        threads: Some(2),
        headless: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/phone-tools/demo.json")),
        max_string: 24,
        ..Default::default()
    };
    tweak(&mut cfg);
    let engine = build_engine(&cfg).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(Mutex::new(State::new(engine, cfg)));
    std::thread::spawn(move || serve_on(listener, state));
    port
}

fn req(port: u16, method: &str, path: &str, body: Option<&Value>) -> (u16, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let b = body.map(|v| v.to_string()).unwrap_or_default();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )
    .unwrap();
    let mut out = Vec::new();
    s.read_to_end(&mut out).unwrap();
    let status = std::str::from_utf8(&out[9..12]).unwrap().parse().unwrap();
    let i = out.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4).unwrap_or(out.len());
    (status, out[i..].to_vec())
}

fn post(port: u16, path: &str, body: Value) -> Value {
    let (st, b) = req(port, "POST", path, Some(&body));
    let text = String::from_utf8(b).unwrap();
    assert_eq!(st, 200, "{path}: {text}");
    serde_json::from_str(&text).unwrap()
}

fn tools_of(page: &Value) -> Vec<String> {
    page["ui"]["sections"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s["cards"].as_array().unwrap())
        .map(|c| c["tool"].as_str().unwrap().to_string())
        .collect()
}

/// A fake TTS engine: records "lang|text" per call and, given an output path, writes a
/// small WAV whose length follows the text.
fn fake_engine(dir: &std::path::Path) -> (PathBuf, PathBuf) {
    let log = dir.join("spoken.log");
    let script = dir.join("fake_tts.py");
    std::fs::write(
        &script,
        format!(
            "import sys, wave\nout, lang, text = sys.argv[1], sys.argv[2], sys.argv[3]\nopen({:?}, 'a', encoding='utf-8').write(lang + '|' + text + '\\n')\nif out != '-':\n    w = wave.open(out, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)\n    w.writeframes(b'\\x01\\x00' * (10 * len(text))); w.close()\n",
            log.display().to_string()
        ),
    )
    .unwrap();
    (script, log)
}

#[test]
fn telugu_question_selects_tools_and_gets_a_telugu_page() {
    if !have_python() {
        return eprintln!("skipped: python3 needed");
    }
    let port = start(|_| {});
    let p = post(port, "/v1/ask", json!({"question": "ఈరోజు వాతావరణం ఎలా ఉంది?"}));
    assert_eq!(tools_of(&p), vec!["weather.current"], "{}", p["ui"]);
    assert_eq!(p["lang"], "te");
    assert_eq!(p["ui"]["lang"], "te");
    assert_eq!(p["ui"]["sections"][0]["title"], "వాతావరణం");
    assert_eq!(p["ui"]["sections"][0]["cards"][0]["status_label"], "సరే");
    let (_, html) = req(port, "GET", p["ui_url"].as_str().unwrap(), None);
    let html = String::from_utf8(html).unwrap();
    assert!(html.contains("<html lang=\"te\">") && html.contains("సమాధానం") && html.contains("సాధనాలు") && html.contains("Noto Sans Telugu"));
    let md = p["markdown"].as_str().unwrap();
    assert!(md.contains("### ☀ వాతావరణం"), "{md}");
    // the spoken summary reads numbers as Telugu words (18.4 → పద్దెనిమిది పాయింట్ నాలుగు)
    let speech = p["speech"].as_str().unwrap();
    assert!(speech.starts_with("వాతావరణం.") && speech.contains("పద్దెనిమిది పాయింట్ నాలుగు"), "{speech}");
    assert!(!speech.contains("18.4"), "{speech}");
}

#[test]
fn inflected_and_code_mixed_telugu_fans_out() {
    if !have_python() {
        return eprintln!("skipped: python3 needed");
    }
    let port = start(|_| {});
    // agglutinated forms (క్యాలెండర్‌లో, అడుగులు, ఖర్చులు) and English words mixed in
    let q = "నా క్యాలెండర్‌లో సమావేశాలు, bank బ్యాలెన్స్, ఈ నెల ఖర్చులు, బ్యాటరీ, అడుగులు మరియు వార్తలు చెప్పు";
    let p = post(port, "/v1/ask", json!({"question": q, "max_tools": 8}));
    let tools = tools_of(&p);
    for want in ["calendar.today", "bank.balance", "bank.spending", "news.headlines", "device.battery", "health.steps"] {
        assert!(tools.contains(&want.to_string()), "missing {want}: {tools:?}");
    }
    assert_eq!(p["lang"], "te");
    let titles: Vec<&str> = p["ui"]["sections"].as_array().unwrap().iter().map(|s| s["title"].as_str().unwrap()).collect();
    assert_eq!(titles, vec!["క్యాలెండర్ & పనులు", "డబ్బు & ఆర్థికం", "వెబ్ & వార్తలు", "పరికరం & సిస్టమ్", "ఆరోగ్యం & ఫిట్‌నెస్"]);
    // steps metric uses Indian grouping (8,432) and the speech has no raw digits left
    let speech = p["speech"].as_str().unwrap();
    assert!(speech.contains("ఎనిమిది వేల నాలుగు వందల ముప్పై రెండు"), "{speech}");
    assert!(!speech.chars().any(|c| c.is_ascii_digit()), "{speech}");
}

#[test]
fn hindi_question_gets_a_hindi_page() {
    if !have_python() {
        return eprintln!("skipped: python3 needed");
    }
    let port = start(|_| {});
    let p = post(port, "/v1/ask", json!({"question": "आज मौसम कैसा है और बैटरी कितनी है?"}));
    let tools = tools_of(&p);
    assert!(tools.contains(&"weather.current".into()) && tools.contains(&"device.battery".into()), "{tools:?}");
    assert_eq!(p["lang"], "hi");
    assert_eq!(p["ui"]["sections"][0]["title"], "मौसम");
    assert!(p["speech"].as_str().unwrap().contains("अठारह दशमलव चार"), "{}", p["speech"]);
    // explicit language wins over detection
    let p = post(port, "/v1/ask", json!({"question": "weather", "lang": "te"}));
    assert_eq!(p["ui"]["sections"][0]["title"], "వాతావరణం");
}

#[test]
fn speech_prepare_and_audio_endpoints() {
    if !have_python() {
        return eprintln!("skipped: python3 needed");
    }
    let dir = tmp("tts");
    let (script, log) = fake_engine(&dir);
    let s = script.display().to_string();
    let port =
        start(|c| c.tts = Some(Backend::Command(vec!["python3".into(), s.clone(), "{out}".into(), "{lang}".into(), "{text}".into()])));
    // prepare: no engine needed, language detected
    let v = post(port, "/v1/speech/prepare", json!({"input": "బ్యాలెన్స్ ₹2,431.75. రేపు 09:00 కి సమావేశం."}));
    assert_eq!(v["language"], "te");
    assert_eq!(v["sentences"].as_array().unwrap().len(), 2);
    assert!(v["text"].as_str().unwrap().contains("రెండు వేల నాలుగు వందల ముప్పై ఒకటి రూపాయలు డెబ్బై ఐదు పైసలు"), "{v}");
    // audio: one engine call per sentence, joined into one WAV
    let (st, wav) = req(port, "POST", "/v1/audio/speech", Some(&json!({"input": "వర్షం 60%. గొడుగు తీసుకెళ్ళండి!"})));
    assert_eq!(st, 200);
    assert_eq!(&wav[..4], b"RIFF");
    let calls = std::fs::read_to_string(&log).unwrap();
    assert_eq!(calls.lines().collect::<Vec<_>>(), vec!["te|వర్షం అరవై శాతం.", "te|గొడుగు తీసుకెళ్ళండి!"], "{calls}");
    // /v1/ask can speak its page; the audio is served next to the page
    let p = post(port, "/v1/ask", json!({"question": "ఫోన్ బ్యాటరీ ఎంత ఉంది?", "speak": true}));
    let url = p["speech_url"].as_str().unwrap_or_else(|| panic!("{p}"));
    let (st, wav) = req(port, "GET", url, None);
    assert!(st == 200 && &wav[..4] == b"RIFF");
    assert!(std::fs::read_to_string(&log).unwrap().contains("te|పరికరం & సిస్టమ్."), "page speech spoken in Telugu");
    // errors are reported
    let (st, _) = req(port, "POST", "/v1/audio/speech", Some(&json!({"input": "  "})));
    assert_eq!(st, 400);
    let bare = start(|_| {});
    let (st, b) = req(bare, "POST", "/v1/audio/speech", Some(&json!({"input": "x"})));
    assert!(st == 400 && String::from_utf8(b).unwrap().contains("--tts"));
}

#[test]
fn chat_reply_is_spoken_while_it_streams() {
    if !have_python() {
        return eprintln!("skipped: python3 needed");
    }
    let dir = tmp("chat");
    let (script, log) = fake_engine(&dir);
    let s = script.display().to_string();
    // no {out}: the engine "plays" the audio itself, so the server streams sentences to it
    let port = start(|c| c.tts = Some(Backend::Command(vec!["python3".into(), s.clone(), "-".into(), "{lang}".into(), "{text}".into()])));
    let r = post(
        port,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "నమస్కారం, ఏదైనా చెప్పు"}], "max_tokens": 48, "temperature": 1.0, "seed": 3, "speak": true}),
    );
    let content = r["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    // speech finishes in the background: wait for it
    let t0 = std::time::Instant::now();
    let mut spoken = String::new();
    while t0.elapsed().as_secs() < 20 {
        spoken = std::fs::read_to_string(&log).unwrap_or_default();
        if !content.trim().is_empty() && !spoken.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    eprintln!("reply {:?}\nspoken {:?}", content.chars().take(80).collect::<String>(), spoken.chars().take(200).collect::<String>());
    if !content.trim().is_empty() {
        assert!(spoken.lines().all(|l| l.starts_with("te|")), "spoken in Telugu: {spoken}");
        assert!(!spoken.is_empty(), "reply was not spoken: {content:?}");
    }
}

#[test]
fn telugu_strings_in_tool_calls_are_always_valid_utf8() {
    let port = start(|c| c.max_string = 7);
    let tools = json!([{"type": "function", "function": {"name": "send_sms", "description": "Send an SMS",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}}}]);
    for seed in 0..6 {
        let r = post(
            port,
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "అమ్మకు \"నేను బయలుదేరాను\" అని మెసేజ్ పంపు"}], "tools": tools, "tool_choice": "required",
                "temperature": 1.0, "seed": seed, "max_tokens": 600}),
        );
        let call = &r["choices"][0]["message"]["tool_calls"][0]["function"];
        let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap_or_else(|| panic!("{r}"))).unwrap();
        for k in ["to", "text"] {
            let v = args[k].as_str().unwrap();
            assert!(!v.contains('\u{FFFD}'), "seed {seed}: broken character in {k}: {v:?}");
            assert!(v.chars().count() <= 7, "cap counts characters: {v:?}");
        }
    }
}

#[test]
fn telugu_memory_recall_matches_inflected_forms() {
    let dir = tmp("mem");
    let port = start(|c| c.memory = Some(dir.clone()));
    post(port, "/v1/memory", json!({"text": "అమ్మ పుట్టినరోజు డిసెంబర్ పదిహేను", "importance": 0.9}));
    post(port, "/v1/memory", json!({"text": "బ్యాంక్ ఖాతా హైదరాబాద్ శాఖలో ఉంది", "importance": 0.9}));
    // different inflection of the same words
    let q: String = "అమ్మ పుట్టినరోజుకు".bytes().map(|b| format!("%{b:02X}")).collect();
    let (_, b) = req(port, "GET", &format!("/v1/memory?q={q}&k=1"), None);
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert!(v["results"][0]["text"].as_str().unwrap().contains("పుట్టినరోజు"), "{v}");
}
