//! Headless tools: programs or local HTTP services the server can call itself, declared
//! in a manifest (`--headless-tools FILE`):
//!
//! ```json
//! {"tools": [
//!   {"name": "weather.now", "description": "Current weather for a city", "category": "Weather",
//!    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
//!    "keywords": ["వాతావరణం", "मौसम"],
//!    "command": ["python3", "weather.py"], "timeout_ms": 5000},
//!   {"name": "calendar.today", "description": "Today's events",
//!    "http": "http://127.0.0.1:9000/calendar"}
//! ]}
//! ```
//!
//! A command receives the arguments as JSON on stdin and in `RIE_ARGS`; `{name}` in its
//! argv is replaced by the argument's value. It runs in the manifest's directory. Its
//! stdout is the result (JSON if it parses, text otherwise); a non-zero exit is an error.
//! An HTTP tool gets the arguments as a JSON POST body (plain `http://` only).

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Largest result kept from a tool (bytes).
pub const MAX_OUTPUT: usize = 256 * 1024;

#[derive(Debug, Clone)]
pub enum Exec {
    Command(Vec<String>),
    Http(String),
}

#[derive(Debug, Clone)]
pub struct HeadlessTool {
    pub name: String,
    pub description: String,
    pub category: Option<String>,
    /// extra words (any language) that select this tool, e.g. ["వాతావరణం", "मौसम"]
    pub keywords: Vec<String>,
    pub parameters: Value,
    pub exec: Exec,
    pub timeout: Duration,
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Error,
    Timeout,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Error => "error",
            Status::Timeout => "timeout",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub status: Status,
    /// parsed JSON result, or a string
    pub value: Value,
    pub error: Option<String>,
    pub truncated: bool,
    pub ms: u64,
}

impl HeadlessTool {
    /// OpenAI function schema for the model.
    pub fn schema(&self) -> Value {
        json!({"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}})
    }

    pub fn has_params(&self) -> bool {
        self.parameters.get("properties").and_then(Value::as_object).is_some_and(|p| !p.is_empty())
    }

    /// Run the tool with `args` (blocking, bounded by the tool's timeout).
    pub fn run(&self, args: &Value) -> Outcome {
        let t0 = Instant::now();
        let (status, raw, error, truncated) = match &self.exec {
            Exec::Command(argv) => run_command(argv, args, &self.dir, self.timeout),
            Exec::Http(url) => run_http(url, args, self.timeout),
        };
        let text = String::from_utf8_lossy(&raw).into_owned();
        let value = match serde_json::from_str::<Value>(text.trim()) {
            Ok(v) if !text.trim().is_empty() => v,
            _ => Value::String(text.trim().to_string()),
        };
        Outcome { status, value, error, truncated, ms: t0.elapsed().as_millis() as u64 }
    }
}

/// Load a manifest file.
pub fn load_manifest(path: &Path) -> Result<Vec<HeadlessTool>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let list = v.get("tools").and_then(Value::as_array).ok_or("manifest needs a \"tools\" array")?;
    let mut out = Vec::new();
    for t in list {
        let name = t.get("name").and_then(Value::as_str).ok_or("tool without a name")?.to_string();
        let exec = if let Some(c) = t.get("command") {
            let argv: Vec<String> = match c {
                Value::String(s) => s.split_whitespace().map(str::to_string).collect(),
                Value::Array(a) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
                _ => Vec::new(),
            };
            if argv.is_empty() {
                return Err(format!("{name}: empty command"));
            }
            Exec::Command(argv)
        } else if let Some(u) = t.get("http").and_then(Value::as_str) {
            Exec::Http(u.to_string())
        } else {
            return Err(format!("{name}: needs \"command\" or \"http\""));
        };
        if out.iter().any(|x: &HeadlessTool| x.name == name) {
            return Err(format!("duplicate tool {name}"));
        }
        out.push(HeadlessTool {
            description: t.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
            category: t.get("category").and_then(Value::as_str).map(str::to_string),
            keywords: t
                .get("keywords")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(|k| rie_core::text::normalize(k).to_lowercase()).collect())
                .unwrap_or_default(),
            parameters: t.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            exec,
            timeout: Duration::from_millis(t.get("timeout_ms").and_then(Value::as_u64).unwrap_or(10_000).clamp(10, 600_000)),
            dir: dir.clone(),
            name,
        });
    }
    Ok(out)
}

fn arg_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read at most `MAX_OUTPUT` bytes, draining (and dropping) the rest.
fn read_limited(mut r: impl Read) -> (Vec<u8>, bool) {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    let mut truncated = false;
    loop {
        match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let room = MAX_OUTPUT.saturating_sub(out.len());
                if n > room {
                    truncated = true;
                }
                out.extend_from_slice(&buf[..n.min(room)]);
            }
        }
    }
    (out, truncated)
}

fn run_command(argv: &[String], args: &Value, dir: &Path, timeout: Duration) -> (Status, Vec<u8>, Option<String>, bool) {
    let argv: Vec<String> = argv
        .iter()
        .map(|a| {
            let mut a = a.clone();
            if let Some(o) = args.as_object() {
                for (k, v) in o {
                    a = a.replace(&format!("{{{k}}}"), &arg_text(v));
                }
            }
            a
        })
        .collect();
    let payload = args.to_string();
    let mut child = match Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(dir)
        .env("RIE_ARGS", &payload)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return (Status::Error, Vec::new(), Some(format!("cannot start {}: {e}", argv[0])), false),
    };
    if let Some(mut stdin) = child.stdin.take() {
        std::thread::spawn(move || {
            let _ = stdin.write_all(payload.as_bytes());
        });
    }
    let out = child.stdout.take().map(|o| std::thread::spawn(move || read_limited(o)));
    let err = child.stderr.take().map(|o| std::thread::spawn(move || read_limited(o)));
    let t0 = Instant::now();
    let exit = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t0.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break None,
        }
    };
    let (stdout, truncated) = out.and_then(|h| h.join().ok()).unwrap_or_default();
    let (stderr, _) = err.and_then(|h| h.join().ok()).unwrap_or_default();
    match exit {
        None => (Status::Timeout, stdout, Some(format!("no result within {} ms", timeout.as_millis())), truncated),
        Some(s) if s.success() => (Status::Ok, stdout, None, truncated),
        Some(s) => {
            let msg = String::from_utf8_lossy(&stderr).trim().to_string();
            let msg = if msg.is_empty() { format!("exited with {s}") } else { msg.chars().take(500).collect() };
            (Status::Error, stdout, Some(msg), truncated)
        }
    }
}

fn run_http(url: &str, args: &Value, timeout: Duration) -> (Status, Vec<u8>, Option<String>, bool) {
    let Some(rest) = url.strip_prefix("http://") else {
        return (Status::Error, Vec::new(), Some("only http:// URLs are supported for headless tools".into()), false);
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
    let fail = |e: String| (Status::Error, Vec::new(), Some(e), false);
    use std::net::ToSocketAddrs;
    let Some(sa) = addr.to_socket_addrs().ok().and_then(|mut a| a.next()) else { return fail(format!("cannot resolve {addr}")) };
    let mut s = match std::net::TcpStream::connect_timeout(&sa, timeout) {
        Ok(s) => s,
        Err(e) => return fail(format!("connect {addr}: {e}")),
    };
    let _ = s.set_read_timeout(Some(timeout));
    let _ = s.set_write_timeout(Some(timeout));
    let body = args.to_string();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    if let Err(e) = s.write_all(req.as_bytes()) {
        return fail(format!("send: {e}"));
    }
    let t0 = Instant::now();
    let (resp, truncated) = read_limited(&mut s);
    if resp.is_empty() && t0.elapsed() >= timeout {
        return (Status::Timeout, Vec::new(), Some(format!("no result within {} ms", timeout.as_millis())), false);
    }
    let split = resp.windows(4).position(|w| w == b"\r\n\r\n");
    let Some(i) = split else { return fail("malformed HTTP response".into()) };
    let head = String::from_utf8_lossy(&resp[..i]).to_string();
    let code: u16 = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut payload = resp[i + 4..].to_vec();
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        payload = dechunk(&payload);
    }
    if (200..300).contains(&code) {
        (Status::Ok, payload, None, truncated)
    } else {
        let msg = format!("HTTP {code}: {}", String::from_utf8_lossy(&payload).chars().take(300).collect::<String>());
        (Status::Error, payload, Some(msg), truncated)
    }
}

fn dechunk(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(e) = b[i..].windows(2).position(|w| w == b"\r\n") {
        let n = usize::from_str_radix(String::from_utf8_lossy(&b[i..i + e]).trim(), 16).unwrap_or(0);
        if n == 0 {
            break;
        }
        let s = i + e + 2;
        out.extend_from_slice(&b[s..(s + n).min(b.len())]);
        i = s + n + 2;
        if i >= b.len() {
            break;
        }
    }
    out
}
