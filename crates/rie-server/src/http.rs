//! Minimal HTTP/1.1 server (no dependencies): keep-alive, Content-Length bodies,
//! JSON responses and Server-Sent Events.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

const MAX_BODY: usize = 64 << 20;

/// Read one request; `Ok(None)` when the client closed the connection.
pub fn read_request(r: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            return Ok(None);
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut req = Request { method, path, headers, body: Vec::new() };
    if let Some(n) = req.header("content-length").and_then(|v| v.parse::<usize>().ok()) {
        if n > MAX_BODY {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "body too large"));
        }
        req.body = vec![0; n];
        r.read_exact(&mut req.body)?;
    } else if req.header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        loop {
            let mut sz = String::new();
            r.read_line(&mut sz)?;
            let n = usize::from_str_radix(sz.trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
            if n == 0 {
                let mut end = String::new();
                r.read_line(&mut end)?;
                break;
            }
            if req.body.len() + n > MAX_BODY {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "body too large"));
            }
            let mut chunk = vec![0; n];
            r.read_exact(&mut chunk)?;
            req.body.extend_from_slice(&chunk);
            let mut crlf = String::new();
            r.read_line(&mut crlf)?;
        }
    }
    Ok(Some(req))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

pub fn write_json(w: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{body}",
        reason(status),
        body.len()
    )?;
    w.flush()
}

pub fn write_text(w: &mut TcpStream, status: u16, content_type: &str, body: &str) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{body}",
        reason(status),
        body.len()
    )?;
    w.flush()
}

/// Value of query parameter `key` in `path` (`/x?a=1&b=2`), percent-decoded.
pub fn query_param(path: &str, key: &str) -> Option<String> {
    let q = path.split_once('?')?.1;
    q.split('&').filter_map(|kv| kv.split_once('=')).find(|(k, _)| *k == key).map(|(_, v)| percent_decode(v))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16) {
                Ok(v) => {
                    out.push(v);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Start an SSE stream (the connection is closed when the stream ends).
pub fn start_sse(w: &mut TcpStream) -> std::io::Result<()> {
    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;
    w.flush()
}

pub fn sse(w: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(w, "data: {data}\n\n")?;
    w.flush()
}
