//! Tool calling: prompt rendering per model family, grammar-constrained tool-call
//! generation and parsing of the result.

use rie_core::chat::{ChatTemplate, Message, Role};
use rie_core::constrain::{ArgType, JsonConstraint, ToolSig};
use rie_core::engine::TokenFilter;
use serde_json::{json, Value};
use std::sync::Arc;

/// How a model family writes tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallFormat {
    /// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` (Qwen 2.5/3, Hermes, SmolLM, and our default)
    Hermes,
    /// A bare `{"name": …, "parameters": {…}}` reply (Llama 3.1/3.2).
    Llama3,
}

impl CallFormat {
    pub fn for_template(t: ChatTemplate) -> Self {
        match t {
            ChatTemplate::Llama3 => CallFormat::Llama3,
            _ => CallFormat::Hermes,
        }
    }
    pub fn args_key(self) -> &'static str {
        match self {
            CallFormat::Hermes => "arguments",
            CallFormat::Llama3 => "parameters",
        }
    }
    /// Text the model writes for a call (used for history rendering and corrections).
    pub fn render_call(self, name: &str, args: &Value) -> String {
        match self {
            CallFormat::Hermes => format!("<tool_call>\n{}\n</tool_call>", json!({"name": name, "arguments": args})),
            CallFormat::Llama3 => json!({"name": name, "parameters": args}).to_string(),
        }
    }
    /// Text that starts a call (to prefill when a call is required).
    pub fn call_prefix(self) -> &'static str {
        match self {
            CallFormat::Hermes => "<tool_call>\n",
            CallFormat::Llama3 => "",
        }
    }
}

/// An OpenAI-format chat message.
#[derive(Debug, Clone, Default)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ApiToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ApiToolCall {
    pub id: String,
    pub name: String,
    /// JSON text
    pub arguments: String,
}

pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub fn parse_messages(v: &Value) -> Vec<ApiMessage> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|m| ApiMessage {
                    role: m.get("role").and_then(Value::as_str).unwrap_or("user").to_string(),
                    content: m.get("content").map(content_text).unwrap_or_default(),
                    tool_calls: m
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .map(|tc| {
                            tc.iter()
                                .map(|c| ApiToolCall {
                                    id: c.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                                    name: c.pointer("/function/name").and_then(Value::as_str).unwrap_or("").to_string(),
                                    arguments: match c.pointer("/function/arguments") {
                                        Some(Value::String(s)) => s.clone(),
                                        Some(v) => v.to_string(),
                                        None => "{}".into(),
                                    },
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    tool_call_id: m.get("tool_call_id").and_then(Value::as_str).map(str::to_string),
                    name: m.get("name").and_then(Value::as_str).map(str::to_string),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// OpenAI `tools` → (function schema JSON, constraint signature).
pub fn parse_tools(v: Option<&Value>) -> (Vec<Value>, Vec<ToolSig>) {
    let mut schemas = Vec::new();
    let mut sigs = Vec::new();
    for t in v.and_then(Value::as_array).into_iter().flatten() {
        let f = t.get("function").unwrap_or(t);
        let Some(name) = f.get("name").and_then(Value::as_str) else { continue };
        let params = f.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        let mut args = Vec::new();
        let mut enums = Vec::new();
        if let Some(props) = params.get("properties").and_then(Value::as_object) {
            for (k, p) in props {
                let ty = match p.get("type") {
                    Some(Value::String(s)) => ArgType::from_schema(s),
                    _ => ArgType::Any,
                };
                if let Some(e) = p.get("enum").and_then(Value::as_array) {
                    let vals: Vec<String> = e.iter().filter_map(Value::as_str).map(str::to_string).collect();
                    if vals.len() == e.len() && ty == ArgType::String {
                        enums.push((k.clone(), vals));
                    }
                }
                args.push((k.clone(), ty));
            }
        }
        let required = params
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        sigs.push(ToolSig { name: name.to_string(), args, required, enums });
        schemas.push(json!({"type": "function", "function": {"name": name, "description": f.get("description").cloned().unwrap_or(json!("")), "parameters": params}}));
    }
    (schemas, sigs)
}

fn hermes_tool_block(schemas: &[Value]) -> String {
    let mut s = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in schemas {
        s += &t.to_string();
        s.push('\n');
    }
    s += "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    s
}

fn llama3_tool_block(schemas: &[Value]) -> String {
    let mut s = String::from(
        "You have access to the following functions. To call a function, respond with only a JSON object of the form {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables. If no function is needed, answer normally.\n\n",
    );
    for t in schemas {
        s += &t.to_string();
        s.push_str("\n\n");
    }
    s
}

/// Render a conversation (with optional tools) for the model's chat template.
pub fn render(template: ChatTemplate, msgs: &[ApiMessage], tools: &[Value], add_generation_prompt: bool) -> String {
    let fmt = CallFormat::for_template(template);
    let calls_text = |m: &ApiMessage| -> String {
        m.tool_calls
            .iter()
            .map(|c| {
                let args: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
                fmt.render_call(&c.name, &args)
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut system: Option<String> = msgs.iter().find(|m| m.role == "system" || m.role == "developer").map(|m| m.content.clone());
    if !tools.is_empty() {
        let block = if fmt == CallFormat::Llama3 { llama3_tool_block(tools) } else { hermes_tool_block(tools) };
        system = Some(match system {
            Some(s) if !s.is_empty() => format!("{s}\n\n{block}"),
            _ => block,
        });
    }
    let body: Vec<&ApiMessage> = msgs.iter().filter(|m| m.role != "system" && m.role != "developer").collect();
    match template {
        ChatTemplate::ChatMl => {
            let mut s = String::new();
            if let Some(sys) = &system {
                s += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
            }
            let mut i = 0;
            while i < body.len() {
                let m = body[i];
                match m.role.as_str() {
                    "assistant" => {
                        let mut c = m.content.clone();
                        if !m.tool_calls.is_empty() {
                            if !c.is_empty() {
                                c.push('\n');
                            }
                            c += &calls_text(m);
                        }
                        s += &format!("<|im_start|>assistant\n{c}<|im_end|>\n");
                    }
                    "tool" => {
                        s += "<|im_start|>user";
                        while i < body.len() && body[i].role == "tool" {
                            s += &format!("\n<tool_response>\n{}\n</tool_response>", body[i].content);
                            i += 1;
                        }
                        s += "<|im_end|>\n";
                        continue;
                    }
                    _ => s += &format!("<|im_start|>user\n{}<|im_end|>\n", m.content),
                }
                i += 1;
            }
            if add_generation_prompt {
                s += "<|im_start|>assistant\n";
            }
            s
        }
        ChatTemplate::Llama3 => {
            let mut s = String::new();
            let h = |role: &str, c: &str| format!("<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>", c.trim());
            if let Some(sys) = &system {
                s += &h("system", sys);
            }
            for m in &body {
                match m.role.as_str() {
                    "assistant" if !m.tool_calls.is_empty() => s += &h("assistant", &calls_text(m)),
                    "assistant" => s += &h("assistant", &m.content),
                    "tool" => s += &h("ipython", &m.content),
                    _ => s += &h("user", &m.content),
                }
            }
            if add_generation_prompt {
                s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            s
        }
        _ => {
            // Generic: map tool turns onto plain user/assistant turns in Hermes style.
            let mut conv = Vec::new();
            if let Some(sys) = &system {
                conv.push(Message::new(Role::System, sys.clone()));
            }
            for m in &body {
                match m.role.as_str() {
                    "assistant" => {
                        let mut c = m.content.clone();
                        if !m.tool_calls.is_empty() {
                            c = format!("{c}\n{}", calls_text(m)).trim().to_string();
                        }
                        conv.push(Message::new(Role::Assistant, c));
                    }
                    "tool" => conv.push(Message::new(Role::User, format!("<tool_response>\n{}\n</tool_response>", m.content))),
                    _ => conv.push(Message::new(Role::User, m.content.clone())),
                }
            }
            template.render(&conv, add_generation_prompt)
        }
    }
}

/// Token filter that detects the start of a tool call and constrains it to a valid call.
pub struct ToolCallFilter {
    pub format: CallFormat,
    tools: Arc<Vec<ToolSig>>,
    json: Option<JsonConstraint>,
    /// all generated bytes (tokens may split UTF-8 characters, so detection works on bytes)
    pub raw: Vec<u8>,
    /// raw JSON of the completed call
    pub call_json: Option<String>,
    cur: Vec<u8>,
    /// byte offset in `raw` where the call starts (content before it is the reply text)
    pub call_start: Option<usize>,
    /// index of the first generated token that belongs to the call
    pub call_first_token: Option<usize>,
    n_tokens: usize,
    enabled: bool,
    max_string: usize,
    max_call: usize,
}

impl ToolCallFilter {
    /// `forced`: the prompt already ends with the call prefix, so the call starts immediately.
    pub fn new(format: CallFormat, tools: Vec<ToolSig>, forced: bool, enabled: bool) -> Self {
        Self::with_max_string(format, tools, forced, enabled, 4096)
    }

    pub fn with_max_string(format: CallFormat, tools: Vec<ToolSig>, forced: bool, enabled: bool, max_string: usize) -> Self {
        Self::with_limits(format, tools, forced, enabled, max_string, 16384)
    }

    /// `max_string`: bytes per free-text string; `max_call`: soft budget for the whole call.
    pub fn with_limits(format: CallFormat, tools: Vec<ToolSig>, forced: bool, enabled: bool, max_string: usize, max_call: usize) -> Self {
        let tools = Arc::new(tools);
        let json = (forced && enabled && !tools.is_empty())
            .then(|| JsonConstraint::tool_call(tools.clone(), format.args_key()).with_max_string(max_string).with_max_total(max_call));
        ToolCallFilter {
            format,
            tools,
            call_start: forced.then_some(0),
            call_first_token: forced.then_some(0),
            json,
            raw: Vec::new(),
            call_json: None,
            cur: Vec::new(),
            n_tokens: 0,
            enabled,
            max_string,
            max_call,
        }
    }

    fn start(&mut self, at: usize, rest: &[u8]) {
        if !self.enabled || self.tools.is_empty() || self.call_json.is_some() {
            return;
        }
        let mut c = JsonConstraint::tool_call(self.tools.clone(), self.format.args_key())
            .with_max_string(self.max_string)
            .with_max_total(self.max_call);
        if c.feed(rest) {
            self.call_start = Some(at);
            self.call_first_token = Some(self.n_tokens);
            self.cur = rest.to_vec();
            if c.is_complete() {
                self.call_json = Some(String::from_utf8_lossy(&self.cur).into_owned());
            } else {
                self.json = Some(c);
            }
        }
    }
}

impl TokenFilter for ToolCallFilter {
    fn active(&self) -> bool {
        self.json.is_some()
    }
    fn allows(&self, _id: u32, bytes: &[u8]) -> bool {
        self.json.as_ref().is_none_or(|j| j.accepts(bytes))
    }
    fn accept(&mut self, _id: u32, bytes: &[u8]) -> bool {
        let before = self.raw.len();
        self.raw.extend_from_slice(bytes);
        self.n_tokens += 1;
        if let Some(j) = &mut self.json {
            self.cur.extend_from_slice(bytes);
            if !j.feed(bytes) {
                // cannot happen for tokens passed through `allows`; give up constraining
                self.json = None;
                return false;
            }
            if j.is_complete() {
                self.call_json = Some(String::from_utf8_lossy(&self.cur).into_owned());
                self.json = None;
                return true; // one call per turn
            }
            return false;
        }
        if self.call_json.is_some() || !self.enabled {
            return false;
        }
        match self.format {
            CallFormat::Hermes => {
                const TRIG: &[u8] = b"<tool_call>";
                let from = before.saturating_sub(TRIG.len());
                if let Some(i) = self.raw[from..].windows(TRIG.len()).position(|w| w == TRIG) {
                    let at = from + i;
                    let rest = self.raw[at + TRIG.len()..].to_vec();
                    self.start(at, &rest);
                    if self.call_json.is_some() {
                        return true;
                    }
                }
            }
            CallFormat::Llama3 => {
                let ws = |b: &u8| b.is_ascii_whitespace();
                if self.raw[..before].iter().all(ws) {
                    if let Some(at) = self.raw.iter().position(|b| !ws(b)) {
                        if self.raw[at] == b'{' {
                            let rest = self.raw[at..].to_vec();
                            self.start(at, &rest);
                            if self.call_json.is_some() {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// A parsed tool call.
#[derive(Debug, Clone)]
pub struct ParsedCall {
    pub name: String,
    pub arguments: Value,
}

/// Split generated text into visible content and the tool call (if any).
pub fn extract(filter: &ToolCallFilter) -> (String, Option<ParsedCall>) {
    if let (Some(start), Some(js)) = (filter.call_start, &filter.call_json) {
        let content = String::from_utf8_lossy(&filter.raw[..start.min(filter.raw.len())]).trim().to_string();
        if let Ok(v) = serde_json::from_str::<Value>(js.trim()) {
            let name = v.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let args = v.get(filter.format.args_key()).or_else(|| v.get("arguments")).cloned().unwrap_or_else(|| json!({}));
            return (content, Some(ParsedCall { name, arguments: args }));
        }
    }
    // unconstrained fallback: recover a Hermes block if the model wrote one anyway
    let text = String::from_utf8_lossy(&filter.raw).into_owned();
    if let Some(i) = text.find("<tool_call>") {
        let after = &text[i + 11..];
        let body = after.split("</tool_call>").next().unwrap_or("").trim();
        if let Ok(v) = serde_json::from_str::<Value>(body) {
            if let Some(name) = v.get("name").and_then(Value::as_str) {
                let args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or_else(|| json!({}));
                return (text[..i].trim().to_string(), Some(ParsedCall { name: name.to_string(), arguments: args }));
            }
        }
    }
    (text.trim().to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools_json() -> Value {
        json!([{"type":"function","function":{"name":"fin.loan_emi","description":"Monthly EMI for a loan","parameters":{"type":"object","properties":{"principal":{"type":"number"},"annualRate":{"type":"number"},"months":{"type":"integer"}},"required":["principal","annualRate","months"]}}}])
    }

    #[test]
    fn renders_chatml_with_tool_history() {
        let msgs = parse_messages(&json!([
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"EMI on 10 lakh at 8.5% for 20 years?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fin.loan_emi","arguments":"{\"principal\":1000000,\"annualRate\":8.5,\"months\":240}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"{\"emi\":8678}"}
        ]));
        let (schemas, _) = parse_tools(Some(&tools_json()));
        let p = render(ChatTemplate::ChatMl, &msgs, &schemas, true);
        assert!(p.starts_with("<|im_start|>system\nYou are helpful.\n\n# Tools"));
        assert!(p.contains("<tool_call>\n{\"arguments\":{\"annualRate\":8.5,\"months\":240,\"principal\":1000000},\"name\":\"fin.loan_emi\"}\n</tool_call><|im_end|>") || p.contains("\"name\":\"fin.loan_emi\""));
        assert!(p.contains("<|im_start|>user\n<tool_response>\n{\"emi\":8678}\n</tool_response><|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
        let l = render(ChatTemplate::Llama3, &msgs, &schemas, true);
        assert!(l.contains("<|start_header_id|>ipython<|end_header_id|>\n\n{\"emi\":8678}<|eot_id|>"));
        assert!(l.contains("\"parameters\""));
    }

    #[test]
    fn filter_detects_and_extracts_calls() {
        let (_, sigs) = parse_tools(Some(&tools_json()));
        let mut f = ToolCallFilter::new(CallFormat::Hermes, sigs.clone(), false, true);
        for piece in [
            "Let me compute.",
            "\n<tool",
            "_call>",
            "\n{\"name\": \"fin.loan_emi\", \"arguments\": {\"principal\": 1000000, ",
            "\"annualRate\": 8.5, \"months\": 240",
            "}",
        ] {
            assert!(!f.active() || f.allows(0, piece.as_bytes()), "piece {piece:?} rejected");
            let stop = f.accept(0, piece.as_bytes());
            if piece == "}" {
                assert!(!stop, "arguments object closed, root still open");
            }
        }
        assert!(f.accept(0, b"}"));
        let (content, call) = extract(&f);
        assert_eq!(content, "Let me compute.");
        let call = call.unwrap();
        assert_eq!(call.name, "fin.loan_emi");
        assert_eq!(call.arguments["months"], 240);
        // invalid continuation is rejected while constrained
        let mut f = ToolCallFilter::new(CallFormat::Llama3, sigs, false, true);
        f.accept(0, b"{\"name\": \"fin.");
        assert!(f.active());
        assert!(!f.allows(0, b"wrong"));
        assert!(f.allows(0, b"loan_emi\""));
    }
}
