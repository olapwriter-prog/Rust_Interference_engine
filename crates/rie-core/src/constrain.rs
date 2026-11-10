//! Grammar-constrained decoding.
//!
//! Small models (0.5B–4B) often pick the right tool but then produce broken JSON, unknown
//! tool names or misspelled argument names. Instead of repairing the output afterwards,
//! [`JsonConstraint`] is consulted for every candidate token during sampling and only
//! tokens that keep the output a valid prefix of
//!
//! ```text
//! {"name": <one of the tool names>, "arguments": {<declared argument>: <value of the declared type>, ...}}
//! ```
//!
//! are allowed (or, in [`JsonConstraint::any_object`] mode, of any JSON object). The check
//! is a byte-level pushdown automaton; testing a candidate costs one small clone plus the
//! token's bytes.

use std::sync::Arc;

/// JSON type expected for an argument value (from the tool's JSON schema).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Any,
}

impl ArgType {
    pub fn from_schema(t: &str) -> Self {
        match t {
            "string" => ArgType::String,
            "number" => ArgType::Number,
            "integer" => ArgType::Integer,
            "boolean" => ArgType::Boolean,
            "array" => ArgType::Array,
            "object" => ArgType::Object,
            _ => ArgType::Any,
        }
    }
}

/// What the constraint needs to know about one tool.
#[derive(Debug, Clone)]
pub struct ToolSig {
    pub name: String,
    /// Declared arguments. Empty means "any keys".
    pub args: Vec<(String, ArgType)>,
    pub required: Vec<String>,
    /// Allowed string values for enum-typed arguments.
    pub enums: Vec<(String, Vec<String>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjSt {
    /// after `{`: key or `}`
    KeyOrEnd,
    /// after `,`: key
    Key,
    Colon,
    Value,
    CommaOrEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrSt {
    ValueOrEnd,
    Value,
    CommaOrEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Schema {
    Any,
    /// `{"name": …, "arguments": {…}}`
    ToolRoot,
    /// arguments object of tool `i`
    Args(usize),
}

#[derive(Debug, Clone)]
enum Frame {
    Obj { st: ObjSt, schema: Schema, seen: Vec<u16>, key: Vec<u8> },
    Arr { st: ArrSt },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NumPhase {
    Sign,
    Int0,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

#[derive(Debug, Clone)]
enum Lex {
    None,
    Str {
        is_key: bool,
        esc: bool,
        uni: u8,
        len: usize,
        /// accumulated bytes when the value must match one of `allowed`
        buf: Vec<u8>,
        allowed: Option<Arc<Vec<Vec<u8>>>>,
        /// UTF-8 continuation bytes still owed by the current character
        u8need: u8,
        /// lead byte whose first continuation byte is still to come (it has a narrower
        /// valid range: no overlong forms, no surrogates, nothing above U+10FFFF)
        u8lead: u8,
    },
    Num {
        phase: NumPhase,
        len: usize,
        integer: bool,
        /// digits in the exponent / integer part (bounded so values always fit an f64 / i64)
        exp_digits: u8,
        int_digits: u8,
    },
    Lit {
        word: &'static [u8],
        pos: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Root {
    Start,
    Inside,
    Done,
}

const MAX_WS_RUN: u8 = 3;
const MAX_STRING: usize = 8192;
const MAX_NUMBER: usize = 32;
const MAX_DEPTH: usize = 24;
const MAX_TOTAL: usize = 16384;

/// Incremental JSON validator with tool-call schema awareness. See the module docs.
#[derive(Debug, Clone)]
pub struct JsonConstraint {
    tools: Arc<Vec<ToolSig>>,
    tool_names: Arc<Vec<Vec<u8>>>,
    /// Key under which arguments appear: "arguments" (Hermes/Qwen) or "parameters" (Llama 3).
    args_key: Arc<Vec<Vec<u8>>>,
    frames: Vec<Frame>,
    lex: Lex,
    root: Root,
    ws_run: u8,
    root_schema: Schema,
    /// chosen tool (index) once its name string is complete
    chosen: Option<usize>,
    /// key whose value is being parsed (set at the colon)
    pending_key: Option<u16>,
    max_string: usize,
    /// bytes consumed so far and the soft budget after which the output winds down
    consumed: usize,
    max_total: usize,
}

impl JsonConstraint {
    /// Constrain to a call of one of `tools`. `args_key` is "arguments" or "parameters".
    pub fn tool_call(tools: Arc<Vec<ToolSig>>, args_key: &str) -> Self {
        let names = tools.iter().map(|t| t.name.as_bytes().to_vec()).collect();
        JsonConstraint {
            tool_names: Arc::new(names),
            tools,
            args_key: Arc::new(vec![args_key.as_bytes().to_vec()]),
            frames: Vec::new(),
            lex: Lex::None,
            root: Root::Start,
            ws_run: 0,
            root_schema: Schema::ToolRoot,
            chosen: None,
            pending_key: None,
            max_string: MAX_STRING,
            consumed: 0,
            max_total: MAX_TOTAL,
        }
    }

    /// Constrain to any JSON object (`response_format: json_object`).
    pub fn any_object() -> Self {
        JsonConstraint {
            tools: Arc::new(Vec::new()),
            tool_names: Arc::new(Vec::new()),
            args_key: Arc::new(Vec::new()),
            frames: Vec::new(),
            lex: Lex::None,
            root: Root::Start,
            ws_run: 0,
            root_schema: Schema::Any,
            chosen: None,
            pending_key: None,
            max_string: MAX_STRING,
            consumed: 0,
            max_total: MAX_TOTAL,
        }
    }

    /// Limit string values to `n` characters (after that the string must close). Guards against
    /// small models rambling inside an argument.
    pub fn with_max_string(mut self, n: usize) -> Self {
        self.max_string = n.max(1);
        self
    }

    /// Soft budget for the whole object. Once it is used up the output *winds down*: free
    /// strings must close, no new keys or array items are opened (except still-missing
    /// required arguments), nested containers close and scalars finish, so a model that
    /// keeps adding content still ends with a complete, valid object.
    pub fn with_max_total(mut self, n: usize) -> Self {
        self.max_total = n.max(16);
        self
    }

    fn winding(&self) -> bool {
        self.consumed > self.max_total
    }

    /// Required arguments of the innermost `Args` object that are still missing.
    fn missing_required(&self, schema: Schema, seen: &[u16]) -> Vec<Vec<u8>> {
        match schema {
            Schema::Args(i) => {
                let t = &self.tools[i];
                t.args
                    .iter()
                    .enumerate()
                    .filter(|(k, (n, _))| t.required.contains(n) && !seen.contains(&(*k as u16)))
                    .map(|(_, (n, _))| n.as_bytes().to_vec())
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// The root object has been closed.
    pub fn is_complete(&self) -> bool {
        self.root == Root::Done
    }

    /// Name of the tool being called, once known.
    pub fn tool_name(&self) -> Option<&str> {
        self.chosen.map(|i| self.tools[i].name.as_str())
    }

    /// Would appending `bytes` keep the output valid?
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false; // control tokens / empty pieces make no progress
        }
        let mut c = self.clone();
        c.feed(bytes)
    }

    /// Append `bytes`. Returns false (leaving the state unspecified) if they are invalid.
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.step(b))
    }

    fn keys_for(&self, schema: Schema, seen: &[u16]) -> Vec<Vec<u8>> {
        match schema {
            Schema::Any => Vec::new(),
            Schema::ToolRoot => {
                if seen.is_empty() {
                    vec![b"name".to_vec()]
                } else if seen.len() == 1 {
                    (*self.args_key).clone()
                } else {
                    Vec::new()
                }
            }
            Schema::Args(i) => self.tools[i]
                .args
                .iter()
                .enumerate()
                .filter(|(k, _)| !seen.contains(&(*k as u16)))
                .map(|(_, (n, _))| n.as_bytes().to_vec())
                .collect(),
        }
    }

    /// May the object with this schema be closed now?
    fn can_close(&self, schema: Schema, seen: &[u16]) -> bool {
        match schema {
            Schema::Any => true,
            Schema::ToolRoot => seen.len() == 2,
            Schema::Args(i) => {
                let t = &self.tools[i];
                t.required.iter().all(|r| t.args.iter().position(|(n, _)| n == r).is_none_or(|k| seen.contains(&(k as u16))))
            }
        }
    }

    fn key_index(&self, schema: Schema, key: &[u8], seen_len: usize) -> u16 {
        match schema {
            Schema::Args(i) => self.tools[i].args.iter().position(|(n, _)| n.as_bytes() == key).map(|k| k as u16).unwrap_or(u16::MAX),
            Schema::ToolRoot => seen_len as u16,
            Schema::Any => u16::MAX,
        }
    }

    /// Constraints for the value that is about to start inside the innermost frame.
    fn value_rule(&self) -> (ArgType, Option<Arc<Vec<Vec<u8>>>>, Schema) {
        if let Some(Frame::Obj { schema, .. }) = self.frames.last() {
            match (*schema, self.pending_key) {
                (Schema::ToolRoot, Some(0)) => return (ArgType::String, Some(self.tool_names.clone()), Schema::Any),
                (Schema::ToolRoot, Some(1)) => {
                    let s = self.chosen.map(Schema::Args).unwrap_or(Schema::Any);
                    return (ArgType::Object, None, s);
                }
                (Schema::Args(i), Some(k)) if k != u16::MAX => {
                    let (name, ty) = &self.tools[i].args[k as usize];
                    let en = self.tools[i]
                        .enums
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, v)| Arc::new(v.iter().map(|s| s.as_bytes().to_vec()).collect()));
                    return (*ty, en, Schema::Any);
                }
                _ => {}
            }
        }
        (ArgType::Any, None, Schema::Any)
    }

    fn is_ws(b: u8) -> bool {
        matches!(b, b' ' | b'\n' | b'\t' | b'\r')
    }

    fn ws(&mut self) -> bool {
        self.ws_run += 1;
        self.ws_run <= MAX_WS_RUN
    }

    fn step(&mut self, b: u8) -> bool {
        self.consumed += 1;
        let winding = self.winding();
        // ---- inside a lexeme ----
        match &mut self.lex {
            Lex::Str { is_key, esc, uni, len, buf, allowed, u8need, u8lead } => {
                // finish a multi-byte character first (Telugu letters are 3 bytes): only
                // continuation bytes, and the length cap waits until the character is whole,
                // so a string is never closed in the middle of a letter
                if *u8need > 0 {
                    let range = match std::mem::take(u8lead) {
                        0xE0 => 0xA0..=0xBF,
                        0xED => 0x80..=0x9F,
                        0xF0 => 0x90..=0xBF,
                        0xF4 => 0x80..=0x8F,
                        _ => 0x80..=0xBF,
                    };
                    if !range.contains(&b) {
                        return false;
                    }
                    *u8need -= 1;
                    if let Some(al) = allowed {
                        buf.push(b);
                        return al.iter().any(|a| a.starts_with(buf));
                    }
                    if *is_key {
                        buf.push(b);
                    }
                    return true;
                }
                // the cap counts characters, not bytes
                *len += 1;
                // at the length cap only the closing quote is allowed (never a dead end)
                // free text = any string that is not restricted to an enum (values and the keys
                // of free-form objects); enum strings are bounded by their allowed values
                let capped = allowed.is_none() && (*len > self.max_string || winding);
                if (capped || *len > MAX_STRING) && (b != b'"' || *esc || *uni > 0) {
                    return false;
                }
                if *uni > 0 {
                    if !b.is_ascii_hexdigit() {
                        return false;
                    }
                    *uni -= 1;
                    return true;
                }
                if *esc {
                    *esc = false;
                    if allowed.is_some() {
                        return false; // enum strings never need escapes
                    }
                    return match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => true,
                        b'u' => {
                            *uni = 4;
                            true
                        }
                        _ => false,
                    };
                }
                return match b {
                    b'"' => {
                        let key = *is_key;
                        if let Some(al) = allowed {
                            if !al.iter().any(|a| a == buf) {
                                return false;
                            }
                        }
                        let text = std::mem::take(buf);
                        self.lex = Lex::None;
                        return if key { self.end_key(text) } else { self.end_value_string(text) };
                    }
                    // enum strings (tool names, argument keys) never contain escapes: rejecting
                    // the backslash here avoids accepting a prefix with no valid continuation
                    b'\\' if allowed.is_some() => false,
                    b'\\' => {
                        *esc = true;
                        true
                    }
                    0..=0x1f => false,
                    // stray continuation byte or invalid lead byte: never valid UTF-8
                    0x80..=0xC1 | 0xF5..=0xFF => false,
                    _ => {
                        *u8need = match b {
                            0xC2..=0xDF => 1,
                            0xE0..=0xEF => 2,
                            0xF0..=0xF4 => 3,
                            _ => 0,
                        };
                        *u8lead = if *u8need > 0 { b } else { 0 };
                        if let Some(al) = allowed {
                            buf.push(b);
                            al.iter().any(|a| a.starts_with(buf))
                        } else {
                            if *is_key {
                                buf.push(b);
                            }
                            true
                        }
                    }
                };
            }
            Lex::Num { phase, len, integer, exp_digits, int_digits } => {
                // the length cap applies to the number's own characters, not to the
                // byte that terminates it (otherwise a number of exactly MAX_NUMBER
                // characters could never be closed)
                *len += 1;
                let too_long = *len > MAX_NUMBER;
                let integer = *integer;
                if b.is_ascii_digit() && !too_long {
                    match phase {
                        NumPhase::Exp | NumPhase::ExpSign | NumPhase::ExpDigits => {
                            *exp_digits += 1;
                            if *exp_digits > 2 {
                                return false;
                            }
                        }
                        NumPhase::Sign | NumPhase::Int => {
                            *int_digits += 1;
                            if integer && *int_digits > 18 {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
                let next = match (phase.clone(), b) {
                    (NumPhase::Sign, b'0') => Some(NumPhase::Int0),
                    (NumPhase::Sign, b'1'..=b'9') => Some(NumPhase::Int),
                    (NumPhase::Int, b'0'..=b'9') => Some(NumPhase::Int),
                    (NumPhase::Int0 | NumPhase::Int, b'.') if !integer => Some(NumPhase::Dot),
                    (NumPhase::Dot | NumPhase::Frac, b'0'..=b'9') => Some(NumPhase::Frac),
                    (NumPhase::Int0 | NumPhase::Int | NumPhase::Frac, b'e' | b'E') if !integer => Some(NumPhase::Exp),
                    (NumPhase::Exp, b'+' | b'-') => Some(NumPhase::ExpSign),
                    (NumPhase::Exp | NumPhase::ExpSign | NumPhase::ExpDigits, b'0'..=b'9') => Some(NumPhase::ExpDigits),
                    _ => None,
                };
                if let Some(n) = next {
                    if too_long {
                        return false;
                    }
                    *phase = n;
                    return true;
                }
                // number ends here if it is complete; then re-process b
                if !matches!(phase, NumPhase::Int0 | NumPhase::Int | NumPhase::Frac | NumPhase::ExpDigits) {
                    return false;
                }
                self.lex = Lex::None;
                if !self.end_value() {
                    return false;
                }
                return self.step(b);
            }
            Lex::Lit { word, pos } => {
                if word[*pos] != b {
                    return false;
                }
                *pos += 1;
                if *pos == word.len() {
                    self.lex = Lex::None;
                    return self.end_value();
                }
                return true;
            }
            Lex::None => {}
        }

        // ---- structural ----
        if Self::is_ws(b) {
            return self.ws();
        }
        self.ws_run = 0;
        match self.root {
            Root::Done => false,
            Root::Start => {
                if b != b'{' {
                    return false;
                }
                self.root = Root::Inside;
                let schema = self.root_schema;
                self.frames.push(Frame::Obj { st: ObjSt::KeyOrEnd, schema, seen: Vec::new(), key: Vec::new() });
                true
            }
            Root::Inside => self.step_structural(b),
        }
    }

    fn step_structural(&mut self, b: u8) -> bool {
        let Some(top) = self.frames.last().cloned() else { return false };
        match top {
            Frame::Obj { st, schema, seen, .. } => match (st, b) {
                (ObjSt::KeyOrEnd | ObjSt::Key, b'"') => {
                    let mut allowed = self.keys_for(schema, &seen);
                    if self.winding() && schema != Schema::ToolRoot {
                        let missing = self.missing_required(schema, &seen);
                        if !missing.is_empty() {
                            allowed = missing;
                        } else if st == ObjSt::KeyOrEnd {
                            return false; // close instead of opening a new key
                        }
                    }
                    let allowed = if schema == Schema::Any { None } else { Some(Arc::new(allowed)) };
                    if let Some(a) = &allowed {
                        if a.is_empty() {
                            return false;
                        }
                    }
                    self.lex = Lex::Str { is_key: true, esc: false, uni: 0, len: 0, buf: Vec::new(), allowed, u8need: 0, u8lead: 0 };
                    true
                }
                (ObjSt::KeyOrEnd | ObjSt::CommaOrEnd, b'}') => {
                    if !self.can_close(schema, &seen) {
                        return false;
                    }
                    self.frames.pop();
                    self.end_value()
                }
                (ObjSt::Colon, b':') => {
                    self.set_obj_state(ObjSt::Value);
                    true
                }
                (ObjSt::CommaOrEnd, b',') => {
                    // only if more keys remain for constrained objects
                    if schema != Schema::Any && self.keys_for(schema, &seen).is_empty() {
                        return false;
                    }
                    if self.winding() && schema != Schema::ToolRoot && self.can_close(schema, &seen) {
                        return false;
                    }
                    self.set_obj_state(ObjSt::Key);
                    true
                }
                (ObjSt::Value, _) => self.start_value(b),
                _ => false,
            },
            Frame::Arr { st } => match (st, b) {
                (ArrSt::ValueOrEnd | ArrSt::CommaOrEnd, b']') => {
                    if st == ArrSt::ValueOrEnd || st == ArrSt::CommaOrEnd {
                        self.frames.pop();
                        return self.end_value();
                    }
                    false
                }
                (ArrSt::CommaOrEnd, b',') => {
                    if self.winding() {
                        return false;
                    }
                    if let Some(Frame::Arr { st }) = self.frames.last_mut() {
                        *st = ArrSt::Value;
                    }
                    true
                }
                (ArrSt::ValueOrEnd | ArrSt::Value, _) => {
                    self.pending_key = None;
                    self.start_value(b)
                }
                _ => false,
            },
        }
    }

    fn set_obj_state(&mut self, s: ObjSt) {
        if let Some(Frame::Obj { st, .. }) = self.frames.last_mut() {
            *st = s;
        }
    }

    fn start_value(&mut self, b: u8) -> bool {
        let (ty, allowed, child) = self.value_rule();
        // when winding down, untyped values must be scalars (no new containers)
        let winding = self.winding();
        let ok_type = |t: ArgType| (ty == ArgType::Any && !(winding && matches!(t, ArgType::Object | ArgType::Array))) || ty == t;
        match b {
            b'"' if ok_type(ArgType::String) => {
                self.lex = Lex::Str { is_key: false, esc: false, uni: 0, len: 0, buf: Vec::new(), allowed, u8need: 0, u8lead: 0 };
                true
            }
            b'{' if ok_type(ArgType::Object) => {
                if self.frames.len() >= MAX_DEPTH {
                    return false;
                }
                self.frames.push(Frame::Obj { st: ObjSt::KeyOrEnd, schema: child, seen: Vec::new(), key: Vec::new() });
                true
            }
            b'[' if ok_type(ArgType::Array) => {
                if self.frames.len() >= MAX_DEPTH {
                    return false;
                }
                self.frames.push(Frame::Arr { st: ArrSt::ValueOrEnd });
                true
            }
            b'-' | b'0'..=b'9' if ok_type(ArgType::Number) || ty == ArgType::Integer => {
                let phase = match b {
                    b'-' => NumPhase::Sign,
                    b'0' => NumPhase::Int0,
                    _ => NumPhase::Int,
                };
                let int_digits = b.is_ascii_digit() as u8;
                self.lex = Lex::Num { phase, len: 1, integer: ty == ArgType::Integer, exp_digits: 0, int_digits };
                true
            }
            b't' if ok_type(ArgType::Boolean) => {
                self.lex = Lex::Lit { word: b"true", pos: 1 };
                true
            }
            b'f' if ok_type(ArgType::Boolean) => {
                self.lex = Lex::Lit { word: b"false", pos: 1 };
                true
            }
            b'n' if ty == ArgType::Any => {
                self.lex = Lex::Lit { word: b"null", pos: 1 };
                true
            }
            _ => false,
        }
    }

    fn end_key(&mut self, text: Vec<u8>) -> bool {
        let Some(Frame::Obj { schema, seen, .. }) = self.frames.last() else { return false };
        let (schema, seen_len) = (*schema, seen.len());
        let idx = self.key_index(schema, &text, seen_len);
        if let Some(Frame::Obj { st, seen, key, .. }) = self.frames.last_mut() {
            if idx != u16::MAX {
                seen.push(idx);
            } else {
                seen.push(u16::MAX - 1);
            }
            *key = text;
            *st = ObjSt::Colon;
        }
        self.pending_key = Some(idx);
        true
    }

    fn end_value_string(&mut self, text: Vec<u8>) -> bool {
        // completing the tool name selects the argument schema
        if let Some(Frame::Obj { schema: Schema::ToolRoot, .. }) = self.frames.last() {
            if self.pending_key == Some(0) {
                self.chosen = self.tool_names.iter().position(|n| *n == text);
            }
        }
        self.end_value()
    }

    /// A value finished: advance the parent container.
    fn end_value(&mut self) -> bool {
        match self.frames.last_mut() {
            None => {
                self.root = Root::Done;
                true
            }
            Some(Frame::Obj { st, .. }) => {
                *st = ObjSt::CommaOrEnd;
                true
            }
            Some(Frame::Arr { st }) => {
                *st = ArrSt::CommaOrEnd;
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Arc<Vec<ToolSig>> {
        Arc::new(vec![
            ToolSig {
                name: "fin.loan_emi".into(),
                args: vec![("principal".into(), ArgType::Number), ("months".into(), ArgType::Integer), ("note".into(), ArgType::String)],
                required: vec!["principal".into(), "months".into()],
                enums: vec![],
            },
            ToolSig {
                name: "browser.open".into(),
                args: vec![("url".into(), ArgType::String), ("mode".into(), ArgType::String)],
                required: vec!["url".into()],
                enums: vec![("mode".into(), vec!["tab".into(), "window".into()])],
            },
        ])
    }

    fn ok(s: &str) -> bool {
        let mut c = JsonConstraint::tool_call(tools(), "arguments");
        c.feed(s.as_bytes()) && c.is_complete()
    }

    #[test]
    fn accepts_valid_calls() {
        assert!(ok(r#"{"name": "fin.loan_emi", "arguments": {"principal": 1000000, "months": 240}}"#));
        assert!(ok(r#"{"name":"fin.loan_emi","arguments":{"months":12,"principal":2.5e5,"note":"a \"q\" é"}}"#));
        assert!(ok(r#"{"name": "browser.open", "arguments": {"url": "https://x.org", "mode": "tab"}}"#));
    }

    #[test]
    fn rejects_invalid_calls() {
        assert!(!ok(r#"{"name": "fin.loan", "arguments": {}}"#), "unknown tool");
        assert!(!ok(r#"{"name": "fin.loan_emi", "arguments": {"principal": 1}}"#), "missing required");
        assert!(!ok(r#"{"name": "fin.loan_emi", "arguments": {"principal": "1", "months": 2}}"#), "wrong type");
        assert!(!ok(r#"{"name": "fin.loan_emi", "arguments": {"principal": 1, "months": 2.5}}"#), "integer");
        assert!(!ok(r#"{"name": "fin.loan_emi", "arguments": {"principal": 1, "months": 2, "x": 1}}"#), "unknown arg");
        assert!(!ok(r#"{"name": "browser.open", "arguments": {"url": "u", "mode": "popup"}}"#), "enum");
        assert!(!ok(r#"{"arguments": {}, "name": "browser.open"}"#), "order");
        assert!(!ok(r#"{"name": "browser.open", "arguments": {"url": "u"}} x"#), "trailing");
        assert!(!ok(r#"{"name": "browser.open",      "arguments": {"url": "u"}}"#), "whitespace run");
    }

    #[test]
    fn prefix_checks_guide_generation() {
        let c = JsonConstraint::tool_call(tools(), "arguments");
        assert!(c.accepts(b"{\"na"));
        assert!(!c.accepts(b"{\"ar"));
        let mut c2 = c.clone();
        assert!(c2.feed(b"{\"name\": \"fin."));
        assert!(c2.accepts(b"loan"));
        assert!(!c2.accepts(b"open"));
        assert!(c2.feed(b"loan_emi\""));
        assert_eq!(c2.tool_name(), Some("fin.loan_emi"));
        assert!(c2.feed(b", \"arguments\": {\""));
        assert!(c2.accepts(b"pri"));
        assert!(!c2.accepts(b"url"));
        assert!(!c2.accepts(b"\\"), "escapes in enum keys would be a dead end");
    }

    #[test]
    fn string_cap_forces_closing_quote() {
        let mut c = JsonConstraint::tool_call(tools(), "arguments").with_max_string(4);
        assert!(c.feed(br#"{"name": "browser.open", "arguments": {"url": "abcd"#));
        assert!(!c.accepts(b"e"), "over the cap");
        assert!(c.accepts(b"\""), "closing quote must stay possible");
    }

    #[test]
    fn budget_winds_down_to_a_complete_object() {
        // a "model" that always prefers to keep the object growing still ends up complete
        let tools = Arc::new(vec![ToolSig {
            name: "use_tool".into(),
            args: vec![("name".into(), ArgType::String), ("arguments".into(), ArgType::Object)],
            required: vec!["name".into(), "arguments".into()],
            enums: vec![],
        }]);
        let mut c = JsonConstraint::tool_call(tools, "arguments").with_max_total(80).with_max_string(16);
        let prefs: &[&[u8]] = &[b"{", b"[", b"\"k", b"\"", b":", b",", b"1", b"x", b"]", b"}", b"name", b"arguments", b"use_tool"];
        let mut out = Vec::new();
        for _ in 0..2000 {
            if c.is_complete() {
                break;
            }
            let pick = if out.is_empty() {
                Some(&b"{\"name\": \"use_tool\", \"arguments\": "[..])
            } else {
                prefs.iter().copied().find(|p| c.accepts(p))
            };
            let p = pick.unwrap_or_else(|| panic!("dead end after {:?}", String::from_utf8_lossy(&out)));
            assert!(c.feed(p));
            out.extend_from_slice(p);
        }
        assert!(c.is_complete(), "did not finish: {}", String::from_utf8_lossy(&out));
        assert!(out.len() < 400, "{}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn strings_never_break_a_multibyte_character() {
        let sig = ToolSig { name: "f".into(), args: vec![("city".into(), ArgType::String)], required: vec![], enums: vec![] };
        let mut c = JsonConstraint::tool_call(Arc::new(vec![sig]), "arguments").with_max_string(3);
        assert!(c.feed(b"{\"name\":\"f\",\"arguments\":{\"city\":\""));
        let te = "తెలుగు".as_bytes(); // 6 characters × 3 bytes
        assert!(!c.accepts(&[0x80]), "stray continuation byte");
        assert!(c.feed(&te[..1]));
        assert!(!c.accepts(b"\""), "no closing quote mid-character");
        assert!(!c.accepts(b"a"), "no new character mid-character");
        // 3 characters fit the cap; the third may complete although the cap is reached
        assert!(c.feed(&te[1..9]));
        assert!(!c.accepts(&te[9..10]), "cap reached: no fourth character");
        assert!(c.feed(b"\"}}") && c.is_complete());
        // overlong forms and surrogates are not UTF-8 either
        for bad in [[0xE0u8, 0x80], [0xED, 0xA0], [0xF0, 0x80], [0xF4, 0x90]] {
            let mut c = JsonConstraint::tool_call(
                Arc::new(vec![ToolSig { name: "f".into(), args: vec![("x".into(), ArgType::String)], required: vec![], enums: vec![] }]),
                "arguments",
            );
            assert!(c.feed(b"{\"name\":\"f\",\"arguments\":{\"x\":\"") && c.feed(&bad[..1]));
            assert!(!c.accepts(&bad[1..]), "{bad:02X?}");
        }
    }

    #[test]
    fn number_at_the_length_cap_can_still_be_closed() {
        let sig = ToolSig {
            name: "f".into(),
            args: vec![("x".into(), ArgType::Number), ("y".into(), ArgType::Integer)],
            required: vec![],
            enums: vec![],
        };
        let mut c = JsonConstraint::tool_call(Arc::new(vec![sig]), "arguments");
        let digits = "5".repeat(MAX_NUMBER);
        assert!(c.feed(format!("{{\"name\":\"f\",\"arguments\":{{\"x\":{digits}").as_bytes()));
        assert!(!c.accepts(b"5"), "cap reached");
        assert!(c.accepts(b",") && c.accepts(b"}") && c.accepts(b" "), "terminators stay allowed");
        assert!(c.feed(b"}}") && c.is_complete());
    }

    #[test]
    fn any_object_mode() {
        let mut c = JsonConstraint::any_object();
        assert!(c.feed(br#"{"a": [1, -2.5e3, true, null, {"b": "c"}], "d": {}}"#) && c.is_complete());
        let mut c = JsonConstraint::any_object();
        assert!(!c.feed(b"[1]"));
        let mut c = JsonConstraint::any_object();
        assert!(!c.feed(br#"{"a": 01}"#));
        // every accepted number must also parse as f64 (no huge exponents)
        let mut c = JsonConstraint::any_object();
        assert!(c.feed(br#"{"a": -7.5e30"#));
        assert!(!c.accepts(b"9"));
    }
}
