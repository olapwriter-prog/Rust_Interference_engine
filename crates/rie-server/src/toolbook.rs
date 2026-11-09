//! Tool book: what the server has learned about the tools clients offer.
//!
//! * **Harvest** — every tool definition seen in a request is registered (name,
//!   description, parameter schema), so the server builds up a catalogue of tools over
//!   time, across clients and sessions.
//! * **Learn** — every tool result that comes back is attributed to the call that
//!   produced it: success/failure counts, the error messages a tool produces (its
//!   *limitations*) and argument sets that worked (its *expectations*).
//! * **Recall** — for a new request the most relevant tools are ranked by similarity to
//!   the request, and short learned notes about the offered tools are put in front of the
//!   model, so it knows what a tool cannot do before it calls it.
//!
//! Persisted as `DIR/tools.json` plus one Markdown card per tool in `DIR/tools/NAME.md`
//! (readable and editable by people: a `## Notes` section written by hand is kept and
//! shown to the model).

use rie_core::memory::embed;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

const MAX_ERRORS: usize = 5;
const MAX_EXAMPLES: usize = 3;

#[derive(Debug, Clone, Default)]
pub struct ToolEntry {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub calls: u32,
    pub failures: u32,
    /// distinct error messages, most recent last
    pub errors: Vec<String>,
    /// argument sets that produced a successful result, most recent last
    pub examples: Vec<String>,
    /// hand-written notes (from the Markdown card)
    pub notes: String,
}

impl ToolEntry {
    fn to_json(&self) -> Value {
        json!({"name": self.name, "description": self.description, "parameters": self.parameters, "calls": self.calls,
            "failures": self.failures, "errors": self.errors, "examples": self.examples})
    }
    fn from_json(v: &Value) -> Option<Self> {
        let strs = |k: &str| {
            v.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
        };
        Some(ToolEntry {
            name: v.get("name")?.as_str()?.to_string(),
            description: v.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
            parameters: v.get("parameters").cloned().unwrap_or(Value::Null),
            calls: v.get("calls").and_then(Value::as_u64).unwrap_or(0) as u32,
            failures: v.get("failures").and_then(Value::as_u64).unwrap_or(0) as u32,
            errors: strs("errors"),
            examples: strs("examples"),
            notes: String::new(),
        })
    }

    fn search_text(&self) -> String {
        let params: Vec<String> = self
            .parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|p| p.iter().map(|(k, v)| format!("{k} {}", v.get("description").and_then(Value::as_str).unwrap_or(""))).collect())
            .unwrap_or_default();
        format!("{} {} {} {}", self.name.replace('_', " "), self.name, self.description, params.join(" "))
    }

    /// One-paragraph note for the model, if anything was learned.
    pub fn note(&self) -> Option<String> {
        if self.calls == 0 && self.notes.is_empty() {
            return None;
        }
        let mut s = format!("- `{}`:", self.name);
        if self.calls > 0 {
            s.push_str(&format!(" {} calls, {} failed.", self.calls, self.failures));
        }
        if !self.notes.is_empty() {
            s.push_str(&format!(" {}", self.notes.trim().replace('\n', " ")));
        }
        if !self.errors.is_empty() {
            let errs: Vec<String> = self.errors.iter().rev().take(2).map(|e| format!("\"{e}\"")).collect();
            s.push_str(&format!(" Known failures: {}.", errs.join("; ")));
        }
        if let Some(ex) = self.examples.last() {
            s.push_str(&format!(" Worked with: {ex}."));
        }
        Some(s)
    }

    /// The Markdown card.
    pub fn card(&self) -> String {
        let mut s = format!("# {}\n\n{}\n\n", self.name, if self.description.is_empty() { "(no description)" } else { &self.description });
        s.push_str("## Parameters\n\n");
        let req: Vec<&str> = self
            .parameters
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        match self.parameters.get("properties").and_then(Value::as_object) {
            Some(p) if !p.is_empty() => {
                for (k, v) in p {
                    let ty = v.get("type").map(|t| t.to_string().trim_matches('"').to_string()).unwrap_or_else(|| "any".into());
                    let d = v.get("description").and_then(Value::as_str).unwrap_or("");
                    let r = if req.contains(&k.as_str()) { ", required" } else { "" };
                    s.push_str(&format!("- `{k}` ({ty}{r}) {d}\n"));
                }
            }
            _ => s.push_str("- none\n"),
        }
        let rate = if self.calls > 0 {
            format!(" ({:.0}% success)", 100.0 * (self.calls - self.failures) as f32 / self.calls as f32)
        } else {
            String::new()
        };
        s.push_str(&format!("\n## Track record\n\n{} calls, {} failed{rate}\n", self.calls, self.failures));
        s.push_str("\n## Limitations (learned from failures)\n\n");
        if self.errors.is_empty() {
            s.push_str("- none observed\n");
        }
        for e in &self.errors {
            s.push_str(&format!("- {e}\n"));
        }
        s.push_str("\n## Expectations (arguments that worked)\n\n");
        if self.examples.is_empty() {
            s.push_str("- none yet\n");
        }
        for e in &self.examples {
            s.push_str(&format!("- `{e}`\n"));
        }
        s.push_str("\n## Notes\n\n");
        s.push_str(if self.notes.trim().is_empty() {
            "<!-- write notes for the model here; they are kept -->\n"
        } else {
            self.notes.trim()
        });
        s.push('\n');
        s
    }
}

/// First informative line of an error, shortened.
fn error_summary(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let mut s: String = line.chars().take(160).collect();
    if line.chars().count() > 160 {
        s.push('…');
    }
    s
}

fn safe_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
        .collect::<String>()
        .trim_start_matches('.')
        .to_string()
}

/// The hand-written `## Notes` section of a tool's card on disk.
fn card_notes(dir: &std::path::Path, name: &str) -> Option<String> {
    let card = std::fs::read_to_string(dir.join("tools").join(format!("{}.md", safe_name(name)))).ok()?;
    let i = card.find("\n## Notes\n")?;
    let notes = card[i + 10..].trim();
    (!notes.is_empty() && !notes.starts_with("<!--")).then(|| notes.to_string())
}

#[derive(Default)]
pub struct ToolBook {
    dir: Option<PathBuf>,
    pub tools: BTreeMap<String, ToolEntry>,
}

impl ToolBook {
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Load (or start) a tool book in `dir`.
    pub fn open(dir: PathBuf) -> Self {
        let mut b = ToolBook { dir: Some(dir.clone()), tools: BTreeMap::new() };
        if let Ok(text) = std::fs::read_to_string(dir.join("tools.json")) {
            if let Ok(Value::Array(a)) = serde_json::from_str::<Value>(&text) {
                for v in &a {
                    if let Some(e) = ToolEntry::from_json(v) {
                        b.tools.insert(e.name.clone(), e);
                    }
                }
            }
        }
        // hand-written notes live in the cards
        for e in b.tools.values_mut() {
            e.notes = card_notes(&dir, &e.name).unwrap_or_default();
        }
        b
    }

    /// Register the tool definitions of a request (OpenAI `function` schemas).
    pub fn harvest(&mut self, schemas: &[Value]) -> bool {
        let mut changed = false;
        for s in schemas {
            let f = s.get("function").unwrap_or(s);
            let Some(name) = f.get("name").and_then(Value::as_str) else { continue };
            let desc = f.get("description").and_then(Value::as_str).unwrap_or("").to_string();
            let params = f.get("parameters").cloned().unwrap_or(Value::Null);
            let e = self.tools.entry(name.to_string()).or_insert_with(|| {
                changed = true;
                ToolEntry { name: name.to_string(), ..Default::default() }
            });
            if e.description != desc || e.parameters != params {
                e.description = desc;
                e.parameters = params;
                changed = true;
            }
        }
        changed
    }

    /// Learn from one tool result.
    pub fn observe(&mut self, name: &str, args: &str, result: &str, ok: bool) {
        let e = self.tools.entry(name.to_string()).or_insert_with(|| ToolEntry { name: name.to_string(), ..Default::default() });
        e.calls += 1;
        if ok {
            let a: String = args.chars().take(200).collect();
            if !a.is_empty() && !e.examples.contains(&a) {
                e.examples.push(a);
                if e.examples.len() > MAX_EXAMPLES {
                    e.examples.remove(0);
                }
            }
        } else {
            e.failures += 1;
            let summary = error_summary(result);
            if !summary.is_empty() {
                e.errors.retain(|x| x != &summary);
                e.errors.push(summary);
                if e.errors.len() > MAX_ERRORS {
                    e.errors.remove(0);
                }
            }
        }
    }

    /// Rank `names` by relevance to `query`; returns the best `k` (stable for ties).
    pub fn relevant(&self, query: &str, names: &[String], k: usize) -> Vec<String> {
        let q = embed(query);
        let mut scored: Vec<(f32, usize, &String)> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let text = self.tools.get(n).map(ToolEntry::search_text).unwrap_or_else(|| n.clone());
                let v = embed(&text);
                let sim: f32 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                // tools with a good track record get a small boost
                let prior = self
                    .tools
                    .get(n)
                    .map(|e| if e.calls > 0 { 0.05 * (e.calls - e.failures) as f32 / e.calls as f32 } else { 0.0 })
                    .unwrap_or(0.0);
                (sim + prior, i, n)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|x| x.2.clone()).collect()
    }

    /// Learned notes about the given tools (empty if nothing was learned).
    pub fn notes(&self, names: &[String]) -> String {
        names.iter().filter_map(|n| self.tools.get(n)).filter_map(ToolEntry::note).collect::<Vec<_>>().join("\n")
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.tools.values().map(ToolEntry::to_json).collect())
    }

    /// Persist `tools.json` and the Markdown cards.
    pub fn save(&mut self) {
        let Some(dir) = self.dir.clone() else { return };
        let cards = dir.join("tools");
        if std::fs::create_dir_all(&cards).is_err() {
            return;
        }
        let tmp = dir.join("tools.json.tmp");
        if std::fs::write(&tmp, serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, dir.join("tools.json"));
        }
        for e in self.tools.values_mut() {
            // notes edited on disk while the server runs win over the copy in memory
            if let Some(n) = card_notes(&dir, &e.name) {
                e.notes = n;
            }
            let _ = std::fs::write(cards.join(format!("{}.md", safe_name(&e.name))), e.card());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schemas() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"read_file","description":"Read a text file from the workspace","parameters":{"type":"object","properties":{"path":{"type":"string","description":"relative path"}},"required":["path"]}}}),
            json!({"type":"function","function":{"name":"web_search","description":"Search the web for current information","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}),
            json!({"type":"function","function":{"name":"send_email","description":"Send an email message to a recipient","parameters":{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}}}}}),
        ]
    }

    #[test]
    fn harvests_learns_recalls_and_persists() {
        let dir = std::env::temp_dir().join(format!("rie-toolbook-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut b = ToolBook::open(dir.clone());
        assert!(b.harvest(&schemas()));
        assert!(!b.harvest(&schemas()));
        let names: Vec<String> = b.tools.keys().cloned().collect();
        assert_eq!(b.relevant("search the web for today's news", &names, 1), vec!["web_search"]);
        assert_eq!(b.relevant("open the file README.md", &names, 1), vec!["read_file"]);
        assert_eq!(b.relevant("email Bob the report", &names, 1), vec!["send_email"]);
        assert!(b.notes(&names).is_empty());
        b.observe("read_file", r#"{"path":"/etc/passwd"}"#, "Error: path must be inside the workspace\nstack...", false);
        b.observe("read_file", r#"{"path":"README.md"}"#, "# Project", true);
        let notes = b.notes(&names);
        assert!(
            notes.contains("2 calls, 1 failed") && notes.contains("path must be inside the workspace") && notes.contains("README.md"),
            "{notes}"
        );
        b.save();
        // hand-written notes in a card are kept and shown to the model
        let card = dir.join("tools/read_file.md");
        let text = std::fs::read_to_string(&card).unwrap();
        assert!(text.contains("## Limitations (learned from failures)"));
        let edited = text.replace("<!-- write notes for the model here; they are kept -->", "Binary files are not supported.");
        std::fs::write(&card, edited).unwrap();
        let b2 = ToolBook::open(dir.clone());
        assert_eq!(b2.tools["read_file"].calls, 2);
        assert!(b2.notes(&["read_file".into()]).contains("Binary files are not supported."));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
