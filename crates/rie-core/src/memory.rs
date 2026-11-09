//! On-device episodic memory: every observation (user requests, actions taken, tool
//! results, notes) is appended to a compact journal on the phone and indexed with
//! 4-bit TurboQuant codes ([`TurboIndex`]), so it can be recalled later by similarity —
//! as a tool the model calls, or automatically when a new request looks related.
//!
//! Like human memory it forgets: the store has a capacity, and when it is exceeded the
//! observations with the lowest *retention* (importance, decayed with age, boosted every
//! time they are recalled) are dropped. Anything still stored can come back into the
//! context when it becomes relevant again.
//!
//! Storage (`DIR/observations.rmem`): a header followed by length-prefixed binary
//! records with a checksum (torn writes after a crash are detected and cut off). About
//! 40 bytes of overhead per observation plus its text; the index adds 132 bytes per
//! observation in RAM and is rebuilt on load. [`MemoryStore::to_markdown`] exports a
//! human-readable journal.
//!
//! Embeddings are hashed lexical features (words, word pairs, character trigrams):
//! instant, model-independent and good at matching tool names, file paths, error
//! messages and keywords, which is what agent memory mostly needs.

use crate::turbo::TurboIndex;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Embedding dimension (power of two for the TurboQuant rotation).
pub const EMBED_DIM: usize = 256;
const MAGIC: &[u8; 8] = b"RIEMEM1\n";
const MAX_TEXT: usize = 2000;
const INDEX_SEED: u64 = 0x6D65_6D6F_7279;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// what the user asked
    User,
    /// an action the model took (tool call or answer)
    Action,
    /// a tool result that came back
    Result,
    /// a fact or lesson worth keeping
    Note,
}

impl Kind {
    fn to_u8(self) -> u8 {
        self as u8
    }
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Kind::User,
            1 => Kind::Action,
            2 => Kind::Result,
            3 => Kind::Note,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::User => "user",
            Kind::Action => "action",
            Kind::Result => "result",
            Kind::Note => "note",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "user" => Kind::User,
            "action" => Kind::Action,
            "result" => Kind::Result,
            "note" => Kind::Note,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub id: u64,
    /// unix seconds
    pub ts: u64,
    pub kind: Kind,
    /// tool involved (empty if none)
    pub tool: String,
    /// outcome, when known
    pub ok: Option<bool>,
    /// 0..1, how much this is worth keeping
    pub importance: f32,
    /// how often it was recalled
    pub recalls: u32,
    pub text: String,
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

const STOP: &[&str] = &[
    "the", "a", "an", "to", "of", "and", "or", "is", "are", "was", "in", "on", "for", "it", "this", "that", "with", "be", "as", "at", "by",
    "i", "you", "me", "my", "we", "do", "can", "please", "what", "how",
];

/// Hashed lexical embedding (unit length, [`EMBED_DIM`] values).
pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    let mut add = |feat: &[u8], w: f32| {
        let h = fnv(feat);
        let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
        v[(h as usize) % EMBED_DIM] += sign * w;
    };
    let lower = text.to_lowercase();
    let words: Vec<&str> =
        lower.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|w| !w.is_empty() && !STOP.contains(w)).take(400).collect();
    for (i, w) in words.iter().enumerate() {
        add(format!("w:{w}").as_bytes(), 1.0);
        // parts of snake_case identifiers
        if w.contains('_') {
            for p in w.split('_').filter(|p| !p.is_empty()) {
                add(format!("w:{p}").as_bytes(), 0.6);
            }
        }
        let padded: Vec<char> = format!("^{w}$").chars().collect();
        let n = padded.len().saturating_sub(2).max(1) as f32;
        for t in padded.windows(3) {
            let s: String = t.iter().collect();
            add(format!("c:{s}").as_bytes(), 0.5 / n.sqrt());
        }
        if i > 0 {
            add(format!("b:{} {w}", words[i - 1]).as_bytes(), 0.7);
        }
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// A recalled observation with its relevance score.
#[derive(Debug, Clone)]
pub struct Recall {
    pub obs: Observation,
    pub score: f32,
}

pub struct MemoryStore {
    path: Option<PathBuf>,
    items: Vec<Observation>,
    index: TurboIndex,
    log: Option<File>,
    next_id: u64,
    capacity: usize,
    /// content hashes, to skip re-recording the same thing (conversations are resent
    /// in full with every request)
    seen: HashSet<u64>,
    /// half-life of an observation's importance, in seconds
    pub half_life: f64,
}

fn content_key(kind: Kind, tool: &str, text: &str) -> u64 {
    fnv(format!("{}\u{1}{tool}\u{1}{text}", kind.name()).as_bytes())
}

fn encode(o: &Observation) -> Vec<u8> {
    let mut p = Vec::with_capacity(32 + o.tool.len() + o.text.len());
    p.extend_from_slice(&o.id.to_le_bytes());
    p.extend_from_slice(&o.ts.to_le_bytes());
    p.push(o.kind.to_u8());
    p.push(match o.ok {
        None => 2,
        Some(true) => 1,
        Some(false) => 0,
    });
    p.extend_from_slice(&o.importance.to_le_bytes());
    p.extend_from_slice(&o.recalls.to_le_bytes());
    p.extend_from_slice(&(o.tool.len() as u16).to_le_bytes());
    p.extend_from_slice(o.tool.as_bytes());
    p.extend_from_slice(o.text.as_bytes());
    let mut rec = Vec::with_capacity(p.len() + 12);
    rec.extend_from_slice(&(p.len() as u32).to_le_bytes());
    rec.extend_from_slice(&p);
    rec.extend_from_slice(&fnv(&p).to_le_bytes());
    rec
}

fn decode(p: &[u8]) -> Option<Observation> {
    if p.len() < 28 {
        return None;
    }
    let u64at = |i: usize| u64::from_le_bytes(p[i..i + 8].try_into().unwrap());
    let tool_len = u16::from_le_bytes([p[26], p[27]]) as usize;
    if 28 + tool_len > p.len() {
        return None;
    }
    Some(Observation {
        id: u64at(0),
        ts: u64at(8),
        kind: Kind::from_u8(p[16])?,
        ok: match p[17] {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        importance: f32::from_le_bytes(p[18..22].try_into().unwrap()),
        recalls: u32::from_le_bytes(p[22..26].try_into().unwrap()),
        tool: String::from_utf8(p[28..28 + tool_len].to_vec()).ok()?,
        text: String::from_utf8(p[28 + tool_len..].to_vec()).ok()?,
    })
}

impl MemoryStore {
    /// A memory that lives only in RAM.
    pub fn in_memory(capacity: usize) -> Self {
        MemoryStore {
            path: None,
            items: Vec::new(),
            index: TurboIndex::new(EMBED_DIM, 4, INDEX_SEED),
            log: None,
            next_id: 1,
            capacity: capacity.max(8),
            seen: HashSet::new(),
            half_life: 14.0 * 86400.0,
        }
    }

    /// Open (or create) the journal in `dir`.
    pub fn open(dir: impl AsRef<Path>, capacity: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        let path = dir.as_ref().join("observations.rmem");
        let mut m = Self::in_memory(capacity);
        let mut valid_len = MAGIC.len() as u64;
        if let Ok(mut f) = File::open(&path) {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            if buf.len() >= MAGIC.len() && &buf[..MAGIC.len()] == MAGIC {
                let mut pos = MAGIC.len();
                while pos + 4 <= buf.len() {
                    let n = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                    if pos + 4 + n + 8 > buf.len() {
                        break;
                    }
                    let p = &buf[pos + 4..pos + 4 + n];
                    if fnv(p).to_le_bytes() != buf[pos + 4 + n..pos + 12 + n] {
                        break;
                    }
                    let Some(o) = decode(p) else { break };
                    pos += 12 + n;
                    valid_len = pos as u64;
                    // later records with the same id (compaction rewrites) replace earlier ones
                    m.next_id = m.next_id.max(o.id + 1);
                    m.insert(o);
                }
            }
        }
        let fresh = !path.exists();
        let mut f = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&path)?;
        if fresh || f.metadata()?.len() < MAGIC.len() as u64 {
            f.set_len(0)?;
            f.write_all(MAGIC)?;
        } else {
            // drop a torn tail
            f.set_len(valid_len)?;
        }
        drop(f);
        m.log = Some(OpenOptions::new().append(true).open(&path)?);
        m.path = Some(path);
        m.forget();
        Ok(m)
    }

    fn insert(&mut self, o: Observation) {
        self.seen.insert(content_key(o.kind, &o.tool, &o.text));
        self.index.add(&embed(&format!("{} {}", o.tool, o.text)));
        self.items.push(o);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn items(&self) -> &[Observation] {
        &self.items
    }
    /// RAM used by the quantized index.
    pub fn index_bytes(&self) -> usize {
        self.index.memory_bytes()
    }

    /// Record an observation. Returns its id, or `None` if the identical observation is
    /// already stored.
    pub fn record(&mut self, kind: Kind, tool: &str, ok: Option<bool>, importance: f32, text: &str) -> Option<u64> {
        let text = truncate(text.trim(), MAX_TEXT);
        let tool = truncate(tool, 200);
        if text.is_empty() || self.seen.contains(&content_key(kind, tool, text)) {
            return None;
        }
        let o = Observation {
            id: self.next_id,
            ts: now(),
            kind,
            tool: tool.to_string(),
            ok,
            importance: importance.clamp(0.0, 1.0),
            recalls: 0,
            text: text.to_string(),
        };
        self.next_id += 1;
        if let Some(f) = &mut self.log {
            let _ = f.write_all(&encode(&o));
        }
        let id = o.id;
        self.insert(o);
        if self.items.len() > self.capacity {
            self.forget();
        }
        Some(id)
    }

    /// Retention strength of an observation at time `t`.
    fn retention(&self, o: &Observation, t: u64) -> f64 {
        let age = t.saturating_sub(o.ts) as f64;
        let decay = 0.5f64.powf(age / self.half_life);
        let base = 0.2 + o.importance as f64;
        // failures are lessons: keep them a little longer
        let lesson = if o.ok == Some(false) { 0.3 } else { 0.0 };
        (base + lesson) * decay + 0.25 * (1.0 + o.recalls as f64).ln()
    }

    /// Recall the `k` observations most relevant to `query` (optionally only of `kind`),
    /// skipping those scoring below `min_score`. Recalled items are strengthened.
    pub fn recall(&mut self, query: &str, k: usize, kind: Option<Kind>, min_score: f32) -> Vec<Recall> {
        if self.items.is_empty() || k == 0 {
            return Vec::new();
        }
        let q = embed(query);
        let t = now();
        let sims = self.index.scores(&q);
        let mut cand: Vec<(usize, f32)> = sims
            .iter()
            .enumerate()
            .filter(|(i, &s)| s >= min_score && kind.is_none_or(|k| self.items[*i].kind == k))
            .map(|(i, &s)| {
                let r = self.retention(&self.items[i], t) as f32;
                (i, s * (0.85 + 0.15 * r.min(1.5)))
            })
            .collect();
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)));
        cand.truncate(k);
        cand.iter()
            .map(|&(i, score)| {
                self.items[i].recalls = self.items[i].recalls.saturating_add(1);
                Recall { obs: self.items[i].clone(), score }
            })
            .collect()
    }

    /// Drop the least-retained observations down to 90% of capacity and compact the
    /// journal. Called automatically when the capacity is exceeded.
    pub fn forget(&mut self) {
        if self.items.len() <= self.capacity {
            return;
        }
        let t = now();
        let keep_n = self.capacity * 9 / 10;
        let mut order: Vec<(usize, f64)> = self.items.iter().enumerate().map(|(i, o)| (i, self.retention(o, t))).collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<usize> = order[..keep_n].iter().map(|x| x.0).collect();
        keep.sort_unstable();
        let old = std::mem::take(&mut self.items);
        self.index = TurboIndex::new(EMBED_DIM, 4, INDEX_SEED);
        self.seen.clear();
        for i in keep {
            self.insert(old[i].clone());
        }
        let _ = self.compact();
    }

    /// Rewrite the journal with the current observations (persists recall counts).
    pub fn compact(&mut self) -> std::io::Result<()> {
        let Some(path) = self.path.clone() else { return Ok(()) };
        let tmp = path.with_extension("rmem.tmp");
        {
            let mut f = File::create(&tmp)?;
            let mut buf = MAGIC.to_vec();
            for o in &self.items {
                buf.extend_from_slice(&encode(o));
            }
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        self.log = Some(OpenOptions::new().append(true).open(&path)?);
        Ok(())
    }

    /// Human-readable journal (newest first, at most `limit` entries).
    pub fn to_markdown(&self, limit: usize) -> String {
        let mut s = format!(
            "# Memory\n\n{} observations, index {} KiB (TurboQuant 4-bit)\n\n",
            self.items.len(),
            self.index_bytes().div_ceil(1024)
        );
        for o in self.items.iter().rev().take(limit) {
            let ok = match o.ok {
                Some(true) => " ✓",
                Some(false) => " ✗",
                None => "",
            };
            let tool = if o.tool.is_empty() { String::new() } else { format!(" `{}`", o.tool) };
            let text = o.text.replace('\n', "\n  ");
            s.push_str(&format!("- **{}**{tool}{ok} · t={} · recalled {}×\n  {text}\n", o.kind.name(), o.ts, o.recalls));
        }
        s
    }
}

impl Drop for MemoryStore {
    fn drop(&mut self) {
        let _ = self.compact();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rie-mem-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn recalls_relevant_observations() {
        let mut m = MemoryStore::in_memory(100);
        m.record(
            Kind::Result,
            "read_file",
            Some(false),
            0.5,
            "Error: no such file: src/main.py (paths are relative to the workspace root)",
        );
        m.record(Kind::Action, "web_search", Some(true), 0.5, "searched for rust async runtime comparison");
        m.record(Kind::Note, "", None, 0.8, "The user prefers answers in British English");
        m.record(Kind::User, "", None, 0.3, "what is the weather in Paris tomorrow");
        let r = m.recall("read_file failed for main.py", 1, None, 0.1);
        assert_eq!(r[0].obs.tool, "read_file");
        let r = m.recall("weather forecast for Paris", 1, None, 0.1);
        assert!(r[0].obs.text.contains("Paris"));
        assert!(m.recall("weather forecast for Paris", 1, Some(Kind::Note), 0.5).is_empty());
        // duplicates are not stored twice
        assert!(m.record(Kind::Note, "", None, 0.8, "The user prefers answers in British English").is_none());
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn persists_survives_torn_writes_and_forgets() {
        let d = tmpdir("persist");
        {
            let mut m = MemoryStore::open(&d, 50).unwrap();
            for i in 0..20 {
                m.record(Kind::Action, "shell", Some(i % 2 == 0), 0.1, &format!("ran command number {i}"));
            }
            m.record(Kind::Note, "", None, 1.0, "important: deploy key lives in vault");
            m.recall("deploy key", 1, None, 0.0);
        }
        // simulate a crash in the middle of a write
        let path = d.join("observations.rmem");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[200, 0, 0, 0, 1, 2, 3]);
        std::fs::write(&path, bytes).unwrap();
        let mut m = MemoryStore::open(&d, 50).unwrap();
        assert_eq!(m.len(), 21);
        let r = m.recall("where is the deploy key", 1, None, 0.0);
        assert!(r[0].obs.text.contains("vault"));
        assert!(r[0].obs.recalls >= 2, "recall count persisted");
        let id = m.record(Kind::Note, "", None, 0.5, "after the crash").unwrap();
        assert!(id > 21);
        // exceed capacity: weakest observations are forgotten, the important note stays
        for i in 0..60 {
            m.record(Kind::Action, "shell", Some(true), 0.0, &format!("routine step {i}"));
        }
        assert!(m.len() <= 50);
        assert!(m.items().iter().any(|o| o.text.contains("vault")));
        let md = m.to_markdown(5);
        assert!(md.starts_with("# Memory"));
        drop(m);
        let m = MemoryStore::open(&d, 50).unwrap();
        assert!(m.len() <= 50 && m.items().iter().any(|o| o.text.contains("vault")));
        let _ = std::fs::remove_dir_all(&d);
    }
}
