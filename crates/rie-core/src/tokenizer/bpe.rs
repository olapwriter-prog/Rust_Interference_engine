//! Byte-level BPE (GPT-2 style) with the pre-tokenizer regexes used by llama.cpp.

use crate::error::{Error, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DIGITS: &str = r"\p{N}";

pub struct Bpe {
    ranks: HashMap<(String, String), u32>,
    regexes: Vec<Regex>,
}

fn byte_to_unicode() -> &'static ([char; 256], HashMap<char, u8>) {
    static T: OnceLock<([char; 256], HashMap<char, u8>)> = OnceLock::new();
    T.get_or_init(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut fwd = ['\0'; 256];
        let mut inv = HashMap::new();
        for (b, c) in bs.iter().zip(&cs) {
            let ch = char::from_u32(*c).unwrap();
            fwd[*b as usize] = ch;
            inv.insert(ch, *b as u8);
        }
        (fwd, inv)
    })
}

impl Bpe {
    pub fn new(merges: &[String], pre: &str) -> Result<Self> {
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i as u32);
            }
        }
        let pats: Vec<&str> = match pre {
            "llama3" | "llama-bpe" | "llama-v3" | "smaug-bpe" | "falcon3" | "pixtral" => vec![LLAMA3],
            "qwen2" | "deepseek-r1-qwen" | "megrez" | "hunyuan" => vec![QWEN2],
            "smollm" | "starcoder" | "refact" | "codeshell" | "exaone" | "minerva-7b" | "command-r" => vec![DIGITS, GPT2],
            "gpt-4o" | "llama4" | "kimi-k2" => vec![GPT4O],
            "gpt2" | "gpt-2" | "default" | "olmo" | "jais" | "stablelm2" | "mpt" | "phi-2" | "granite" => vec![GPT2],
            _ => vec![LLAMA3],
        };
        let regexes = pats
            .into_iter()
            .map(|p| Regex::new(p).map_err(|e| Error::Format(format!("pre-tokenizer regex: {e}"))))
            .collect::<Result<_>>()?;
        Ok(Bpe { ranks, regexes })
    }

    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = vec![text];
        for re in &self.regexes {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for p in pieces {
                let mut last = 0;
                for m in re.find_iter(p).flatten() {
                    if m.start() > last {
                        next.push(&p[last..m.start()]);
                    }
                    if m.end() > m.start() {
                        next.push(m.as_str());
                    }
                    last = m.end();
                }
                if last < p.len() {
                    next.push(&p[last..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    pub fn encode(&self, text: &str, vocab: &HashMap<String, u32>, out: &mut Vec<u32>) {
        let (fwd, _) = byte_to_unicode();
        for word in self.pre_tokenize(text) {
            let mapped: String = word.bytes().map(|b| fwd[b as usize]).collect();
            if let Some(&id) = vocab.get(&mapped) {
                out.push(id);
                continue;
            }
            let mut parts: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
            loop {
                let mut best: Option<(u32, usize)> = None;
                for i in 0..parts.len().saturating_sub(1) {
                    if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                        if best.is_none_or(|(br, _)| r < br) {
                            best = Some((r, i));
                        }
                    }
                }
                let Some((_, i)) = best else { break };
                let merged = format!("{}{}", parts[i], parts[i + 1]);
                parts[i] = merged;
                parts.remove(i + 1);
            }
            for p in parts {
                match vocab.get(&p) {
                    Some(&id) => out.push(id),
                    None => {
                        // fall back to single byte-chars if present
                        for ch in p.chars() {
                            if let Some(&id) = vocab.get(&ch.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn decode(text: &str) -> Vec<u8> {
    let (_, inv) = byte_to_unicode();
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        match inv.get(&ch) {
            Some(&b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}
