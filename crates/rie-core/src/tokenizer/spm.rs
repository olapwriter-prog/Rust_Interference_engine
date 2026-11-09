//! SentencePiece-style BPE using merge scores (llama.cpp `llm_tokenizer_spm`).

use super::TokenType;
use std::collections::{BinaryHeap, HashMap};

pub struct Spm {
    scores: Vec<f32>,
    byte_tokens: [Option<u32>; 256],
    add_space_prefix: bool,
    unk: Option<u32>,
}

#[derive(PartialEq)]
struct Bigram {
    score: f32,
    left: usize,
    size: usize,
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // highest score first, then leftmost
        self.score.partial_cmp(&o.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| o.left.cmp(&self.left))
    }
}

struct Sym {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl Spm {
    pub fn new(tokens: &[String], scores: &[f32], types: &[TokenType], add_space_prefix: bool, unk: Option<u32>) -> Self {
        let mut byte_tokens = [None; 256];
        for (i, t) in tokens.iter().enumerate() {
            if types.get(i) == Some(&TokenType::Byte) {
                if let Some(hex) = t.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        byte_tokens[b as usize] = Some(i as u32);
                    }
                }
            }
        }
        Spm { scores: scores.to_vec(), byte_tokens, add_space_prefix, unk }
    }

    pub fn encode(&self, text: &str, at_start: bool, vocab: &HashMap<String, u32>, out: &mut Vec<u32>) {
        let mut s = String::with_capacity(text.len() + 3);
        if self.add_space_prefix && at_start {
            s.push('\u{2581}');
        }
        for ch in text.chars() {
            s.push(if ch == ' ' { '\u{2581}' } else { ch });
        }
        let bytes = s.as_bytes();
        // initial symbols: UTF-8 characters
        let mut syms: Vec<Sym> = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let l = utf8_len(bytes[i]).min(bytes.len() - i);
            let idx = syms.len();
            syms.push(Sym { start: i, len: l, prev: idx.checked_sub(1), next: None });
            if idx > 0 {
                syms[idx - 1].next = Some(idx);
            }
            i += l;
        }
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &Vec<Sym>, left: usize| {
            if let Some(right) = syms[left].next {
                let st = syms[left].start;
                let end = syms[right].start + syms[right].len;
                if let Some(&id) = vocab.get(&s[st..end]) {
                    heap.push(Bigram { score: self.scores[id as usize], left, size: end - st });
                }
            }
        };
        for i in 0..syms.len() {
            try_add(&mut heap, &syms, i);
        }
        while let Some(bg) = heap.pop() {
            let left = bg.left;
            let Some(right) = syms[left].next else { continue };
            // stale entry?
            if syms[left].len == 0 || syms[left].len + syms[right].len != bg.size {
                continue;
            }
            syms[left].len += syms[right].len;
            syms[right].len = 0;
            let nn = syms[right].next;
            syms[left].next = nn;
            if let Some(nn) = nn {
                syms[nn].prev = Some(left);
            }
            if let Some(p) = syms[left].prev {
                try_add(&mut heap, &syms, p);
            }
            try_add(&mut heap, &syms, left);
        }
        let mut cur = if syms.is_empty() { None } else { Some(0) };
        while let Some(i) = cur {
            let sym = &syms[i];
            let piece = &s[sym.start..sym.start + sym.len];
            match vocab.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    for &b in piece.as_bytes() {
                        if let Some(id) = self.byte_tokens[b as usize].or(self.unk) {
                            out.push(id);
                        }
                    }
                }
            }
            cur = sym.next;
        }
    }
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

pub fn decode(text: &str, ty: TokenType) -> Vec<u8> {
    if ty == TokenType::Byte {
        if let Some(hex) = text.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            if let Ok(b) = u8::from_str_radix(hex, 16) {
                return vec![b];
            }
        }
    }
    text.replace('\u{2581}', " ").into_bytes()
}
