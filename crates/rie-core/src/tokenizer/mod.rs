//! Tokenizers read from GGUF metadata: SentencePiece-style (`llama`: Llama 2, Gemma,
//! Mistral v1, Phi-3) and byte-level BPE (`gpt2`: Llama 3, Qwen, SmolLM, Phi-4, Granite).

mod bpe;
mod spm;

use crate::error::{bail, Error, Result};
use crate::gguf::GgufFile;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_i64(v: i64) -> Self {
        match v {
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

enum Kind {
    Spm(Box<spm::Spm>),
    Bpe(Box<bpe::Bpe>),
}

pub struct Tokenizer {
    kind: Kind,
    pub tokens: Vec<String>,
    pub types: Vec<TokenType>,
    pub token_to_id: HashMap<String, u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    /// Tokens that end generation (EOS, EOT, `<|im_end|>`, `<end_of_turn>`, …).
    pub stop_tokens: Vec<u32>,
    /// Special tokens matched literally in input text when `parse_special` is set,
    /// sorted longest first.
    specials: Vec<(String, u32)>,
    pub chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or("llama");
        let tokens: Vec<String> = g
            .get_arr("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Missing("tokenizer.ggml.tokens".into()))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let n = tokens.len();
        let types: Vec<TokenType> = match g.get_arr("tokenizer.ggml.token_type") {
            Some(a) => a.iter().map(|v| TokenType::from_i64(v.as_i64().unwrap_or(1))).collect(),
            None => vec![TokenType::Normal; n],
        };
        let scores: Vec<f32> = match g.get_arr("tokenizer.ggml.scores") {
            Some(a) => a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect(),
            None => vec![0.0; n],
        };
        let id = |k: &str| g.get_u64(k).map(|v| v as u32).filter(|&v| (v as usize) < n);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let eot = id("tokenizer.ggml.eot_token_id");
        let mut token_to_id = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }
        let kind = match model {
            "llama" | "gemma" => {
                let add_space_prefix = g.get_bool("tokenizer.ggml.add_space_prefix").unwrap_or(model == "llama");
                let unk = id("tokenizer.ggml.unknown_token_id");
                Kind::Spm(Box::new(spm::Spm::new(&tokens, &scores, &types, add_space_prefix, unk)))
            }
            "gpt2" => {
                let merges: Vec<String> = g
                    .get_arr("tokenizer.ggml.merges")
                    .ok_or_else(|| Error::Missing("tokenizer.ggml.merges".into()))?
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect();
                let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("default");
                Kind::Bpe(Box::new(bpe::Bpe::new(&merges, pre)?))
            }
            other => bail!(Unsupported, "tokenizer model '{other}'"),
        };
        let add_bos = g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(matches!(kind, Kind::Spm(_)));
        let add_eos = g.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);

        let mut specials: Vec<(String, u32)> = types
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, TokenType::Control | TokenType::UserDefined))
            .filter(|(i, _)| !tokens[*i].is_empty())
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));

        let mut stop_tokens: Vec<u32> = [eos, eot].into_iter().flatten().collect();
        for s in
            ["<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "<|eom_id|>", "<|end_of_text|>", "</s>", "<|return|>"]
        {
            if let Some(&t) = token_to_id.get(s) {
                if !stop_tokens.contains(&t) && matches!(types[t as usize], TokenType::Control | TokenType::UserDefined) {
                    stop_tokens.push(t);
                }
            }
        }
        Ok(Tokenizer {
            kind,
            tokens,
            types,
            token_to_id,
            bos,
            eos,
            eot,
            add_bos,
            add_eos,
            stop_tokens,
            specials,
            chat_template: g.get_str("tokenizer.chat_template").map(str::to_string),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    pub fn is_stop(&self, t: u32) -> bool {
        self.stop_tokens.contains(&t)
    }

    /// Tokenize `text`. With `parse_special`, control tokens written literally in the text
    /// (e.g. `<|im_start|>`) are mapped to their ids; otherwise they are tokenized as text.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        let mut first_text = true;
        let mut rest = text;
        while !rest.is_empty() {
            // find earliest special token occurrence
            let mut hit: Option<(usize, &str, u32)> = None;
            if parse_special {
                for (s, id) in &self.specials {
                    if let Some(i) = rest.find(s.as_str()) {
                        if hit.is_none_or(|(j, hs, _)| i < j || (i == j && s.len() > hs.len())) {
                            hit = Some((i, s, *id));
                        }
                    }
                }
            }
            let (chunk, special) = match hit {
                Some((i, s, id)) => (&rest[..i], Some((s.len(), id, i))),
                None => (rest, None),
            };
            if !chunk.is_empty() {
                self.encode_text(chunk, first_text, &mut out);
                first_text = false;
            }
            match special {
                Some((len, id, i)) => {
                    out.push(id);
                    rest = &rest[i + len..];
                }
                None => break,
            }
        }
        if add_special && self.add_eos {
            if let Some(e) = self.eos {
                out.push(e);
            }
        }
        out
    }

    fn encode_text(&self, text: &str, at_start: bool, out: &mut Vec<u32>) {
        match &self.kind {
            Kind::Spm(s) => s.encode(text, at_start, &self.token_to_id, out),
            Kind::Bpe(b) => b.encode(text, &self.token_to_id, out),
        }
    }

    /// Raw bytes of one token (may be a partial UTF-8 sequence). Control tokens decode to
    /// nothing unless `show_special`.
    pub fn decode_token(&self, id: u32, show_special: bool) -> Vec<u8> {
        let Some(text) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        match self.types[id as usize] {
            TokenType::Control | TokenType::Unknown | TokenType::Unused => {
                return if show_special { text.as_bytes().to_vec() } else { Vec::new() };
            }
            TokenType::UserDefined => return text.as_bytes().to_vec(),
            _ => {}
        }
        match &self.kind {
            Kind::Spm(_) => spm::decode(text, self.types[id as usize]),
            Kind::Bpe(_) => bpe::decode(text),
        }
    }

    pub fn decode(&self, ids: &[u32], show_special: bool) -> String {
        let mut bytes = Vec::new();
        for &t in ids {
            bytes.extend(self.decode_token(t, show_special));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Build a tokenizer from parts (tests / custom vocabularies).
    #[doc(hidden)]
    pub fn from_gguf_metadata_for_tests(g: &GgufFile) -> Result<Self> {
        Self::from_gguf(g)
    }
}

/// Incremental UTF-8 assembler for streaming output: tokens may split multi-byte
/// characters, so bytes are buffered until they form complete characters.
#[derive(Default)]
pub struct Utf8Stream {
    buf: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let s = s.to_string();
                self.buf.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() {
                    // invalid sequence: emit lossily and reset
                    let s = String::from_utf8_lossy(&self.buf).into_owned();
                    self.buf.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.buf[..valid]).into_owned();
                self.buf.drain(..valid);
                s
            }
        }
    }
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}
