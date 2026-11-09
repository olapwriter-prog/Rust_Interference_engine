use rie_core::gguf::writer::GgufWriter;
use rie_core::gguf::{GgufFile, Value};
use rie_core::tokenizer::{Tokenizer, Utf8Stream};

/// GPT-2 byte → unicode mapping (same as the tokenizer's internal table).
fn byte_char(b: u8) -> char {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for x in 0..256u32 {
        if !bs.contains(&x) {
            bs.push(x);
            cs.push(256 + n);
            n += 1;
        }
    }
    let i = bs.iter().position(|&x| x == b as u32).unwrap();
    char::from_u32(cs[i]).unwrap()
}

fn bpe_tokenizer(pre: &str) -> Tokenizer {
    let mut tokens: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
    let mut types = vec![Value::I32(1); 256];
    for t in ["he", "Ġt", "Ġthe", "ll", "llo"] {
        tokens.push(t.into());
        types.push(Value::I32(1));
    }
    for t in ["<|begin_of_text|>", "<|eot_id|>"] {
        tokens.push(t.into());
        types.push(Value::I32(3));
    }
    let merges = ["h e", "Ġ t", "Ġt he", "l l", "ll o"];
    let mut w = GgufWriter::new();
    w.kv("tokenizer.ggml.model", Value::String("gpt2".into()))
        .kv("tokenizer.ggml.pre", Value::String(pre.into()))
        .kv("tokenizer.ggml.tokens", Value::Array(tokens.into_iter().map(Value::String).collect()))
        .kv("tokenizer.ggml.token_type", Value::Array(types))
        .kv("tokenizer.ggml.merges", Value::Array(merges.iter().map(|m| Value::String(m.to_string())).collect()))
        .kv("tokenizer.ggml.bos_token_id", Value::U32(261))
        .kv("tokenizer.ggml.eos_token_id", Value::U32(262))
        .kv("tokenizer.ggml.add_bos_token", Value::Bool(true));
    Tokenizer::from_gguf(&GgufFile::from_bytes(w.to_bytes()).unwrap()).unwrap()
}

#[test]
fn byte_level_bpe_merges_and_roundtrip() {
    for pre in ["llama-bpe", "qwen2", "default", "smollm", "gpt-4o"] {
        let tok = bpe_tokenizer(pre);
        let ids = tok.encode("hello the", true, false);
        assert_eq!(ids, vec![261, 256, 260, 258], "pre={pre}"); // BOS, he, llo, Ġthe
        assert_eq!(tok.decode(&ids, false), "hello the");
        // arbitrary UTF-8 survives via byte tokens
        let s = "Привет 👋 123 — ok\n\ttab";
        assert_eq!(tok.decode(&tok.encode(s, false, false), false), s, "pre={pre}");
    }
}

#[test]
fn special_tokens_are_parsed_only_when_requested() {
    let tok = bpe_tokenizer("llama-bpe");
    let ids = tok.encode("hello<|eot_id|>the", false, true);
    assert_eq!(ids[2], 262);
    assert!(tok.is_stop(262));
    let ids = tok.encode("hello<|eot_id|>", false, false);
    assert!(!ids.contains(&262));
    assert_eq!(tok.decode(&ids, false), "hello<|eot_id|>");
}

#[test]
fn utf8_stream_reassembles_split_characters() {
    let mut s = Utf8Stream::default();
    let bytes = "a👋b".as_bytes();
    let mut out = String::new();
    for b in bytes {
        out += &s.push(&[*b]);
    }
    out += &s.flush();
    assert_eq!(out, "a👋b");
}
