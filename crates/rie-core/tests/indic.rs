//! Indian-language tokenization with the real tokenizers in llama.cpp's reference
//! vocabularies (`RIE_VOCAB_DIR`, skipped otherwise): Telugu round-trips exactly,
//! decomposed (NFD) input gives the same tokens as composed (NFC) input, streaming never
//! shows a broken character, and the measured fertility documents why model choice
//! matters for Telugu.

use rie_core::gguf::GgufFile;
use rie_core::tokenizer::{Tokenizer, Utf8Stream};

const TELUGU: &str = include_str!("../../../examples/text/telugu.txt");
const ENGLISH: &str = include_str!("../../../examples/text/english.txt");

fn vocabs() -> Vec<(String, Tokenizer)> {
    let Ok(dir) = std::env::var("RIE_VOCAB_DIR") else { return Vec::new() };
    let mut out = Vec::new();
    let mut paths: Vec<_> =
        std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "gguf")).collect();
    paths.sort();
    for p in paths {
        if let Ok(t) = Tokenizer::from_gguf(&GgufFile::open(&p).unwrap()) {
            out.push((p.file_name().unwrap().to_string_lossy().into_owned(), t));
        }
    }
    out
}

#[test]
fn telugu_round_trips_streams_cleanly_and_ignores_normalization_form() {
    let v = vocabs();
    if v.is_empty() {
        return eprintln!("RIE_VOCAB_DIR not set: skipping");
    }
    use unicode_normalization::UnicodeNormalization;
    let nfd: String = TELUGU.nfd().collect();
    assert_ne!(nfd, TELUGU, "the sample contains composable characters");
    for (name, tok) in &v {
        let st = tok.stats(TELUGU);
        assert!(st.round_trip, "{name}: Telugu does not round-trip");
        assert_eq!(tok.encode(&nfd, false, false), tok.encode(TELUGU, false, false), "{name}: NFD and NFC tokenize differently");
        // token-by-token streaming: no replacement characters, identical text
        let mut s = Utf8Stream::default();
        let mut streamed = String::new();
        for id in tok.encode(TELUGU, false, false) {
            streamed.push_str(&s.push(tok.token_bytes(id)));
        }
        let streamed = streamed.strip_prefix(' ').unwrap_or(&streamed);
        assert!(!streamed.contains('\u{FFFD}'), "{name}: broken character while streaming");
        assert_eq!(streamed, TELUGU, "{name}");
        let en = tok.stats(ENGLISH);
        eprintln!(
            "{name:32} Telugu {:5.2} tokens/word, English {:4.2} ({:.1}x)",
            st.fertility(),
            en.fertility(),
            st.fertility() / en.fertility()
        );
    }
}
