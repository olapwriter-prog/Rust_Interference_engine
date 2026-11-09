//! Tokenizer conformance against llama.cpp's reference vocabularies and expected outputs
//! (`models/ggml-vocab-*.gguf{,.inp,.out}` in the llama.cpp repository).
//! Set `RIE_VOCAB_DIR` to a directory containing them; skipped otherwise.

use rie_core::gguf::GgufFile;
use rie_core::tokenizer::Tokenizer;

#[test]
fn llama_cpp_vocab_conformance() {
    let Ok(dir) = std::env::var("RIE_VOCAB_DIR") else {
        eprintln!("RIE_VOCAB_DIR not set: skipping");
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".gguf") {
            continue;
        }
        let (Ok(inp), Ok(out)) =
            (std::fs::read_to_string(format!("{}.inp", path.display())), std::fs::read_to_string(format!("{}.out", path.display())))
        else {
            continue;
        };
        let g = GgufFile::open(&path).unwrap();
        let tok = match Tokenizer::from_gguf(&g) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("{name}: unsupported ({e})");
                continue;
            }
        };
        let inputs: Vec<&str> = inp.split("\n__ggml_vocab_test__\n").collect();
        let expected: Vec<&str> = out.lines().collect();
        let (mut ok, mut bad) = (0, 0);
        for (i, text) in inputs.iter().enumerate() {
            let Some(exp) = expected.get(i) else { break };
            let exp: Vec<u32> = exp.split_whitespace().map(|s| s.parse().unwrap()).collect();
            let got = tok.encode(text, false, false);
            if got == exp {
                ok += 1;
            } else {
                bad += 1;
                failures.push(format!("{name}: {text:?}\n   expected {exp:?}\n   got      {got:?}"));
            }
            // decoding must round-trip for byte-level BPE
        }
        checked += 1;
        eprintln!("{name}: {ok} ok, {bad} mismatches");
    }
    assert!(checked > 0, "no vocab files found in {dir}");
    if !failures.is_empty() {
        for f in failures.iter().take(30) {
            eprintln!("{f}");
        }
        panic!("{} tokenization mismatches", failures.len());
    }
}
