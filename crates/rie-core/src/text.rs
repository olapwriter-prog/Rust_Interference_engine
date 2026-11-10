//! Text utilities that are correct for Indian scripts.
//!
//! * [`is_word_char`] / [`words`]: word splitting that keeps Indic words whole. Rust's
//!   `char::is_alphanumeric` is false for the virama (్ / ्), nukta and ZWJ/ZWNJ, so a
//!   naive split shreds every conjunct consonant ("క్షేత్రం" → "క", "షేత", "రం").
//! * [`detect_script`]: the dominant script of a text, to pick UI language, speech
//!   voice and number reading.
//! * [`normalize`]: canonical composition (NFC) — the tokenizer applies it too.

/// Scripts the engine distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    Latin,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Other,
}

impl Script {
    pub fn of(c: char) -> Option<Script> {
        let u = c as u32;
        Some(match u {
            0x41..=0x5A | 0x61..=0x7A | 0xC0..=0x24F => Script::Latin,
            0x0900..=0x097F | 0xA8E0..=0xA8FF => Script::Devanagari,
            0x0980..=0x09FF => Script::Bengali,
            0x0A00..=0x0A7F => Script::Gurmukhi,
            0x0A80..=0x0AFF => Script::Gujarati,
            0x0B00..=0x0B7F => Script::Oriya,
            0x0B80..=0x0BFF => Script::Tamil,
            0x0C00..=0x0C7F => Script::Telugu,
            0x0C80..=0x0CFF => Script::Kannada,
            0x0D00..=0x0D7F => Script::Malayalam,
            _ if c.is_alphabetic() => Script::Other,
            _ => return None,
        })
    }

    /// BCP-47 language most associated with the script (Devanagari → Hindi).
    pub fn lang(self) -> &'static str {
        match self {
            Script::Latin => "en",
            Script::Devanagari => "hi",
            Script::Bengali => "bn",
            Script::Gurmukhi => "pa",
            Script::Gujarati => "gu",
            Script::Oriya => "or",
            Script::Tamil => "ta",
            Script::Telugu => "te",
            Script::Kannada => "kn",
            Script::Malayalam => "ml",
            Script::Other => "und",
        }
    }

    pub fn is_indic(self) -> bool {
        !matches!(self, Script::Latin | Script::Other)
    }
}

/// True for characters that belong inside a word, including Indic combining signs
/// (virama, nukta, vowel signs, anusvara…) and ZWJ/ZWNJ.
pub fn is_word_char(c: char) -> bool {
    if c.is_alphanumeric() || c == '_' || c == '\u{200C}' || c == '\u{200D}' {
        return true;
    }
    let u = c as u32;
    // Indic blocks, minus their punctuation (danda, double danda, abbreviation sign)
    (0x0900..=0x0DFF).contains(&u) && !matches!(u, 0x0964 | 0x0965 | 0x0970)
}

/// Words of a text (lower-cased), Indic-aware.
pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !is_word_char(c)).filter(|w| !w.is_empty()).map(|w| w.to_lowercase())
}

/// The script most letters of `text` are written in (`Latin` for empty text).
pub fn detect_script(text: &str) -> Script {
    let mut counts = [0usize; 11];
    let all = [
        Script::Latin,
        Script::Devanagari,
        Script::Bengali,
        Script::Gurmukhi,
        Script::Gujarati,
        Script::Oriya,
        Script::Tamil,
        Script::Telugu,
        Script::Kannada,
        Script::Malayalam,
        Script::Other,
    ];
    // count words (by their first letter), not characters: an Indic syllable takes
    // several code points, a Latin letter one
    for w in text.split(|c: char| !is_word_char(c)) {
        if let Some(s) = w.chars().find_map(Script::of) {
            counts[all.iter().position(|x| *x == s).unwrap()] += 1;
        }
    }
    // an Indic script wins as soon as it has a third of the words: code-mixed
    // sentences ("నా bank balance ఎంత?") are still that language
    let total: usize = counts.iter().sum();
    let (best_indic, n) = (1..10).map(|i| (i, counts[i])).max_by_key(|x| x.1).unwrap();
    if n > 0 && n * 3 >= total {
        return all[best_indic];
    }
    let (i, _) = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
    if total == 0 {
        Script::Latin
    } else {
        all[i]
    }
}

/// Built-in cross-lingual lexicon for phone tasks (Telugu, Hindi → English concepts).
const LEXICON: &str = include_str!("lexicon_indic.tsv");

/// English concept keywords for a word in an Indian language (prefix match on stems, so
/// agglutinated forms like "వాతావరణాన్ని" or "క్యాలెండర్‌లో" match). Empty if unknown.
pub fn concepts(word: &str) -> Vec<&'static str> {
    use std::sync::OnceLock;
    static ENTRIES: OnceLock<Vec<(String, Vec<&'static str>)>> = OnceLock::new();
    let entries = ENTRIES.get_or_init(|| {
        LEXICON
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut f = l.split('\t');
                let (_lang, stem, kw) = (f.next()?, f.next()?, f.next()?);
                Some((normalize(stem), kw.split_whitespace().collect()))
            })
            .collect()
    });
    let w = normalize(word);
    entries.iter().filter(|(stem, _)| w.starts_with(stem.as_str())).flat_map(|(_, kw)| kw.iter().copied()).collect()
}

/// Canonical composition (NFC).
pub fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfc().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indic_words_stay_whole() {
        let w: Vec<String> = words("క్షేత్రస్థాయిలో హైదరాబాద్‌లో, ప్రజల సమస్యలు। नमस्ते क़लम").collect();
        assert_eq!(w, vec!["క్షేత్రస్థాయిలో", "హైదరాబాద్\u{200C}లో", "ప్రజల", "సమస్యలు", "नमस्ते", "क़लम"]);
        assert_eq!(words("Hello, wörld_2!").collect::<Vec<_>>(), vec!["hello", "wörld_2"]);
    }

    #[test]
    fn detects_scripts_including_code_mixed_text() {
        assert_eq!(detect_script("ఈరోజు వాతావరణం ఎలా ఉంది?"), Script::Telugu);
        assert_eq!(detect_script("నా bank balance ఎంత?"), Script::Telugu);
        assert_eq!(detect_script("आज मौसम कैसा है"), Script::Devanagari);
        assert_eq!(detect_script("இன்று வானிலை"), Script::Tamil);
        assert_eq!(detect_script("What's the weather"), Script::Latin);
        assert_eq!(detect_script("123 ?"), Script::Latin);
        assert_eq!(Script::Telugu.lang(), "te");
    }

    #[test]
    fn lexicon_matches_inflected_forms() {
        assert!(concepts("వాతావరణం").contains(&"weather"));
        assert!(concepts("వాతావరణాన్ని").contains(&"weather"));
        assert!(concepts("క్యాలెండర్\u{200C}లో").contains(&"calendar"));
        assert!(concepts("बैटरी").contains(&"battery"));
        assert!(concepts("ఉంది").is_empty());
    }

    #[test]
    fn nfc_composes_decomposed_indic() {
        assert_eq!(normalize("హ\u{0C46}\u{0C56}"), "హై");
        assert_eq!(normalize("ம\u{0BC6}\u{0BBE}"), "மொ");
    }
}
