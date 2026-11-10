//! Text preparation for speech (TTS), for Indian languages.
//!
//! A TTS voice reads what it is given: "₹2,431.75" or "18:30" come out wrong or in
//! English unless they are written out in the target language first. This module does
//! that front-end work, the "text normalization" stage of production Indic TTS systems
//! (Sarvam's Bulbul documents the same: numbers, currency, dates, Indian number formats,
//! code-mixed text):
//!
//! * [`verbalize`] — numbers (Indian system: లక్ష / लाख, కోటి / करोड़), decimals,
//!   negatives, percentages, currency (₹ Rs INR, $ USD, € EUR), times (HH:MM) and °C,
//!   written out in Telugu or Hindi. Other languages keep digits (the voice reads them).
//! * [`sentences`] — split text into speakable chunks (। ॥ . ? ! and newlines, capped
//!   in length) so speech can start after the first sentence instead of the whole reply.
//!
//! The Telugu and Hindi wordings follow standard usage; corrections from native
//! speakers are welcome (the tables are plain data at the top of the file).

const TE_UNITS: [&str; 20] = [
    "సున్నా",
    "ఒకటి",
    "రెండు",
    "మూడు",
    "నాలుగు",
    "ఐదు",
    "ఆరు",
    "ఏడు",
    "ఎనిమిది",
    "తొమ్మిది",
    "పది",
    "పదకొండు",
    "పన్నెండు",
    "పదమూడు",
    "పద్నాలుగు",
    "పదిహేను",
    "పదహారు",
    "పదిహేడు",
    "పద్దెనిమిది",
    "పందొమ్మిది",
];
const TE_TENS: [&str; 10] = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"];

const HI_0_99: [&str; 100] = [
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उन्यासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
];

/// Language-specific words around numbers.
struct Words {
    point: &'static str,
    minus: &'static str,
    percent: &'static str,
    rupee_one: &'static str,
    rupees: &'static str,
    paise: &'static str,
    dollars: &'static str,
    euros: &'static str,
    degrees_c: &'static str,
    hours: &'static str,
    /// "H hours M minutes" connector forms
    hours_join: &'static str,
    minutes: &'static str,
}

const TE_WORDS: Words = Words {
    point: "పాయింట్",
    minus: "మైనస్",
    percent: "శాతం",
    rupee_one: "రూపాయి",
    rupees: "రూపాయలు",
    paise: "పైసలు",
    dollars: "డాలర్లు",
    euros: "యూరోలు",
    degrees_c: "డిగ్రీల సెల్సియస్",
    hours: "గంటలు",
    hours_join: "గంటల",
    minutes: "నిమిషాలు",
};

const HI_WORDS: Words = Words {
    point: "दशमलव",
    minus: "माइनस",
    percent: "प्रतिशत",
    rupee_one: "रुपया",
    rupees: "रुपये",
    paise: "पैसे",
    dollars: "डॉलर",
    euros: "यूरो",
    degrees_c: "डिग्री सेल्सियस",
    hours: "बजे",
    hours_join: "बजकर",
    minutes: "मिनट",
};

fn te_below_100(n: u64) -> String {
    if n < 20 {
        TE_UNITS[n as usize].to_string()
    } else if n % 10 == 0 {
        TE_TENS[(n / 10) as usize].to_string()
    } else {
        format!("{} {}", TE_TENS[(n / 10) as usize], TE_UNITS[(n % 10) as usize])
    }
}

fn te_below_1000(n: u64) -> String {
    let (h, r) = (n / 100, n % 100);
    match (h, r) {
        (0, r) => te_below_100(r),
        (1, 0) => "వంద".into(),
        (1, r) => format!("నూట {}", te_below_100(r)),
        (h, 0) => format!("{} వందలు", TE_UNITS[h as usize]),
        (h, r) => format!("{} వందల {}", TE_UNITS[h as usize], te_below_100(r)),
    }
}

/// A whole number in Telugu words (Indian system: వెయ్యి, లక్ష, కోటి).
pub fn te_number(n: u64) -> String {
    if n < 1000 {
        return te_below_1000(n);
    }
    let crore = n / 10_000_000;
    let lakh = n / 100_000 % 100;
    let thousand = n / 1000 % 100;
    let rest = n % 1000;
    let mut parts: Vec<String> = Vec::new();
    // each group: standalone form when nothing follows, joining form otherwise
    let tail_after = |level: u32| match level {
        3 => lakh > 0 || thousand > 0 || rest > 0,
        2 => thousand > 0 || rest > 0,
        _ => rest > 0,
    };
    if crore > 0 {
        let more = tail_after(3);
        parts.push(match crore {
            1 => "ఒక కోటి".into(),
            c => format!("{} {}", te_number(c), if more { "కోట్ల" } else { "కోట్లు" }),
        });
    }
    if lakh > 0 {
        let more = tail_after(2);
        parts.push(match lakh {
            1 => "ఒక లక్ష".into(),
            l => format!("{} {}", te_below_100(l), if more { "లక్షల" } else { "లక్షలు" }),
        });
    }
    if thousand > 0 {
        let more = tail_after(1);
        parts.push(match thousand {
            1 => "వెయ్యి".into(),
            t => format!("{} {}", te_below_100(t), if more { "వేల" } else { "వేలు" }),
        });
    }
    if rest > 0 {
        parts.push(te_below_1000(rest));
    }
    parts.join(" ")
}

/// A whole number in Hindi words (Indian system: हज़ार, लाख, करोड़).
pub fn hi_number(n: u64) -> String {
    if n < 100 {
        return HI_0_99[n as usize].to_string();
    }
    let crore = n / 10_000_000;
    let lakh = n / 100_000 % 100;
    let thousand = n / 1000 % 100;
    let hundred = n / 100 % 10;
    let rest = n % 100;
    let mut parts: Vec<String> = Vec::new();
    if crore > 0 {
        parts.push(format!("{} करोड़", hi_number(crore)));
    }
    if lakh > 0 {
        parts.push(format!("{} लाख", HI_0_99[lakh as usize]));
    }
    if thousand > 0 {
        parts.push(format!("{} हज़ार", HI_0_99[thousand as usize]));
    }
    if hundred > 0 {
        parts.push(format!("{} सौ", HI_0_99[hundred as usize]));
    }
    if rest > 0 {
        parts.push(HI_0_99[rest as usize].to_string());
    }
    parts.join(" ")
}

fn number_words(n: u64, lang: &str) -> Option<String> {
    match lang {
        "te" => Some(te_number(n)),
        "hi" => Some(hi_number(n)),
        _ => None,
    }
}

fn digit_words(digits: &str, lang: &str) -> String {
    digits.chars().filter_map(|d| d.to_digit(10)).filter_map(|d| number_words(d as u64, lang)).collect::<Vec<_>>().join(" ")
}

fn words_for(lang: &str) -> Option<&'static Words> {
    match lang {
        "te" => Some(&TE_WORDS),
        "hi" => Some(&HI_WORDS),
        _ => None,
    }
}

/// A parsed number token.
struct Num {
    negative: bool,
    int: u64,
    /// fractional digits as written
    frac: String,
}

impl Num {
    fn say(&self, lang: &str, w: &Words) -> String {
        let mut s = String::new();
        if self.negative {
            s.push_str(w.minus);
            s.push(' ');
        }
        s.push_str(&number_words(self.int, lang).unwrap_or_default());
        if !self.frac.is_empty() {
            s.push_str(&format!(" {} {}", w.point, digit_words(&self.frac, lang)));
        }
        s
    }
}

/// Parse digits with optional thousands separators (any grouping) and decimals at
/// `b[i..]`; returns the number and the end index.
fn parse_num(b: &[char], mut i: usize) -> Option<(Num, usize)> {
    let start = i;
    let mut int = String::new();
    while i < b.len() && (b[i].is_ascii_digit() || (b[i] == ',' && i + 1 < b.len() && b[i + 1].is_ascii_digit() && !int.is_empty())) {
        if b[i] != ',' {
            int.push(b[i]);
        }
        i += 1;
    }
    if int.is_empty() || int.len() > 15 {
        return None;
    }
    let mut frac = String::new();
    if i + 1 < b.len() && b[i] == '.' && b[i + 1].is_ascii_digit() {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            frac.push(b[i]);
            i += 1;
        }
    }
    let _ = start;
    Some((Num { negative: false, int: int.parse().ok()?, frac }, i))
}

fn starts_with_at(b: &[char], i: usize, pat: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    b.len() >= i + p.len() && b[i..i + p.len()].iter().zip(&p).all(|(a, c)| a.eq_ignore_ascii_case(c))
}

/// Write numbers, currency, percentages, times and temperatures out as words in `lang`
/// (`te`, `hi`). For other languages the text is returned with digit-group commas
/// removed (so "12,34,567" is not read as three numbers) and otherwise unchanged.
pub fn verbalize(text: &str, lang: &str) -> String {
    let text = crate::text::normalize(text);
    let b: Vec<char> = text.chars().collect();
    let Some(w) = words_for(lang) else {
        // keep digits, drop separators inside numbers
        let mut out = String::with_capacity(text.len());
        for (i, &c) in b.iter().enumerate() {
            if c == ',' && i > 0 && b[i - 1].is_ascii_digit() && b.get(i + 1).is_some_and(|n| n.is_ascii_digit()) {
                continue;
            }
            out.push(c);
        }
        return out;
    };
    let mut out = String::with_capacity(text.len() * 3);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let prev_is_word = i > 0 && (b[i - 1].is_alphanumeric());
        // currency prefixes
        let (cur, skip) = if c == '₹' {
            (Some("inr"), 1)
        } else if c == '$' {
            (Some("usd"), 1)
        } else if c == '€' {
            (Some("eur"), 1)
        } else if !prev_is_word && starts_with_at(&b, i, "rs.") {
            (Some("inr"), 3)
        } else if !prev_is_word && starts_with_at(&b, i, "rs") && b.get(i + 2).is_some_and(|x| x.is_ascii_digit() || *x == ' ') {
            (Some("inr"), 2)
        } else {
            (None, 0)
        };
        if let Some(cur) = cur {
            let mut j = i + skip;
            while j < b.len() && b[j] == ' ' {
                j += 1;
            }
            if let Some((n, end)) = parse_num(&b, j) {
                out.push_str(&say_money(&n, cur, lang, w));
                i = end;
                continue;
            }
        }
        // digits inside identifiers (Q4_0, A17, v2) are kept as written
        if c.is_ascii_digit() && i > 0 && (b[i - 1].is_ascii_alphabetic() || b[i - 1] == '_') {
            while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                out.push(b[i]);
                i += 1;
            }
            continue;
        }
        if c.is_ascii_digit() {
            // time HH:MM
            if let Some(colon) = (i + 1..(i + 3).min(b.len())).find(|&j| b[j] == ':') {
                let hh: String = b[i..colon].iter().collect();
                if hh.chars().all(|d| d.is_ascii_digit()) && b.len() > colon + 2 {
                    let mm: String = b[colon + 1..colon + 3].iter().collect();
                    let after_ok = b.get(colon + 3).is_none_or(|x| !x.is_ascii_digit());
                    if mm.chars().all(|d| d.is_ascii_digit()) && after_ok {
                        let (h, m): (u64, u64) = (hh.parse().unwrap_or(0), mm.parse().unwrap_or(0));
                        if h < 24 && m < 60 {
                            let hw = number_words(h, lang).unwrap_or_default();
                            if m == 0 {
                                out.push_str(&format!("{hw} {}", w.hours));
                            } else {
                                out.push_str(&format!("{hw} {} {} {}", w.hours_join, number_words(m, lang).unwrap_or_default(), w.minutes));
                            }
                            i = colon + 3;
                            continue;
                        }
                    }
                }
            }
            if let Some((mut n, end)) = parse_num(&b, i) {
                if i > 0 && b[i - 1] == '-' && (i < 2 || !b[i - 2].is_alphanumeric()) {
                    // "-84.2": the minus was already copied; replace it
                    if out.ends_with('-') {
                        out.pop();
                    }
                    n.negative = true;
                }
                let mut j = end;
                let mut spaced = j;
                while spaced < b.len() && b[spaced] == ' ' {
                    spaced += 1;
                }
                let said = n.say(lang, w);
                if j < b.len() && b[j] == '%' {
                    out.push_str(&format!("{said} {}", w.percent));
                    j += 1;
                } else if starts_with_at(&b, j, "°c") || starts_with_at(&b, j, "° c") {
                    out.push_str(&format!("{said} {}", w.degrees_c));
                    j += if b[j + 1] == ' ' { 3 } else { 2 };
                } else if let Some((cur, len)) = ["inr", "rupees", "usd", "eur"]
                    .iter()
                    .find(|k| starts_with_at(&b, spaced, k) && b.get(spaced + k.len()).is_none_or(|x| !x.is_alphanumeric()))
                    .map(|k| (if *k == "rupees" { "inr" } else { k }, k.len()))
                {
                    out.push_str(&say_money(&n, cur, lang, w));
                    j = spaced + len;
                } else {
                    out.push_str(&said);
                }
                i = j;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn say_money(n: &Num, cur: &str, lang: &str, w: &Words) -> String {
    let minus = if n.negative { format!("{} ", w.minus) } else { String::new() };
    match cur {
        "inr" => {
            let unit = if n.int == 1 && n.frac.is_empty() { w.rupee_one } else { w.rupees };
            // Telugu uses the adjectival "ఒక" for one rupee
            let amount = if lang == "te" && n.int == 1 { "ఒక".to_string() } else { number_words(n.int, lang).unwrap_or_default() };
            let mut s = format!("{minus}{amount} {unit}");
            // paise: the first two decimals
            let paise: String = n.frac.chars().chain(std::iter::repeat('0')).take(2).collect();
            let p: u64 = paise.parse().unwrap_or(0);
            if !n.frac.is_empty() && p > 0 {
                s.push_str(&format!(" {} {}", number_words(p, lang).unwrap_or_default(), w.paise));
            }
            s
        }
        other => {
            let unit = if other == "usd" { w.dollars } else { w.euros };
            format!("{} {unit}", Num { negative: n.negative, int: n.int, frac: n.frac.clone() }.say(lang, w))
        }
    }
}

/// Split text into speakable chunks: at sentence ends (। ॥ . ? ! and line breaks),
/// with long sentences broken at commas or spaces so no chunk exceeds `max_chars`.
pub fn sentences(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(20);
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let next_is_space = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        let end = matches!(c, '।' | '॥' | '?' | '!' | '\n')
            || (c == '.' && next_is_space && !cur.trim_end_matches('.').ends_with(|x: char| x.is_ascii_digit()));
        if end {
            push_chunk(&mut out, &cur, max_chars);
            cur.clear();
        }
    }
    push_chunk(&mut out, &cur, max_chars);
    out
}

/// Split streamed text into (complete sentences, unfinished remainder). The remainder is
/// returned untrimmed so the next streamed piece joins it correctly.
pub fn split_complete(text: &str) -> (&str, &str) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut cut = 0;
    for (k, &(i, c)) in chars.iter().enumerate() {
        let next_is_space = chars.get(k + 1).is_some_and(|n| n.1.is_whitespace());
        let digit_before = text[..i].ends_with(|x: char| x.is_ascii_digit());
        if matches!(c, '।' | '॥' | '?' | '!' | '\n') || (c == '.' && next_is_space && !digit_before) {
            cut = i + c.len_utf8();
        }
    }
    text.split_at(cut)
}

fn push_chunk(out: &mut Vec<String>, s: &str, max: usize) {
    let mut s = s.trim().to_string();
    while s.chars().count() > max {
        let cut: String = s.chars().take(max).collect();
        let at = cut.rfind([',', '،', ';']).or_else(|| cut.rfind(' ')).filter(|&k| k > 0).map(|k| k + 1).unwrap_or(cut.len());
        out.push(s[..at].trim().to_string());
        s = s[at..].trim().to_string();
    }
    if !s.is_empty() {
        out.push(s);
    }
}

#[cfg(test)]
// literals are grouped the Indian way (lakh, crore) on purpose
#[allow(clippy::inconsistent_digit_grouping)]
mod tests {
    use super::*;

    #[test]
    fn telugu_numbers() {
        assert_eq!(te_number(0), "సున్నా");
        assert_eq!(te_number(7), "ఏడు");
        assert_eq!(te_number(21), "ఇరవై ఒకటి");
        assert_eq!(te_number(100), "వంద");
        assert_eq!(te_number(105), "నూట ఐదు");
        assert_eq!(te_number(300), "మూడు వందలు");
        assert_eq!(te_number(431), "నాలుగు వందల ముప్పై ఒకటి");
        assert_eq!(te_number(1000), "వెయ్యి");
        assert_eq!(te_number(2000), "రెండు వేలు");
        assert_eq!(te_number(2431), "రెండు వేల నాలుగు వందల ముప్పై ఒకటి");
        assert_eq!(te_number(100_000), "ఒక లక్ష");
        assert_eq!(te_number(250_000), "రెండు లక్షల యాభై వేలు");
        assert_eq!(te_number(10_000_000), "ఒక కోటి");
        assert_eq!(te_number(30_000_000), "మూడు కోట్లు");
        assert_eq!(te_number(12_34_56_789), "పన్నెండు కోట్ల ముప్పై నాలుగు లక్షల యాభై ఆరు వేల ఏడు వందల ఎనభై తొమ్మిది");
    }

    #[test]
    fn hindi_numbers() {
        assert_eq!(hi_number(5), "पाँच");
        assert_eq!(hi_number(99), "निन्यानवे");
        assert_eq!(hi_number(2431), "दो हज़ार चार सौ इकतीस");
        assert_eq!(hi_number(1_50_000), "एक लाख पचास हज़ार");
        assert_eq!(hi_number(2_00_00_000), "दो करोड़");
    }

    #[test]
    fn verbalizes_money_percent_time_and_temperature() {
        assert_eq!(verbalize("బ్యాలెన్స్ ₹2,431.75", "te"), "బ్యాలెన్స్ రెండు వేల నాలుగు వందల ముప్పై ఒకటి రూపాయలు డెబ్బై ఐదు పైసలు");
        assert_eq!(verbalize("Rs. 1,00,000", "te"), "ఒక లక్ష రూపాయలు");
        assert_eq!(verbalize("₹1", "te"), "ఒక రూపాయి");
        assert_eq!(verbalize("వర్షం 60%", "te"), "వర్షం అరవై శాతం");
        assert_eq!(verbalize("18:30 కి", "te"), "పద్దెనిమిది గంటల ముప్పై నిమిషాలు కి");
        assert_eq!(verbalize("09:00", "te"), "తొమ్మిది గంటలు");
        assert_eq!(verbalize("18.4°C", "te"), "పద్దెనిమిది పాయింట్ నాలుగు డిగ్రీల సెల్సియస్");
        assert_eq!(verbalize("pending -84.2 EUR", "te"), "pending మైనస్ ఎనభై నాలుగు పాయింట్ రెండు యూరోలు");
        assert_eq!(verbalize("₹2,431.75", "hi"), "दो हज़ार चार सौ इकतीस रुपये पचहत्तर पैसे");
        assert_eq!(verbalize("10:15", "hi"), "दस बजकर पंद्रह मिनट");
        // other languages: digits stay, grouping commas go
        assert_eq!(verbalize("₹12,34,567 at 10:15", "en"), "₹1234567 at 10:15");
        // words with digits are left alone
        assert_eq!(verbalize("model Q4_0 on A17", "te"), "model Q4_0 on A17");
    }

    #[test]
    fn splits_into_speakable_sentences() {
        let s = sentences("ఈరోజు వర్షం పడుతుంది. గొడుగు తీసుకెళ్ళండి! ధర 2.5 లక్షలు। తర్వాత", 200);
        assert_eq!(s, vec!["ఈరోజు వర్షం పడుతుంది.", "గొడుగు తీసుకెళ్ళండి!", "ధర 2.5 లక్షలు।", "తర్వాత"]);
        let long = "పదం ".repeat(100);
        assert!(sentences(&long, 50).iter().all(|c| c.chars().count() <= 50));
    }
}
