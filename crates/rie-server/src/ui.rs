//! On-the-fly UI for tool results, in one fixed style.
//!
//! Every answer page has the same shape (layout v1):
//!
//! * a header with the question, the time and summary chips;
//! * one **section per category**, always in the order of [`CATEGORIES`], showing only
//!   categories that have results;
//! * one **card per tool**, sorted by tool name, with a status badge and duration;
//! * card bodies built from a small set of item types ([`Item`]): metric, fields,
//!   table, list, text and error. Any JSON or text a tool returns is normalized into
//!   these, and numbers, booleans and keys are formatted the same way everywhere.
//!
//! The model never writes markup: it only fills tool arguments (grammar-constrained). The
//! page is rendered deterministically, so the same data always looks the same. The same
//! page is available as HTML (phone WebView or browser), as a JSON spec (native
//! Android/iOS renderers) and as Markdown (chat bubbles, notifications).

use crate::headless::Status;
use serde_json::{json, Value};

pub const LAYOUT_VERSION: u32 = 1;

pub struct Category {
    pub key: &'static str,
    pub title: &'static str,
    pub icon: &'static str,
    keywords: &'static str,
}

/// The fixed categories, in display order (the last one catches everything else).
pub const CATEGORIES: [Category; 10] = [
    Category { key: "weather", title: "Weather", icon: "☀", keywords: "weather forecast temperature rain wind humidity climate sunny" },
    Category {
        key: "calendar",
        title: "Calendar & Tasks",
        icon: "▦",
        keywords: "calendar event events meeting schedule agenda appointment task tasks todo reminder today date",
    },
    Category {
        key: "communication",
        title: "Messages & Contacts",
        icon: "✉",
        keywords: "email mail message messages sms chat contact contacts call inbox notification send",
    },
    Category {
        key: "finance",
        title: "Money & Finance",
        icon: "₿",
        keywords: "finance bank balance account payment money stock price loan emi budget expense currency invoice",
    },
    Category {
        key: "travel",
        title: "Places & Travel",
        icon: "➤",
        keywords: "map maps location place route directions travel flight train traffic distance nearby address gps",
    },
    Category {
        key: "files",
        title: "Files & Documents",
        icon: "▤",
        keywords: "file files document documents folder read write search pdf notes photo photos storage",
    },
    Category {
        key: "web",
        title: "Web & News",
        icon: "◎",
        keywords: "web search internet news headlines article browse url page wikipedia",
    },
    Category {
        key: "device",
        title: "Device & System",
        icon: "⚙",
        keywords: "device battery system phone wifi bluetooth storage memory cpu settings app apps network",
    },
    Category {
        key: "health",
        title: "Health & Fitness",
        icon: "♥",
        keywords: "health fitness steps heart rate sleep workout calories exercise weight",
    },
    Category { key: "other", title: "Other", icon: "•", keywords: "" },
];

/// Category index for a tool: the manifest's declared category if it names one,
/// otherwise the best keyword match of the tool's name and description.
pub fn categorize(name: &str, description: &str, declared: Option<&str>) -> usize {
    if let Some(d) = declared {
        let d = d.trim().to_lowercase();
        if let Some(i) = CATEGORIES.iter().position(|c| c.key == d || c.title.to_lowercase() == d || c.title.to_lowercase().starts_with(&d))
        {
            return i;
        }
    }
    // exact keyword matches (singular/plural folded): deterministic and predictable
    let stem = |w: &str| -> String {
        let w = w.to_lowercase();
        match w.strip_suffix('s') {
            Some(b) if b.len() >= 3 && !b.ends_with('s') => b.to_string(),
            _ => w,
        }
    };
    let words: Vec<String> =
        format!("{name} {description}").split(|c: char| !rie_core::text::is_word_char(c)).filter(|w| !w.is_empty()).map(stem).collect();
    let mut best = (0usize, CATEGORIES.len() - 1);
    for (i, c) in CATEGORIES.iter().enumerate().take(CATEGORIES.len() - 1) {
        // the tool's name counts double: it is the most specific signal
        let name_words: Vec<String> = name.split(|c: char| !rie_core::text::is_word_char(c)).map(stem).collect();
        let score: usize = c
            .keywords
            .split_whitespace()
            .map(stem)
            .map(|k| words.iter().filter(|w| **w == k).count() + 2 * name_words.iter().filter(|w| **w == k).count())
            .sum();
        if score > best.0 {
            best = (score, i);
        }
    }
    best.1
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Metric { label: String, value: String },
    Fields { title: Option<String>, rows: Vec<(String, String)> },
    Table { title: Option<String>, columns: Vec<String>, rows: Vec<Vec<String>>, more: usize },
    List { title: Option<String>, items: Vec<String>, more: usize },
    Text(String),
    Error(String),
}

/// Display strings of one language. Only the labels change between languages; the
/// layout, order and formatting rules stay the same.
pub struct Labels {
    pub answer: &'static str,
    pub tools: &'static str,
    pub ok: &'static str,
    pub failed: &'static str,
    pub categories: &'static str,
    pub empty: &'static str,
    pub show_all: &'static str,
    pub truncated: &'static str,
    pub more: &'static str,
    pub footer: &'static str,
    /// badge text for ok / error / timeout
    pub status: [&'static str; 3],
    /// category titles, in [`CATEGORIES`] order
    pub cats: [&'static str; 10],
    pub yes: &'static str,
    pub no: &'static str,
    pub no_data: &'static str,
    pub empty_text: &'static str,
    pub no_results: &'static str,
    pub none: &'static str,
    /// spoken summary: "N {items}" for tables, "N {failed_tools}" for failures
    pub items: &'static str,
    pub failed_tools: &'static str,
    /// sentence end for spoken text
    pub stop: &'static str,
}

pub const EN: Labels = Labels {
    answer: "Answer",
    tools: "tools",
    ok: "ok",
    failed: "failed",
    categories: "categories",
    empty: "No relevant tools were found for this request.",
    show_all: "Show all",
    truncated: "Output truncated.",
    more: "more",
    footer: "Generated on device by RIE",
    status: ["ok", "error", "timeout"],
    cats: [
        "Weather",
        "Calendar & Tasks",
        "Messages & Contacts",
        "Money & Finance",
        "Places & Travel",
        "Files & Documents",
        "Web & News",
        "Device & System",
        "Health & Fitness",
        "Other",
    ],
    yes: "Yes",
    no: "No",
    no_data: "(no data)",
    empty_text: "(empty)",
    no_results: "(no results)",
    none: "(none)",
    items: "items",
    failed_tools: "tools failed",
    stop: ".",
};

/// Telugu labels (review by native speakers welcome).
pub const TE: Labels = Labels {
    answer: "సమాధానం",
    tools: "సాధనాలు",
    ok: "విజయవంతం",
    failed: "విఫలం",
    categories: "విభాగాలు",
    empty: "ఈ అభ్యర్థనకు సంబంధించిన సాధనాలు ఏవీ కనబడలేదు.",
    show_all: "అన్నీ చూపించు",
    truncated: "ఫలితం కుదించబడింది.",
    more: "మరిన్ని",
    footer: "RIE ద్వారా పరికరంలోనే రూపొందించబడింది",
    status: ["సరే", "లోపం", "సమయం మించింది"],
    cats: [
        "వాతావరణం",
        "క్యాలెండర్ & పనులు",
        "సందేశాలు & పరిచయాలు",
        "డబ్బు & ఆర్థికం",
        "ప్రదేశాలు & ప్రయాణం",
        "ఫైళ్లు & పత్రాలు",
        "వెబ్ & వార్తలు",
        "పరికరం & సిస్టమ్",
        "ఆరోగ్యం & ఫిట్‌నెస్",
        "ఇతరాలు",
    ],
    yes: "అవును",
    no: "కాదు",
    no_data: "(సమాచారం లేదు)",
    empty_text: "(ఖాళీ)",
    no_results: "(ఫలితాలు లేవు)",
    none: "(ఏదీ లేదు)",
    items: "అంశాలు",
    failed_tools: "సాధనాలు విఫలమయ్యాయి",
    stop: ".",
};

/// Hindi labels (review by native speakers welcome).
pub const HI: Labels = Labels {
    answer: "उत्तर",
    tools: "टूल",
    ok: "सफल",
    failed: "विफल",
    categories: "श्रेणियाँ",
    empty: "इस अनुरोध के लिए कोई संबंधित टूल नहीं मिला।",
    show_all: "सब दिखाएँ",
    truncated: "आउटपुट छोटा किया गया।",
    more: "और",
    footer: "RIE द्वारा डिवाइस पर बनाया गया",
    status: ["सफल", "त्रुटि", "समय समाप्त"],
    cats: [
        "मौसम",
        "कैलेंडर और काम",
        "संदेश और संपर्क",
        "पैसा और वित्त",
        "स्थान और यात्रा",
        "फ़ाइलें और दस्तावेज़",
        "वेब और समाचार",
        "डिवाइस और सिस्टम",
        "स्वास्थ्य और फ़िटनेस",
        "अन्य",
    ],
    yes: "हाँ",
    no: "नहीं",
    no_data: "(कोई डेटा नहीं)",
    empty_text: "(खाली)",
    no_results: "(कोई परिणाम नहीं)",
    none: "(कोई नहीं)",
    items: "आइटम",
    failed_tools: "टूल विफल रहे",
    stop: "।",
};

/// Language of a page: its labels and how numbers are grouped.
#[derive(Clone, Copy)]
pub struct Locale {
    pub lang: &'static str,
    pub labels: &'static Labels,
    /// Indian digit grouping (12,34,567: lakh and crore)
    pub indian: bool,
}

pub const LOCALE_EN: Locale = Locale { lang: "en", labels: &EN, indian: false };

impl Locale {
    /// `te` and `hi` have their own labels; other Indian languages use English labels
    /// for now with Indian digit grouping; `en-IN` is English with Indian grouping.
    pub fn for_lang(lang: &str) -> Locale {
        let l = lang.to_ascii_lowercase();
        match l.as_str() {
            "te" | "te-in" => Locale { lang: "te", labels: &TE, indian: true },
            "hi" | "hi-in" => Locale { lang: "hi", labels: &HI, indian: true },
            "en-in" => Locale { lang: "en", labels: &EN, indian: true },
            _ => match ["bn", "ta", "kn", "ml", "mr", "gu", "pa", "or"].iter().find(|c| l.starts_with(*c)) {
                Some(c) => Locale { lang: c, labels: &EN, indian: true },
                None => LOCALE_EN,
            },
        }
    }

    /// Locale for a question: its dominant script decides.
    pub fn detect(text: &str) -> Locale {
        Locale::for_lang(rie_core::text::detect_script(text).lang())
    }
}

#[derive(Debug, Clone)]
pub struct Card {
    pub tool: String,
    pub status: Status,
    pub ms: u64,
    pub args: Value,
    pub items: Vec<Item>,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub category: usize,
    pub cards: Vec<Card>,
}

#[derive(Debug, Clone)]
pub struct Page {
    pub id: String,
    pub question: String,
    /// unix seconds
    pub created: u64,
    pub sections: Vec<Section>,
    /// BCP-47 language of the labels
    pub lang: &'static str,
}

const MAX_ROWS: usize = 20;
const MAX_COLS: usize = 6;
const MAX_CELL: usize = 200;
const MAX_TEXT: usize = 2000;
/// Characters of a long text shown before "Show all".
const TEXT_PREVIEW: usize = 280;

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// `temp_c` → "Temp c", `annualRate` → "Annual rate".
pub fn humanize(key: &str) -> String {
    let mut words = String::new();
    let mut prev_lower = false;
    for ch in key.chars() {
        if ch == '_' || ch == '-' || ch == '.' {
            words.push(' ');
            prev_lower = false;
            continue;
        }
        if ch.is_uppercase() && prev_lower {
            words.push(' ');
        }
        prev_lower = ch.is_lowercase() || ch.is_ascii_digit();
        words.extend(ch.to_lowercase());
    }
    let w = words.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut c = w.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// 1234567 → "12,34,567" (lakh/crore grouping: last three digits, then pairs).
fn group_indian(digits: &str) -> String {
    if digits.len() <= 3 {
        return digits.to_string();
    }
    let (head, tail) = digits.split_at(digits.len() - 3);
    let mut out = String::new();
    for (i, ch) in head.chars().enumerate() {
        if i > 0 && (head.len() - i) % 2 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    format!("{out},{tail}")
}

fn group_thousands(digits: &str) -> String {
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// One formatting rule for every scalar on every page.
pub fn fmt_value(v: &Value) -> String {
    fmt_value_in(v, &LOCALE_EN)
}

/// [`fmt_value`] for a locale (yes/no words, digit grouping).
pub fn fmt_value_in(v: &Value, loc: &Locale) -> String {
    let group_thousands = |d: &str| if loc.indian { group_indian(d) } else { group_thousands(d) };
    match v {
        Value::Null => "—".into(),
        Value::Bool(b) => if *b { loc.labels.yes } else { loc.labels.no }.into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let s = group_thousands(&i.unsigned_abs().to_string());
                if i < 0 {
                    format!("-{s}")
                } else {
                    s
                }
            } else if let Some(u) = n.as_u64() {
                group_thousands(&u.to_string())
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                if !f.is_finite() {
                    return "—".into();
                }
                let s = format!("{:.2}", f.abs());
                let (int, frac) = s.split_once('.').unwrap_or((&s, ""));
                let frac = frac.trim_end_matches('0');
                let sign = if f < 0.0 && s != "0.00" { "-" } else { "" };
                if frac.is_empty() {
                    format!("{sign}{}", group_thousands(int))
                } else {
                    format!("{sign}{}.{frac}", group_thousands(int))
                }
            }
        }
        Value::String(s) => clip(s.trim(), MAX_CELL),
        other => clip(&other.to_string(), MAX_CELL),
    }
}

fn is_scalar(v: &Value) -> bool {
    !matches!(v, Value::Array(_) | Value::Object(_))
}

fn table(title: Option<String>, arr: &[Value], loc: &Locale) -> Item {
    let mut columns: Vec<String> = Vec::new();
    for o in arr.iter().filter_map(Value::as_object) {
        for k in o.keys() {
            if !columns.contains(k) && columns.len() < MAX_COLS {
                columns.push(k.clone());
            }
        }
    }
    let rows =
        arr.iter().take(MAX_ROWS).map(|o| columns.iter().map(|c| fmt_value_in(o.get(c).unwrap_or(&Value::Null), loc)).collect()).collect();
    Item::Table { title, columns: columns.iter().map(|c| humanize(c)).collect(), rows, more: arr.len().saturating_sub(MAX_ROWS) }
}

fn array_item(title: Option<String>, arr: &[Value], loc: &Locale) -> Item {
    if !arr.is_empty() && arr.iter().all(Value::is_object) {
        table(title, arr, loc)
    } else {
        Item::List {
            title,
            items: arr.iter().take(MAX_ROWS).map(|v| fmt_value_in(v, loc)).collect(),
            more: arr.len().saturating_sub(MAX_ROWS),
        }
    }
}

/// Normalize any tool result into display items.
pub fn normalize(tool: &str, v: &Value) -> Vec<Item> {
    normalize_in(tool, v, &LOCALE_EN)
}

/// [`normalize`] for a locale.
pub fn normalize_in(tool: &str, v: &Value, loc: &Locale) -> Vec<Item> {
    let fmt_value = |v: &Value| fmt_value_in(v, loc);
    let l = loc.labels;
    match v {
        Value::Null => vec![Item::Text(l.no_data.into())],
        Value::Bool(_) | Value::Number(_) => {
            vec![Item::Metric { label: humanize(tool.rsplit('.').next().unwrap_or(tool)), value: fmt_value(v) }]
        }
        Value::String(s) if s.trim().is_empty() => vec![Item::Text(l.empty_text.into())],
        Value::String(s) => vec![Item::Text(clip(s.trim(), MAX_TEXT))],
        Value::Array(a) if a.is_empty() => vec![Item::Text(l.no_results.into())],
        Value::Array(a) => vec![array_item(None, a, loc)],
        Value::Object(o) => {
            if let Some(e) = o.get("error").filter(|e| !e.is_null() && o.len() <= 2) {
                return vec![Item::Error(fmt_value(e))];
            }
            let mut fields = Vec::new();
            let mut rest = Vec::new();
            for (k, val) in o {
                match val {
                    s if is_scalar(s) => fields.push((humanize(k), fmt_value(s))),
                    Value::Array(a) if a.is_empty() => fields.push((humanize(k), l.none.into())),
                    Value::Array(a) => rest.push(array_item(Some(humanize(k)), a, loc)),
                    Value::Object(inner) => {
                        let rows: Vec<(String, String)> =
                            inner.iter().take(MAX_ROWS).map(|(ik, iv)| (humanize(ik), fmt_value(iv))).collect();
                        rest.push(Item::Fields { title: Some(humanize(k)), rows });
                    }
                    _ => {}
                }
            }
            let mut items = Vec::new();
            if !fields.is_empty() {
                items.push(Item::Fields { title: None, rows: fields });
            }
            items.extend(rest);
            if items.is_empty() {
                items.push(Item::Text(l.no_data.into()));
            }
            items
        }
    }
}

impl Card {
    pub fn new(tool: &str, status: Status, ms: u64, args: Value, value: &Value, error: Option<&str>, truncated: bool) -> Self {
        Self::new_in(tool, status, ms, args, value, error, truncated, &LOCALE_EN)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_in(
        tool: &str,
        status: Status,
        ms: u64,
        args: Value,
        value: &Value,
        error: Option<&str>,
        truncated: bool,
        loc: &Locale,
    ) -> Self {
        let mut items = Vec::new();
        if let Some(e) = error {
            items.push(Item::Error(clip(e, 500)));
        }
        let has_output = !matches!(value, Value::Null) && !matches!(value, Value::String(s) if s.trim().is_empty());
        if status == Status::Ok || has_output {
            items.extend(normalize_in(tool, value, loc));
        }
        Card { tool: tool.to_string(), status, ms, args, items, truncated }
    }
}

/// Build a page: cards grouped by category (fixed order), cards sorted by tool name.
pub fn build(id: &str, question: &str, created: u64, cards: Vec<(usize, Card)>) -> Page {
    build_in(id, question, created, cards, &LOCALE_EN)
}

/// [`build`] with the labels of a locale (cards should be made with the same locale).
pub fn build_in(id: &str, question: &str, created: u64, cards: Vec<(usize, Card)>, loc: &Locale) -> Page {
    let mut sections: Vec<Section> = Vec::new();
    for (ci, _) in CATEGORIES.iter().enumerate() {
        let mut cs: Vec<Card> = cards.iter().filter(|(c, _)| *c == ci).map(|(_, card)| card.clone()).collect();
        if cs.is_empty() {
            continue;
        }
        cs.sort_by(|a, b| a.tool.cmp(&b.tool));
        sections.push(Section { category: ci, cards: cs });
    }
    Page { id: id.to_string(), question: question.to_string(), created, sections, lang: loc.lang }
}

/// `2026-11-09 14:05 UTC`
pub fn fmt_time(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, rem % 3600 / 60)
}

impl Page {
    pub fn labels(&self) -> &'static Labels {
        Locale::for_lang(self.lang).labels
    }

    fn status_label(&self, st: Status) -> &'static str {
        self.labels().status[match st {
            Status::Ok => 0,
            Status::Error => 1,
            Status::Timeout => 2,
        }]
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        let cards: Vec<&Card> = self.sections.iter().flat_map(|s| &s.cards).collect();
        let ok = cards.iter().filter(|c| c.status == Status::Ok).count();
        (cards.len(), ok, cards.len() - ok)
    }

    /// JSON spec (for native renderers); same structure as the HTML.
    pub fn to_json(&self) -> Value {
        let (n, ok, failed) = self.counts();
        let item = |i: &Item| match i {
            Item::Metric { label, value } => json!({"type": "metric", "label": label, "value": value}),
            Item::Fields { title, rows } => {
                json!({"type": "fields", "title": title, "rows": rows.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>()})
            }
            Item::Table { title, columns, rows, more } => {
                json!({"type": "table", "title": title, "columns": columns, "rows": rows, "more": more})
            }
            Item::List { title, items, more } => json!({"type": "list", "title": title, "items": items, "more": more}),
            Item::Text(t) => json!({"type": "text", "text": t}),
            Item::Error(e) => json!({"type": "error", "text": e}),
        };
        json!({
            "layout": LAYOUT_VERSION,
            "lang": self.lang,
            "id": self.id,
            "question": self.question,
            "created": self.created,
            "created_text": fmt_time(self.created),
            "summary": {"tools": n, "ok": ok, "failed": failed, "categories": self.sections.len()},
            "sections": self.sections.iter().map(|s| {
                let c = &CATEGORIES[s.category];
                json!({"category": c.key, "title": self.labels().cats[s.category], "icon": c.icon,
                    "cards": s.cards.iter().map(|card| json!({"tool": card.tool, "status": card.status.name(), "status_label": self.status_label(card.status), "ms": card.ms,
                        "args": card.args, "truncated": card.truncated, "items": card.items.iter().map(item).collect::<Vec<_>>()})).collect::<Vec<_>>()})
            }).collect::<Vec<_>>(),
        })
    }

    /// Short text to read aloud, in the page's language (not yet verbalized: pass it
    /// through `rie_core::speech::verbalize` / `tts::say`). Same structure every time:
    /// per category its title, then per tool its first facts.
    pub fn speech_text(&self) -> String {
        let l = self.labels();
        let (_, _, failed) = self.counts();
        let mut out: Vec<String> = Vec::new();
        if self.sections.is_empty() {
            return l.empty.to_string();
        }
        for sec in &self.sections {
            let mut parts: Vec<String> = Vec::new();
            for card in &sec.cards {
                if card.status != Status::Ok {
                    continue;
                }
                for it in card.items.iter().take(2) {
                    match it {
                        Item::Metric { label, value } => parts.push(format!("{label} {value}")),
                        Item::Fields { rows, .. } => parts.extend(rows.iter().take(4).map(|(k, v)| format!("{k} {v}"))),
                        Item::Table { rows, more, .. } => {
                            let mut p = format!("{} {}", rows.len() + more, l.items);
                            if let Some(r) = rows.first() {
                                p.push_str(&format!(": {}", r.join(", ")));
                            }
                            parts.push(p);
                        }
                        Item::List { items, .. } => parts.push(items.iter().take(3).cloned().collect::<Vec<_>>().join(", ")),
                        Item::Text(t) => parts.push(clip(t, 120)),
                        Item::Error(_) => {}
                    }
                }
            }
            if !parts.is_empty() {
                out.push(format!("{}{} {}{}", l.cats[sec.category], l.stop, parts.join(", "), l.stop));
            }
        }
        if failed > 0 {
            out.push(format!("{failed} {}{}", l.failed_tools, l.stop));
        }
        out.join(" ")
    }

    pub fn to_markdown(&self) -> String {
        let (n, ok, failed) = self.counts();
        let l = self.labels();
        let mut s = format!(
            "## {}\n\n_{} · {n} {} · {ok} {} · {failed} {}_\n",
            self.question.trim(),
            fmt_time(self.created),
            l.tools,
            l.ok,
            l.failed
        );
        if self.sections.is_empty() {
            s.push_str(&format!("\n{}\n", l.empty));
        }
        for sec in &self.sections {
            let c = &CATEGORIES[sec.category];
            s.push_str(&format!("\n### {} {}\n", c.icon, l.cats[sec.category]));
            for card in &sec.cards {
                s.push_str(&format!(
                    "\n**`{}`** — {} · {} ms{}\n\n",
                    card.tool,
                    self.status_label(card.status),
                    card.ms,
                    if card.truncated { format!(" · {}", l.truncated) } else { String::new() }
                ));
                for it in &card.items {
                    match it {
                        Item::Metric { label, value } => s.push_str(&format!("- **{label}:** {value}\n")),
                        Item::Fields { title, rows } => {
                            if let Some(t) = title {
                                s.push_str(&format!("\n*{t}*\n\n"));
                            }
                            for (k, v) in rows {
                                s.push_str(&format!("- **{k}:** {v}\n"));
                            }
                        }
                        Item::Table { title, columns, rows, more } => {
                            if let Some(t) = title {
                                s.push_str(&format!("\n*{t}*\n\n"));
                            }
                            let esc = |x: &str| x.replace('|', "\\|").replace('\n', " ");
                            s.push_str(&format!(
                                "\n| {} |\n|{}\n",
                                columns.iter().map(|c| esc(c)).collect::<Vec<_>>().join(" | "),
                                "---|".repeat(columns.len())
                            ));
                            for r in rows {
                                s.push_str(&format!("| {} |\n", r.iter().map(|c| esc(c)).collect::<Vec<_>>().join(" | ")));
                            }
                            if *more > 0 {
                                s.push_str(&format!("\n_+{more} {}_\n", l.more));
                            }
                            s.push('\n');
                        }
                        Item::List { title, items, more } => {
                            if let Some(t) = title {
                                s.push_str(&format!("\n*{t}*\n\n"));
                            }
                            for x in items {
                                s.push_str(&format!("- {x}\n"));
                            }
                            if *more > 0 {
                                s.push_str(&format!("- _+{more} {}_\n", l.more));
                            }
                        }
                        Item::Text(t) => s.push_str(&format!("{t}\n")),
                        Item::Error(e) => s.push_str(&format!("> ⚠ {e}\n")),
                    }
                }
            }
        }
        s
    }

    pub fn to_html(&self) -> String {
        let (n, ok, failed) = self.counts();
        let l = self.labels();
        let mut b = String::with_capacity(8192);
        b.push_str(&format!("<!doctype html><html lang=\"{}\"><head>", self.lang));
        b.push_str("<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
        b.push_str(&format!("<title>{}</title><style>{CSS}</style></head><body><main>", esc(&clip(self.question.trim(), 80))));
        b.push_str(&format!(
            "<header><p class=\"eyebrow\">{} · {}</p><h1>{}</h1><div class=\"chips\"><span class=\"chip\">{n} {}</span><span class=\"chip ok\">{ok} {}</span>{}<span class=\"chip\">{} {}</span></div></header>",
            esc(l.answer),
            esc(&fmt_time(self.created)),
            esc(self.question.trim()),
            esc(l.tools),
            esc(l.ok),
            if failed > 0 { format!("<span class=\"chip bad\">{failed} {}</span>", esc(l.failed)) } else { String::new() },
            self.sections.len(),
            esc(l.categories)
        ));
        if self.sections.is_empty() {
            b.push_str(&format!("<section class=\"empty\"><p>{}</p></section>", esc(l.empty)));
        }
        for sec in &self.sections {
            let c = &CATEGORIES[sec.category];
            b.push_str(&format!(
                "<section data-category=\"{}\"><h2><span class=\"icon\" aria-hidden=\"true\">{}</span>{}<span class=\"count\">{}</span></h2>",
                c.key,
                c.icon,
                esc(l.cats[sec.category]),
                sec.cards.len()
            ));
            for card in &sec.cards {
                b.push_str(&format!(
                    "<article class=\"card st-{}\"><div class=\"card-head\"><code>{}</code><span class=\"badge b-{}\">{}</span><span class=\"ms\">{} ms</span></div>",
                    card.status.name(),
                    esc(&card.tool),
                    card.status.name(),
                    esc(self.status_label(card.status)),
                    card.ms
                ));
                for it in &card.items {
                    item_html(&mut b, it, l);
                }
                if card.truncated {
                    b.push_str(&format!("<p class=\"note\">{}</p>", esc(l.truncated)));
                }
                b.push_str("</article>");
            }
            b.push_str("</section>");
        }
        b.push_str(&format!(
            "<footer>{} · layout v{LAYOUT_VERSION} · <code>{}</code></footer></main></body></html>",
            esc(l.footer),
            esc(&self.id)
        ));
        b
    }
}

fn item_html(b: &mut String, it: &Item, l: &Labels) {
    let title = |b: &mut String, t: &Option<String>| {
        if let Some(t) = t {
            b.push_str(&format!("<h3>{}</h3>", esc(t)));
        }
    };
    match it {
        Item::Metric { label, value } => b.push_str(&format!(
            "<div class=\"metric\"><span class=\"value\">{}</span><span class=\"label\">{}</span></div>",
            esc(value),
            esc(label)
        )),
        Item::Fields { title: t, rows } => {
            title(b, t);
            b.push_str("<dl>");
            for (k, v) in rows {
                b.push_str(&format!("<dt>{}</dt><dd>{}</dd>", esc(k), esc(v)));
            }
            b.push_str("</dl>");
        }
        Item::Table { title: t, columns, rows, more } => {
            title(b, t);
            b.push_str("<div class=\"scroll\"><table><thead><tr>");
            for c in columns {
                b.push_str(&format!("<th>{}</th>", esc(c)));
            }
            b.push_str("</tr></thead><tbody>");
            for r in rows {
                b.push_str("<tr>");
                for c in r {
                    b.push_str(&format!("<td>{}</td>", esc(c)));
                }
                b.push_str("</tr>");
            }
            b.push_str("</tbody></table></div>");
            if *more > 0 {
                b.push_str(&format!("<p class=\"note\">+{more} {}</p>", esc(l.more)));
            }
        }
        Item::List { title: t, items, more } => {
            title(b, t);
            b.push_str("<ul>");
            for x in items {
                b.push_str(&format!("<li>{}</li>", esc(x)));
            }
            if *more > 0 {
                b.push_str(&format!("<li class=\"note\">+{more} {}</li>", esc(l.more)));
            }
            b.push_str("</ul>");
        }
        Item::Text(t) if t.chars().count() > TEXT_PREVIEW => {
            // long output: a short preview, the rest folded away
            let head: String = t.chars().take(TEXT_PREVIEW).collect();
            b.push_str(&format!(
                "<p class=\"text\">{}…</p><details><summary>{}</summary><p class=\"text\">{}</p></details>",
                esc(&head),
                esc(l.show_all),
                esc(t)
            ));
        }
        Item::Text(t) => b.push_str(&format!("<p class=\"text\">{}</p>", esc(t))),
        Item::Error(e) => b.push_str(&format!("<p class=\"error\" role=\"alert\">{}</p>", esc(e))),
    }
}

/// HTML-escape text.
pub fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

const CSS: &str = ":root{--bg:#f6f7f9;--surface:#fff;--text:#16181d;--muted:#5d6470;--line:#e3e6eb;--accent:#2f6fde;--ok:#1e7f4f;--ok-bg:#e3f4ea;--bad:#b42318;--bad-bg:#fde8e7;--warn:#9a5b00;--warn-bg:#fff1d6;color-scheme:light dark}\
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--surface:#181b21;--text:#e8eaee;--muted:#9aa2ae;--line:#2a2f38;--accent:#7aa7ff;--ok:#5fd394;--ok-bg:#15301f;--bad:#ff8a80;--bad-bg:#3a1714;--warn:#ffc266;--warn-bg:#3a2a0d}}\
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,'Segoe UI',Roboto,'Noto Sans Telugu','Noto Sans Devanagari','Noto Sans Tamil','Noto Sans Kannada','Noto Sans Malayalam','Noto Sans Bengali','Noto Sans Gujarati','Noto Sans Gurmukhi','Noto Sans Oriya',sans-serif}\
main{max-width:760px;margin:0 auto;padding:16px}header{padding:8px 0 12px}.eyebrow{margin:0;color:var(--muted);font-size:13px}\
h1{margin:4px 0 10px;font-size:22px;line-height:1.3;overflow-wrap:anywhere}.chips{display:flex;flex-wrap:wrap;gap:6px}\
.chip{font-size:12px;padding:2px 10px;border-radius:999px;background:var(--surface);border:1px solid var(--line);color:var(--muted)}\
.chip.ok{color:var(--ok);background:var(--ok-bg);border-color:transparent}.chip.bad{color:var(--bad);background:var(--bad-bg);border-color:transparent}\
section{margin-top:18px}h2{display:flex;align-items:center;gap:8px;font-size:16px;margin:0 0 8px}.icon{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:8px;background:var(--surface);border:1px solid var(--line);color:var(--accent);font-size:14px}\
.count{margin-left:auto;font-size:12px;color:var(--muted);font-weight:400}\
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px}\
.card-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}.card-head code{font-size:13px;font-weight:600;overflow-wrap:anywhere}\
.badge{font-size:11px;padding:1px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.03em}.b-ok{color:var(--ok);background:var(--ok-bg)}.b-error{color:var(--bad);background:var(--bad-bg)}.b-timeout{color:var(--warn);background:var(--warn-bg)}\
.ms{margin-left:auto;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}\
h3{font-size:13px;color:var(--muted);margin:10px 0 4px;font-weight:600}dl{display:grid;grid-template-columns:minmax(90px,40%) 1fr;gap:2px 12px;margin:0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}\
.metric{display:flex;flex-direction:column}.metric .value{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.metric .label{color:var(--muted);font-size:13px}\
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}th{color:var(--muted);font-weight:600;font-size:12px}\
ul{margin:0;padding-left:18px}.text{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.error{margin:0 0 6px;padding:8px 10px;border-radius:8px;background:var(--bad-bg);color:var(--bad);overflow-wrap:anywhere}\
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13px}.text{word-break:break-word}\
.note{color:var(--muted);font-size:12px;margin:6px 0 0}.empty{background:var(--surface);border:1px dashed var(--line);border-radius:12px;padding:16px;color:var(--muted)}\
footer{margin:24px 0 8px;color:var(--muted);font-size:12px}";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorizes_by_declaration_and_by_keywords() {
        assert_eq!(CATEGORIES[categorize("x", "", Some("Weather"))].key, "weather");
        assert_eq!(CATEGORIES[categorize("x", "", Some("finance"))].key, "finance");
        assert_eq!(CATEGORIES[categorize("weather.now", "Current conditions for a city", None)].key, "weather");
        assert_eq!(CATEGORIES[categorize("calendar.today", "Today's events", None)].key, "calendar");
        assert_eq!(CATEGORIES[categorize("bank.balance", "Account balance", None)].key, "finance");
        assert_eq!(CATEGORIES[categorize("battery.status", "Phone battery level", None)].key, "device");
        assert_eq!(CATEGORIES[categorize("zz.qq", "frobnicate", None)].key, "other");
    }

    #[test]
    fn formats_values_consistently() {
        assert_eq!(fmt_value(&json!(1234567)), "1,234,567");
        assert_eq!(fmt_value(&json!(-1234)), "-1,234");
        assert_eq!(fmt_value(&json!(1.23456)), "1.23");
        assert_eq!(fmt_value(&json!(2.5)), "2.5");
        assert_eq!(fmt_value(&json!(1000.0)), "1,000");
        assert_eq!(fmt_value(&json!(true)), "Yes");
        assert_eq!(fmt_value(&Value::Null), "—");
        assert_eq!(humanize("annualRate"), "Annual rate");
        assert_eq!(humanize("temp_c"), "Temp c");
        assert_eq!(fmt_time(0), "1970-01-01 00:00 UTC");
        assert_eq!(fmt_time(1_762_700_000), "2025-11-09 14:53 UTC");
    }

    #[test]
    fn indian_locales_group_digits_in_lakhs_and_translate_labels() {
        let te = Locale::for_lang("te");
        assert_eq!(fmt_value_in(&json!(1234567), &te), "12,34,567");
        assert_eq!(fmt_value_in(&json!(123456789.5), &te), "12,34,56,789.5");
        assert_eq!(fmt_value_in(&json!(999), &te), "999");
        assert_eq!(fmt_value_in(&json!(true), &te), "అవును");
        assert_eq!(fmt_value_in(&json!(1234567), &LOCALE_EN), "1,234,567");
        assert_eq!(Locale::for_lang("en-IN").labels.answer, "Answer");
        assert!(Locale::for_lang("en-IN").indian);
        assert!(Locale::for_lang("ta").indian && Locale::for_lang("ta").lang == "ta");
        assert_eq!(Locale::detect("ఈరోజు వాతావరణం").lang, "te");
        assert_eq!(Locale::detect("आज का मौसम").labels.cats[0], "मौसम");
        let card = Card::new_in("bank.balance", Status::Ok, 1, json!({}), &json!({"balance": 250000}), None, false, &te);
        let p = build_in("id", "బ్యాలెన్స్", 0, vec![(3, card)], &te);
        assert!(p.to_html().contains("2,50,000") && p.to_html().contains("డబ్బు &amp; ఆర్థికం"));
        assert_eq!(p.speech_text(), "డబ్బు & ఆర్థికం. Balance 2,50,000.");
        assert_eq!(rie_core::speech::verbalize(&p.speech_text(), "te"), "డబ్బు & ఆర్థికం. Balance రెండు లక్షల యాభై వేలు.");
    }

    #[test]
    fn normalizes_shapes() {
        assert!(
            matches!(&normalize("bank.balance", &json!(12.5))[0], Item::Metric { label, value } if label == "Balance" && value == "12.5")
        );
        let it = normalize("t", &json!([{"a": 1, "b": "x"}, {"a": 2, "c": true}]));
        assert!(
            matches!(&it[0], Item::Table { columns, rows, .. } if columns == &vec!["A".to_string(), "B".into(), "C".into()] && rows[1][2] == "Yes")
        );
        let it = normalize("t", &json!({"city": "Paris", "temp": 21, "hours": [1, 2], "wind": {"speed": 3}}));
        assert_eq!(it.len(), 3);
        assert!(matches!(&normalize("t", &json!({"error": "boom"}))[0], Item::Error(e) if e == "boom"));
        let many: Vec<Value> = (0..50).map(|i| json!(i)).collect();
        assert!(matches!(&normalize("t", &Value::Array(many))[0], Item::List { items, more: 30, .. } if items.len() == 20));
    }

    #[test]
    fn html_is_escaped_and_stable() {
        let card = Card::new("evil", Status::Ok, 3, json!({}), &json!({"x": "<script>alert(1)</script>"}), None, false);
        let p = build("id1", "Q <b>", 0, vec![(9, card)]);
        let h = p.to_html();
        assert!(!h.contains("<script>alert") && h.contains("&lt;script&gt;") && h.contains("Q &lt;b&gt;"));
        assert_eq!(h, p.to_html());
        assert!(p.to_markdown().contains("### • Other"));
    }
}
