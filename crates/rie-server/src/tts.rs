//! Text-to-speech through a pluggable on-device engine (`--tts SPEC`).
//!
//! The server prepares the text (language-aware verbalization of numbers, currency,
//! times; sentence chunking, see `rie_core::speech`) and hands it to an engine:
//!
//! * `termux` — Android's system TTS via Termux:API (`termux-tts-speak`). Uses whatever
//!   voices the phone has; Google's speech services offer offline Telugu and Hindi.
//!   Plays on the phone's speaker.
//! * `cmd:PROGRAM ARGS…` — any local engine. Placeholders: `{text}` (the text as one
//!   argument), `{lang}`, `{voice}`, `{out}` (a WAV path to write). With `{out}` the
//!   engine produces audio files (returned by `/v1/audio/speech`); without it, it is
//!   expected to play the audio itself. Examples:
//!   `cmd:sherpa-onnx-offline-tts --vits-model=te.onnx --vits-tokens=tokens.txt --output-filename={out} {text}`,
//!   `cmd:espeak-ng -v {lang} {text}`. No shell is involved (no injection).
//!
//! Speaking engines get one sentence at a time from a queue, so the phone starts talking
//! after the first sentence of a reply instead of waiting for the whole reply.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Sender};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    Termux,
    Command(Vec<String>),
}

impl Backend {
    pub fn parse(spec: &str) -> Result<Backend, String> {
        if spec == "termux" {
            return Ok(Backend::Termux);
        }
        if let Some(c) = spec.strip_prefix("cmd:") {
            let argv: Vec<String> = c.split_whitespace().map(str::to_string).collect();
            if argv.is_empty() {
                return Err("--tts cmd: needs a program".into());
            }
            return Ok(Backend::Command(argv));
        }
        Err(format!("unknown --tts {spec} (use termux or cmd:PROGRAM ARGS)"))
    }

    /// True if the engine writes audio files (`{out}`) instead of playing.
    pub fn writes_files(&self) -> bool {
        matches!(self, Backend::Command(a) if a.iter().any(|x| x.contains("{out}")))
    }
}

pub enum Spoken {
    /// the engine played the audio
    Played,
    /// WAV bytes
    Audio(Vec<u8>),
}

const TIMEOUT: Duration = Duration::from_secs(120);

fn run(mut cmd: Command, stdin: Option<&str>) -> Result<(), String> {
    cmd.stdout(Stdio::null()).stderr(Stdio::piped()).stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
    let mut child = cmd.spawn().map_err(|e| format!("cannot start TTS engine: {e}"))?;
    if let (Some(text), Some(mut si)) = (stdin, child.stdin.take()) {
        let _ = si.write_all(text.as_bytes());
    }
    let t0 = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) if st.success() => return Ok(()),
            Ok(Some(st)) => {
                let mut err = String::new();
                if let Some(mut e) = child.stderr.take() {
                    use std::io::Read;
                    let _ = e.read_to_string(&mut err);
                }
                return Err(format!("TTS engine failed ({st}): {}", err.trim().chars().take(300).collect::<String>()));
            }
            Ok(None) if t0.elapsed() > TIMEOUT => {
                let _ = child.kill();
                return Err("TTS engine timed out".into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Synthesize (or play) already-prepared text.
pub fn speak(backend: &Backend, text: &str, lang: &str, voice: &str) -> Result<Spoken, String> {
    // control characters (NUL, escapes) cannot be passed to a program and mean nothing
    // to a voice; replacement characters are noise
    let clean: String = text.chars().map(|c| if (c.is_control() && c != '\n') || c == '\u{FFFD}' { ' ' } else { c }).collect();
    let text = clean.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Ok(Spoken::Played);
    }
    let text = text.as_str();
    match backend {
        Backend::Termux => {
            let mut c = Command::new("termux-tts-speak");
            if !lang.is_empty() {
                c.args(["-l", lang]);
            }
            run(c, Some(text)).map(|_| Spoken::Played)
        }
        Backend::Command(argv) => {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let out: PathBuf = std::env::temp_dir().join(format!(
                "rie-tts-{}-{}.wav",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let out_s = out.to_string_lossy().into_owned();
            let args: Vec<String> = argv
                .iter()
                .map(|a| a.replace("{text}", text).replace("{lang}", lang).replace("{voice}", voice).replace("{out}", &out_s))
                .collect();
            let mut c = Command::new(&args[0]);
            c.args(&args[1..]);
            run(c, None)?;
            if backend.writes_files() {
                let bytes = std::fs::read(&out).map_err(|e| format!("TTS engine wrote no audio: {e}"))?;
                let _ = std::fs::remove_file(&out);
                Ok(Spoken::Audio(bytes))
            } else {
                Ok(Spoken::Played)
            }
        }
    }
}

/// Concatenate WAV files with the same format (PCM data chunks appended).
pub fn concat_wav(parts: &[Vec<u8>]) -> Option<Vec<u8>> {
    fn split(w: &[u8]) -> Option<(&[u8], &[u8])> {
        if w.len() < 12 || &w[..4] != b"RIFF" || &w[8..12] != b"WAVE" {
            return None;
        }
        let mut i = 12;
        let mut fmt = None;
        while i + 8 <= w.len() {
            let id = &w[i..i + 4];
            let n = u32::from_le_bytes(w[i + 4..i + 8].try_into().ok()?) as usize;
            let body = w.get(i + 8..(i + 8 + n).min(w.len()))?;
            if id == b"fmt " {
                fmt = Some(body);
            } else if id == b"data" {
                return Some((fmt?, body));
            }
            i += 8 + n + (n & 1);
        }
        None
    }
    let first = split(parts.first()?)?;
    let mut data = Vec::new();
    for p in parts {
        let (f, d) = split(p)?;
        if f != first.0 {
            return None;
        }
        data.extend_from_slice(d);
    }
    let fmt = first.0;
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((4 + 8 + fmt.len() + 8 + data.len()) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    out.extend_from_slice(fmt);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    Some(out)
}

/// Prepare text for speech (verbalize in `lang`) and speak it sentence by sentence;
/// audio from file-writing engines is joined into one WAV.
pub fn say(backend: &Backend, text: &str, lang: &str, voice: &str) -> Result<(Spoken, String, usize), String> {
    let prepared = rie_core::speech::verbalize(text, lang);
    let chunks = rie_core::speech::sentences(&prepared, 300);
    if chunks.is_empty() {
        return Err("nothing to say".into());
    }
    let mut audio = Vec::new();
    for c in &chunks {
        if let Spoken::Audio(a) = speak(backend, c, lang, voice)? {
            audio.push(a);
        }
    }
    let spoken = if audio.is_empty() {
        Spoken::Played
    } else if audio.len() == 1 {
        Spoken::Audio(audio.pop().unwrap())
    } else {
        Spoken::Audio(concat_wav(&audio).ok_or("TTS engine produced WAV files with different formats")?)
    };
    Ok((spoken, prepared, chunks.len()))
}

/// Speaks streamed text as it arrives: complete sentences go to a background queue.
pub struct Speaker {
    tx: Option<Sender<String>>,
    pending: String,
    lang: String,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Speaker {
    /// Only for engines that play audio themselves.
    pub fn start(backend: Backend, lang: &str, voice: &str) -> Speaker {
        let (tx, rx) = channel::<String>();
        let (l, v) = (lang.to_string(), voice.to_string());
        let handle = std::thread::spawn(move || {
            for sentence in rx {
                if let Err(e) = speak(&backend, &sentence, &l, &v) {
                    eprintln!("rie-server: speech failed: {e}");
                }
            }
        });
        Speaker { tx: Some(tx), pending: String::new(), lang: lang.to_string(), handle: Some(handle) }
    }

    /// Add streamed text; every completed sentence is queued for speaking.
    pub fn feed(&mut self, piece: &str) {
        self.pending.push_str(piece);
        let (done, rest) = rie_core::speech::split_complete(&self.pending);
        if !done.is_empty() {
            for c in rie_core::speech::sentences(done, 300) {
                self.send(&c);
            }
            self.pending = rest.to_string();
        }
    }

    fn send(&self, s: &str) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(rie_core::speech::verbalize(s, &self.lang));
        }
    }

    /// Speak what is left and wait until everything has been spoken.
    pub fn finish(mut self) {
        let rest = std::mem::take(&mut self.pending);
        if !rest.trim().is_empty() {
            self.send(rest.trim());
        }
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
        fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
        fmt.extend_from_slice(&16000u32.to_le_bytes());
        fmt.extend_from_slice(&32000u32.to_le_bytes());
        fmt.extend_from_slice(&2u16.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        concat_wav_raw(&fmt, &data)
    }

    fn concat_wav_raw(fmt: &[u8], data: &[u8]) -> Vec<u8> {
        let mut o = b"RIFF".to_vec();
        o.extend_from_slice(&((4 + 8 + fmt.len() + 8 + data.len()) as u32).to_le_bytes());
        o.extend_from_slice(b"WAVEfmt ");
        o.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        o.extend_from_slice(fmt);
        o.extend_from_slice(b"data");
        o.extend_from_slice(&(data.len() as u32).to_le_bytes());
        o.extend_from_slice(data);
        o
    }

    #[test]
    fn parses_backends() {
        assert_eq!(Backend::parse("termux").unwrap(), Backend::Termux);
        let b = Backend::parse("cmd:engine --out={out} {text}").unwrap();
        assert!(b.writes_files());
        assert!(!Backend::parse("cmd:espeak-ng -v {lang} {text}").unwrap().writes_files());
        assert!(Backend::parse("cmd:").is_err() && Backend::parse("nope").is_err());
    }

    #[test]
    fn joins_wav_files() {
        let j = concat_wav(&[wav(&[1, 2]), wav(&[3])]).unwrap();
        assert_eq!(j, wav(&[1, 2, 3]));
        assert!(concat_wav(&[b"not a wav".to_vec()]).is_none());
    }

    #[test]
    fn speaks_sentence_by_sentence_through_a_command() {
        // the "engine" appends each utterance to a file, one line per call
        let log = std::env::temp_dir().join(format!("rie-tts-log-{}", std::process::id()));
        let _ = std::fs::remove_file(&log);
        let script = format!("printf '%s|%s\\n' \"$0\" \"$1\" >> {}", log.display());
        let backend = Backend::Command(vec!["sh".into(), "-c".into(), script, "{lang}".into(), "{text}".into()]);
        let mut sp = Speaker::start(backend, "te", "");
        for piece in ["ఈరోజు ఉష్ణోగ్రత 18.4°C. ", "వర్షం ", "60% అవకాశం", " ఉంది."]
        {
            sp.feed(piece);
        }
        sp.finish();
        let got = std::fs::read_to_string(&log).unwrap();
        let lines: Vec<&str> = got.lines().collect();
        assert_eq!(lines.len(), 2, "{got}");
        assert_eq!(lines[0], "te|ఈరోజు ఉష్ణోగ్రత పద్దెనిమిది పాయింట్ నాలుగు డిగ్రీల సెల్సియస్.");
        assert_eq!(lines[1], "te|వర్షం అరవై శాతం అవకాశం ఉంది.");
        let _ = std::fs::remove_file(&log);
    }
}
