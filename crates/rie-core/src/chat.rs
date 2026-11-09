//! Chat prompt formatting for the common model families.

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    pub fn parse(s: &str) -> Option<Role> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" | "model" => Some(Role::Assistant),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message { role, content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    Llama3,
    ChatMl,
    Gemma,
    Phi3,
    Mistral,
    Granite,
    /// Plain concatenation (base models).
    Raw,
}

impl ChatTemplate {
    /// Pick a template from the GGUF `tokenizer.chat_template` string / vocabulary.
    pub fn detect(tok: &Tokenizer) -> Self {
        let t = tok.chat_template.as_deref().unwrap_or("");
        let has = |s: &str| t.contains(s) || (t.is_empty() && tok.token_id(s).is_some());
        if has("<|start_header_id|>") {
            ChatTemplate::Llama3
        } else if has("<|im_start|>") {
            ChatTemplate::ChatMl
        } else if has("<start_of_turn>") {
            ChatTemplate::Gemma
        } else if has("<|start_of_role|>") {
            ChatTemplate::Granite
        } else if has("<|assistant|>") && has("<|end|>") {
            ChatTemplate::Phi3
        } else if t.contains("[INST]") {
            ChatTemplate::Mistral
        } else {
            ChatTemplate::Raw
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "llama3" => ChatTemplate::Llama3,
            "chatml" | "qwen" => ChatTemplate::ChatMl,
            "gemma" => ChatTemplate::Gemma,
            "phi3" | "phi" => ChatTemplate::Phi3,
            "mistral" | "llama2" => ChatTemplate::Mistral,
            "granite" => ChatTemplate::Granite,
            "raw" | "none" => ChatTemplate::Raw,
            _ => return None,
        })
    }

    /// Render the conversation. BOS is *not* included (the tokenizer adds it).
    pub fn render(&self, msgs: &[Message], add_generation_prompt: bool) -> String {
        let mut s = String::new();
        match self {
            ChatTemplate::Llama3 => {
                for m in msgs {
                    s += &format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", m.role.as_str(), m.content.trim());
                }
                if add_generation_prompt {
                    s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
                }
            }
            ChatTemplate::ChatMl => {
                for m in msgs {
                    s += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role.as_str(), m.content);
                }
                if add_generation_prompt {
                    s += "<|im_start|>assistant\n";
                }
            }
            ChatTemplate::Gemma => {
                let mut system = String::new();
                for m in msgs {
                    match m.role {
                        Role::System => system = format!("{}\n\n", m.content.trim()),
                        Role::User => {
                            s += &format!("<start_of_turn>user\n{}{}<end_of_turn>\n", system, m.content.trim());
                            system.clear();
                        }
                        Role::Assistant => s += &format!("<start_of_turn>model\n{}<end_of_turn>\n", m.content.trim()),
                    }
                }
                if add_generation_prompt {
                    s += "<start_of_turn>model\n";
                }
            }
            ChatTemplate::Phi3 => {
                for m in msgs {
                    s += &format!("<|{}|>\n{}<|end|>\n", m.role.as_str(), m.content);
                }
                if add_generation_prompt {
                    s += "<|assistant|>\n";
                }
            }
            ChatTemplate::Mistral => {
                let mut system = String::new();
                for m in msgs {
                    match m.role {
                        Role::System => system = format!("{}\n\n", m.content.trim()),
                        Role::User => {
                            s += &format!("[INST] {}{} [/INST]", system, m.content.trim());
                            system.clear();
                        }
                        Role::Assistant => s += &format!("{}</s>", m.content.trim()),
                    }
                }
            }
            ChatTemplate::Granite => {
                for m in msgs {
                    s += &format!("<|start_of_role|>{}<|end_of_role|>{}<|end_of_text|>\n", m.role.as_str(), m.content);
                }
                if add_generation_prompt {
                    s += "<|start_of_role|>assistant<|end_of_role|>";
                }
            }
            ChatTemplate::Raw => {
                for m in msgs {
                    s += &m.content;
                    s.push('\n');
                }
            }
        }
        s
    }
}
