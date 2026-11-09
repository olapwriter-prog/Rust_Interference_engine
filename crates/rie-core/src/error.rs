use std::fmt;

/// Unified error type for the engine.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The GGUF file is malformed or uses an unsupported feature.
    Format(String),
    /// The model uses an architecture or tensor type this engine does not support.
    Unsupported(String),
    /// A tensor or metadata key required by the architecture is missing.
    Missing(String),
    /// Runtime errors, e.g. the context window is full.
    Runtime(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(s) => write!(f, "invalid model file: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Missing(s) => write!(f, "missing: {s}"),
            Error::Runtime(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::error::Error::$kind(format!($($arg)*)))
    };
}
pub(crate) use bail;
