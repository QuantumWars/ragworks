//! One error type, with retry classification built in.
//!
//! Borrowed from agno's `EmbeddingError`, which normalises every provider's
//! exception into one type while preserving the HTTP status. The reason that
//! matters: a caller must be able to tell a rate limit from an auth failure
//! from a malformed request, because only one of those is worth retrying.
//! Against a rate-limited free tier that distinction is the difference between
//! a run that finishes and a run that burns its quota on doomed retries.

use std::fmt;

/// Why a provider call failed, in terms the retry logic can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderFault {
    /// 429 and friends. Back off and retry.
    RateLimited,
    /// Network timeout or connection reset. Retry.
    Timeout,
    /// 5xx. Retry, but a repeat means the provider is down.
    Server,
    /// 401/403. Retrying cannot help.
    Auth,
    /// 400/422. The request is wrong; retrying sends the same wrong request.
    Invalid,
}

impl ProviderFault {
    /// Classify from an HTTP status when the provider gave us one.
    pub fn from_status(status: u16) -> Self {
        match status {
            429 => Self::RateLimited,
            401 | 403 => Self::Auth,
            408 | 504 => Self::Timeout,
            500..=599 => Self::Server,
            _ => Self::Invalid,
        }
    }

    pub fn is_retryable(self) -> bool {
        matches!(self, Self::RateLimited | Self::Timeout | Self::Server)
    }
}

impl fmt::Display for ProviderFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::RateLimited => "rate limited",
            Self::Timeout => "timeout",
            Self::Server => "server error",
            Self::Auth => "authentication failed",
            Self::Invalid => "invalid request",
        };
        f.write_str(s)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{provider}: {fault}{}: {message}", status.map(|s| format!(" (HTTP {s})")).unwrap_or_default())]
    Provider {
        provider: String,
        fault: ProviderFault,
        status: Option<u16>,
        message: String,
    },

    #[error("config for {plugin:?}: {message}")]
    Config { plugin: String, message: String },

    #[error("no {kind} named {name:?}; registered: [{available}]")]
    UnknownPlugin {
        kind: &'static str,
        name: String,
        available: String,
    },

    #[error("{kind} {name:?} is already registered")]
    DuplicatePlugin { kind: &'static str, name: String },

    #[error("document {0} is not in this corpus")]
    NoSuchDoc(u64),

    #[error("span {start}..{end} is not on a character boundary of document {doc}")]
    UnalignedSpan { doc: u64, start: u64, end: u64 },

    #[error("dimension mismatch: index holds {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// True when retrying the identical call could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { fault, .. } => fault.is_retryable(),
            Self::Io(e) => matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::ConnectionReset
            ),
            _ => false,
        }
    }

    pub fn provider(provider: impl Into<String>, status: Option<u16>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            fault: status.map_or(ProviderFault::Invalid, ProviderFault::from_status),
            status,
            message: message.into(),
        }
    }

    pub fn config(plugin: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Config { plugin: plugin.into(), message: message.into() }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limits_and_server_errors_retry_auth_does_not() {
        assert!(ProviderFault::from_status(429).is_retryable());
        assert!(ProviderFault::from_status(503).is_retryable());
        assert!(!ProviderFault::from_status(401).is_retryable());
        assert!(!ProviderFault::from_status(422).is_retryable());
    }

    #[test]
    fn provider_error_keeps_status_and_classifies() {
        let e = Error::provider("openrouter", Some(429), "slow down");
        assert!(e.is_retryable());
        assert!(e.to_string().contains("429"), "status must survive into the message: {e}");
    }

    #[test]
    fn a_provider_error_without_a_status_is_not_retried_blindly() {
        // No status means we could not classify it. Treating that as retryable
        // would turn an unknown permanent failure into an infinite loop.
        let e = Error::provider("local", None, "malformed response");
        assert!(!e.is_retryable());
    }
}
