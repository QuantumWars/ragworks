//! Plain text, with an encoding fallback.
//!
//! Real corpora are not reliably UTF-8. A reader that assumes they are either
//! panics or silently drops content; this one decodes what it can, records the
//! encoding it used and whether anything was replaced, so a downstream quality
//! problem can be traced back to its cause.

use ragworks_core::{Component, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    /// Fallback encoding when the bytes are not valid UTF-8, e.g. "windows-1252".
    #[serde(default)]
    pub fallback_encoding: Option<String>,
}

#[derive(Debug, Default)]
pub struct Text {
    fallback: Option<&'static encoding_rs::Encoding>,
}

impl Component for Text {
    type Config = TextConfig;
    const NAME: &'static str = "text";
    const SUMMARY: &'static str = "Plain text, with a configurable non-UTF-8 fallback encoding.";

    fn build(config: Self::Config) -> Result<Self> {
        let fallback = match &config.fallback_encoding {
            None => None,
            Some(label) => Some(
                encoding_rs::Encoding::for_label(label.as_bytes()).ok_or_else(|| {
                    ragworks_core::Error::config(
                        Self::NAME,
                        format!("unknown encoding {label:?}"),
                    )
                })?,
            ),
        };
        Ok(Self { fallback })
    }
}

/// Decode bytes, reporting which encoding was used and whether characters were
/// replaced. A byte-order mark takes precedence over the configured fallback.
pub fn decode(
    bytes: &[u8],
    fallback: Option<&'static encoding_rs::Encoding>,
) -> (String, &'static str, bool) {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return (s.to_string(), "utf-8", false);
    }
    let enc = fallback.unwrap_or(encoding_rs::UTF_8);
    let (text, used, had_errors) = enc.decode(bytes);
    (text.into_owned(), used.name(), had_errors)
}

impl Reader for Text {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["txt", "text", "log", "rst"]
    }

    fn read(&self, bytes: &[u8], _uri: &str) -> Result<Extracted> {
        let (text, encoding, lossy) = decode(bytes, self.fallback);
        Ok(Extracted {
            text,
            meta: serde_json::json!({"encoding": encoding, "lossy": lossy}),
        })
    }
}
