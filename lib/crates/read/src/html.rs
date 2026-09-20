//! HTML to text.
//!
//! Output is rendered as plain text with markdown-like structure, so the
//! `markdown` chunker can consume it directly and a web page keeps its heading
//! hierarchy instead of collapsing into one blob.

use ragworks_core::{Component, Error, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::text::decode;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HtmlConfig {
    /// Wrap width. Wide enough that paragraphs are not hard-wrapped mid-sentence,
    /// which would corrupt sentence-boundary chunking downstream.
    #[serde(default = "default_width")]
    pub width: usize,
}

fn default_width() -> usize {
    10_000
}

impl Default for HtmlConfig {
    fn default() -> Self {
        Self { width: default_width() }
    }
}

#[derive(Debug)]
pub struct Html {
    width: usize,
}

impl Component for Html {
    type Config = HtmlConfig;
    const NAME: &'static str = "html";
    const SUMMARY: &'static str = "HTML rendered to structured plain text.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.width == 0 {
            return Err(Error::config(Self::NAME, "width must be > 0"));
        }
        Ok(Self { width: config.width })
    }
}

/// Pull `<title>` out of the raw markup. Done by hand because the renderer
/// drops it, and a document's title is worth more than the tag it came from.
fn title_of(raw: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open_end = lower[start..].find('>')? + start + 1;
    let end = lower[open_end..].find("</title>")? + open_end;
    let t = raw[open_end..end].trim();
    (!t.is_empty()).then(|| t.to_string())
}

impl Reader for Html {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["html", "htm", "xhtml"]
    }

    fn read(&self, bytes: &[u8], _uri: &str) -> Result<Extracted> {
        let (raw, encoding, lossy) = decode(bytes, None);
        let text = html2text::from_read(raw.as_bytes(), self.width)
            .map_err(|e| Error::provider("html", None, format!("render failed: {e}")))?;
        let mut meta = serde_json::json!({"encoding": encoding, "lossy": lossy});
        if let Some(t) = title_of(&raw) {
            meta["title"] = serde_json::Value::String(t);
        }
        Ok(Extracted { text, meta })
    }
}
