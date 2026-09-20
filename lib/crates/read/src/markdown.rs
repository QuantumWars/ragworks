//! Markdown, with YAML front matter lifted into metadata.
//!
//! The body is passed through unchanged rather than rendered: the heading
//! structure is the point, and the `markdown` chunker consumes it directly.

use ragworks_core::{Component, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::text::decode;

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct MarkdownConfig {
    /// Parse a leading `---` fenced block into metadata.
    #[serde(default = "yes")]
    pub front_matter: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug)]
pub struct Markdown {
    front_matter: bool,
}

impl Component for Markdown {
    type Config = MarkdownConfig;
    const NAME: &'static str = "markdown";
    const SUMMARY: &'static str = "Markdown, preserving heading structure and lifting front matter.";

    fn build(config: Self::Config) -> Result<Self> {
        Ok(Self { front_matter: config.front_matter })
    }
}

/// Split a leading `---` block from the body. Only `key: value` pairs are
/// understood; anything else is left in place rather than guessed at, since a
/// wrong parse is worse than no parse.
fn split_front_matter(text: &str) -> (serde_json::Value, &str) {
    let rest = match text.strip_prefix("---\n") {
        Some(r) => r,
        None => return (serde_json::Value::Null, text),
    };
    let Some(end) = rest.find("\n---") else {
        return (serde_json::Value::Null, text);
    };
    let (block, after) = rest.split_at(end);
    let body = after.trim_start_matches("\n---").trim_start_matches(['\r', '\n']);

    let mut map = serde_json::Map::new();
    for line in block.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            let v = v.trim().trim_matches(['"', '\'']);
            if !k.is_empty() && !k.starts_with('#') {
                map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
            }
        }
    }
    if map.is_empty() {
        (serde_json::Value::Null, body)
    } else {
        (serde_json::Value::Object(map), body)
    }
}

/// First level-one heading, as a title.
fn first_h1(text: &str) -> Option<&str> {
    text.lines().find_map(|l| l.strip_prefix("# ").map(str::trim))
}

impl Reader for Markdown {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["md", "markdown", "mdx"]
    }

    fn read(&self, bytes: &[u8], _uri: &str) -> Result<Extracted> {
        let (raw, encoding, lossy) = decode(bytes, None);
        let (front, body) = if self.front_matter {
            split_front_matter(&raw)
        } else {
            (serde_json::Value::Null, raw.as_str())
        };

        let mut meta = serde_json::json!({"encoding": encoding, "lossy": lossy});
        if let Some(t) = first_h1(body) {
            meta["title"] = serde_json::Value::String(t.to_string());
        }
        if let serde_json::Value::Object(m) = front {
            meta["front_matter"] = serde_json::Value::Object(m);
        }
        Ok(Extracted { text: body.to_string(), meta })
    }
}
