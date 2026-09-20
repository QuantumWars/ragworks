//! JSON and line-delimited JSON.
//!
//! Handles both from one reader: a whole-file parse is tried first, and a
//! failure falls back to parsing line by line. Guessing from the extension
//! alone is unreliable, since `.json` files holding one object per line are
//! common in practice.

use ragworks_core::{Component, Error, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::text::decode;

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct JsonConfig {
    /// JSON Pointer paths (RFC 6901) to extract, e.g. `/title` or `/body/text`.
    /// Empty means every string leaf, in document order.
    #[serde(default)]
    pub fields: Vec<String>,
    /// Prefix each value with the field it came from.
    #[serde(default)]
    pub label_fields: bool,
}

#[derive(Debug, Default)]
pub struct Json {
    fields: Vec<String>,
    label_fields: bool,
}

impl Component for Json {
    type Config = JsonConfig;
    const NAME: &'static str = "json";
    const SUMMARY: &'static str = "JSON or JSONL; extracts chosen pointers, or every string leaf.";

    fn build(config: Self::Config) -> Result<Self> {
        for f in &config.fields {
            if !f.starts_with('/') {
                return Err(Error::config(
                    Self::NAME,
                    format!("{f:?} is not a JSON Pointer; paths must start with '/'"),
                ));
            }
        }
        Ok(Self { fields: config.fields, label_fields: config.label_fields })
    }
}

/// Every string leaf, depth-first. Numbers and booleans are skipped: they carry
/// no retrievable language, and including them mostly adds noise.
fn string_leaves(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => {
            if !s.trim().is_empty() {
                out.push(s.clone());
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| string_leaves(x, out)),
        serde_json::Value::Object(m) => m.values().for_each(|x| string_leaves(x, out)),
        _ => {}
    }
}

impl Json {
    fn record_text(&self, v: &serde_json::Value) -> String {
        let mut parts = Vec::new();
        if self.fields.is_empty() {
            string_leaves(v, &mut parts);
        } else {
            for f in &self.fields {
                let Some(found) = v.pointer(f) else { continue };
                let mut vals = Vec::new();
                string_leaves(found, &mut vals);
                for val in vals {
                    parts.push(if self.label_fields {
                        format!("{}: {val}", f.trim_start_matches('/'))
                    } else {
                        val
                    });
                }
            }
        }
        parts.join("\n")
    }
}

impl Reader for Json {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["json", "jsonl", "ndjson"]
    }

    fn read(&self, bytes: &[u8], _uri: &str) -> Result<Extracted> {
        let (raw, encoding, lossy) = decode(bytes, None);

        let records: Vec<serde_json::Value> = match serde_json::from_str(&raw) {
            Ok(serde_json::Value::Array(a)) => a,
            Ok(v) => vec![v],
            Err(whole_file_err) => {
                let mut v = Vec::new();
                for (n, line) in raw.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    v.push(serde_json::from_str(line).map_err(|e| {
                        Error::config(
                            Self::NAME,
                            format!(
                                "not valid JSON ({whole_file_err}) nor JSON Lines (line {}: {e})",
                                n + 1
                            ),
                        )
                    })?);
                }
                v
            }
        };

        let text = records
            .iter()
            .map(|r| self.record_text(r))
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(Extracted {
            text,
            meta: serde_json::json!({
                "records": records.len(), "encoding": encoding, "lossy": lossy
            }),
        })
    }
}
