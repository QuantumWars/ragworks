//! Fixed-size chunking with overlap. The baseline everything is compared to.

use ragworks_core::{Chunk, Chunker, Component, DocView, Error, Result, Span};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FixedConfig {
    /// Target chunk size in bytes.
    #[serde(default = "default_size")]
    pub size: usize,
    /// Bytes of the previous chunk repeated at the start of the next.
    #[serde(default)]
    pub overlap: usize,
}

fn default_size() -> usize {
    1024
}

impl Default for FixedConfig {
    fn default() -> Self {
        Self { size: default_size(), overlap: 0 }
    }
}

#[derive(Debug)]
pub struct Fixed {
    size: usize,
    overlap: usize,
}

impl Component for Fixed {
    type Config = FixedConfig;
    const NAME: &'static str = "fixed";
    const SUMMARY: &'static str = "Fixed-size byte windows with optional overlap.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.size == 0 {
            return Err(Error::config(Self::NAME, "size must be > 0"));
        }
        // Rejected at build time rather than looping forever at run time: with
        // overlap >= size the window can never advance.
        if config.overlap >= config.size {
            return Err(Error::config(
                Self::NAME,
                format!("overlap ({}) must be < size ({})", config.overlap, config.size),
            ));
        }
        Ok(Self { size: config.size, overlap: config.overlap })
    }
}

/// Smallest index >= `i` that lies on a character boundary.
fn snap_up(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

impl Chunker for Fixed {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn chunk(&self, doc: DocView<'_>, out: &mut Vec<Chunk>) -> Result<()> {
        let text = doc.text;
        if text.is_empty() {
            return Ok(());
        }
        let mut start = 0usize;
        let mut ordinal = 0u32;
        loop {
            let end = snap_up(text, start + self.size);
            out.push(Chunk::leaf(
                doc.id,
                Span::new(start as u64, (end - start) as u32).offset_by(doc.base),
                ordinal,
            ));
            ordinal += 1;
            if end >= text.len() {
                break;
            }
            let next = snap_up(text, end.saturating_sub(self.overlap));
            // `overlap < size` is enforced in build(), but snapping could still
            // land us back on `start` for pathological multi-byte input.
            start = if next > start { next } else { end };
        }
        Ok(())
    }
}
