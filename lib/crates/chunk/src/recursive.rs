//! Recursive separator chunking -- the strategy most RAG stacks default to.
//!
//! Split on the most semantic separator that works (paragraph, then line, then
//! sentence, then word), and only fall through to a harder split when a piece
//! is still too large. Unlike the usual implementation, pieces are carried as
//! spans, so no substring is ever copied.

use ragworks_core::{Chunk, Chunker, Component, DocView, Error, Result, Span};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecursiveConfig {
    #[serde(default = "default_size")]
    pub size: usize,
    #[serde(default)]
    pub overlap: usize,
    /// Tried in order, most semantic first.
    #[serde(default = "default_separators")]
    pub separators: Vec<String>,
}

fn default_size() -> usize {
    1024
}
fn default_separators() -> Vec<String> {
    ["\n\n", "\n", ". ", " "].iter().map(|s| s.to_string()).collect()
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self { size: default_size(), overlap: 0, separators: default_separators() }
    }
}

#[derive(Debug)]
pub struct Recursive {
    size: usize,
    overlap: usize,
    separators: Vec<String>,
}

impl Component for Recursive {
    type Config = RecursiveConfig;
    const NAME: &'static str = "recursive";
    const SUMMARY: &'static str = "Split on the most semantic separator that fits; fall through as needed.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.size == 0 {
            return Err(Error::config(Self::NAME, "size must be > 0"));
        }
        if config.overlap >= config.size {
            return Err(Error::config(Self::NAME, "overlap must be < size"));
        }
        if config.separators.iter().any(|s| s.is_empty()) {
            return Err(Error::config(Self::NAME, "separators must not be empty strings"));
        }
        Ok(Self { size: config.size, overlap: config.overlap, separators: config.separators })
    }
}

impl Recursive {
    /// Break `[lo, hi)` into pieces no larger than `size` where the separators
    /// allow it. A run with no usable separator is emitted whole; splitting
    /// mid-word to hit a byte target loses more than the size bound gains.
    fn split(&self, text: &str, lo: usize, hi: usize, depth: usize, out: &mut Vec<(usize, usize)>) {
        if hi <= lo {
            return;
        }
        if hi - lo <= self.size || depth >= self.separators.len() {
            out.push((lo, hi));
            return;
        }
        let sep = &self.separators[depth];
        let slice = &text[lo..hi];
        let mut last = lo;
        let mut any = false;
        for (off, _) in slice.match_indices(sep.as_str()) {
            let cut = lo + off + sep.len();
            if cut > last {
                self.split(text, last, cut, depth + 1, out);
                last = cut;
                any = true;
            }
        }
        if any {
            self.split(text, last, hi, depth + 1, out);
        } else {
            self.split(text, lo, hi, depth + 1, out);
        }
    }
}

impl Chunker for Recursive {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn chunk(&self, doc: DocView<'_>, out: &mut Vec<Chunk>) -> Result<()> {
        let text = doc.text;
        if text.is_empty() {
            return Ok(());
        }
        let mut pieces = Vec::new();
        self.split(text, 0, text.len(), 0, &mut pieces);

        // Merge adjacent pieces up to `size`, then step back by `overlap`.
        let mut ordinal = 0u32;
        let mut i = 0usize;
        while i < pieces.len() {
            let start = pieces[i].0;
            let mut end = pieces[i].1;
            let mut j = i + 1;
            while j < pieces.len() && pieces[j].1 - start <= self.size {
                end = pieces[j].1;
                j += 1;
            }
            out.push(Chunk::leaf(
                doc.id,
                Span::new(start as u64, (end - start) as u32).offset_by(doc.base),
                ordinal,
            ));
            ordinal += 1;
            if j >= pieces.len() {
                break;
            }
            if self.overlap > 0 {
                let target = end.saturating_sub(self.overlap);
                let mut back = j;
                while back > i + 1 && pieces[back - 1].0 >= target {
                    back -= 1;
                }
                i = back;
            } else {
                i = j;
            }
        }
        Ok(())
    }
}
