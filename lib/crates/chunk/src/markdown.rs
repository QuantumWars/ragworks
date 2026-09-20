//! Heading-aware chunking: the structure-preserving strategy.
//!
//! A document's heading tree is a table of contents, and its sections are
//! human-authored retrieval units -- coherent and non-overlapping in a way no
//! byte window is. This is the mechanism behind structure-aware retrieval, and
//! it is why [`Chunk`] carries `parent` and `depth`: flat chunking is the
//! degenerate case of this one, not a different thing.
//!
//! Emits every section by default, so one index holds several granularities at
//! once -- a broad parent section and its precise leaves. Set `leaves_only` for
//! non-overlapping leaf retrieval.

use ragworks_core::{Chunk, Chunker, Component, DocView, Error, Result, Span};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MarkdownConfig {
    /// Emit only sections that have no subsections.
    #[serde(default)]
    pub leaves_only: bool,
    /// Headings deeper than this do not open a new section.
    #[serde(default = "default_max_level")]
    pub max_level: u8,
    /// Drop sections whose body is shorter than this, in bytes.
    #[serde(default)]
    pub min_bytes: usize,
}

fn default_max_level() -> u8 {
    6
}

impl Default for MarkdownConfig {
    fn default() -> Self {
        Self { leaves_only: false, max_level: default_max_level(), min_bytes: 0 }
    }
}

#[derive(Debug)]
pub struct Markdown {
    leaves_only: bool,
    max_level: u8,
    min_bytes: usize,
}

impl Component for Markdown {
    type Config = MarkdownConfig;
    const NAME: &'static str = "markdown";
    const SUMMARY: &'static str = "Heading-aware sections with parent links, preserving document structure.";

    fn build(config: Self::Config) -> Result<Self> {
        if !(1..=6).contains(&config.max_level) {
            return Err(Error::config(Self::NAME, "max_level must be between 1 and 6"));
        }
        Ok(Self {
            leaves_only: config.leaves_only,
            max_level: config.max_level,
            min_bytes: config.min_bytes,
        })
    }
}

struct Heading {
    level: u8,
    start: usize,
    title: (usize, usize),
}

/// Parse an ATX heading (`## Title`). Returns level and the title's byte range.
fn parse_heading(line: &str, line_start: usize) -> Option<(u8, (usize, usize))> {
    let hashes = line.bytes().take_while(|b| *b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &line[hashes..];
    if !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None; // "#hashtag" is not a heading
    }
    let trimmed = rest.trim_start();
    let off = hashes + (rest.len() - trimmed.len());
    let title = trimmed.trim_end();
    Some((hashes as u8, (line_start + off, line_start + off + title.len())))
}

impl Chunker for Markdown {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn chunk(&self, doc: DocView<'_>, out: &mut Vec<Chunk>) -> Result<()> {
        let text = doc.text;
        if text.is_empty() {
            return Ok(());
        }

        let mut heads: Vec<Heading> = Vec::new();
        let mut pos = 0usize;
        for line in text.split_inclusive('\n') {
            if let Some((level, title)) = parse_heading(line, pos)
                && level <= self.max_level
            {
                heads.push(Heading { level, start: pos, title });
            }
            pos += line.len();
        }

        let base = out.len() as u32;
        let mut ordinal = 0u32;

        // Text before the first heading is a section in its own right.
        let preamble_end = heads.first().map_or(text.len(), |h| h.start);
        let mut emitted: Vec<Option<u32>> = Vec::with_capacity(heads.len());
        if preamble_end > 0 && text[..preamble_end].trim().len() >= self.min_bytes {
            out.push(Chunk {
                doc: doc.id,
                span: Span::new(0, preamble_end as u32).offset_by(doc.base),
                ordinal,
                parent: None,
                depth: 0,
                label_span: Span::new(0, 0),
            });
            ordinal += 1;
        }

        for (i, h) in heads.iter().enumerate() {
            // A section ends where the next heading of the same or shallower
            // level begins; anything deeper is nested inside it.
            let end = heads[i + 1..]
                .iter()
                .find(|n| n.level <= h.level)
                .map_or(text.len(), |n| n.start);

            let has_child = heads.get(i + 1).is_some_and(|n| n.level > h.level);
            let body_len = text[h.start..end].trim().len();

            if (self.leaves_only && has_child) || body_len < self.min_bytes {
                emitted.push(None);
                continue;
            }

            // Nearest enclosing section that was actually emitted.
            let parent = heads[..i]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, p)| p.level < h.level)
                .and_then(|(k, _)| emitted.get(k).copied().flatten());

            emitted.push(Some(base + ordinal));
            out.push(Chunk {
                doc: doc.id,
                span: Span::new(h.start as u64, (end - h.start) as u32).offset_by(doc.base),
                ordinal,
                parent,
                depth: h.level as u16,
                label_span: Span::new(h.title.0 as u64, (h.title.1 - h.title.0) as u32)
                    .offset_by(doc.base),
            });
            ordinal += 1;
        }
        Ok(())
    }
}
