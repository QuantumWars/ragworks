//! Zero-copy corpus store.
//!
//! The design decision that separates this from a Python RAG stack: a chunk is
//! a **view**, not a copy. agno's `ChunkingStrategy.chunk()` returns a list of
//! new `Document` objects, each owning a fresh copy of its text; chunking a
//! 1 GB corpus at 20% overlap allocates well over a gigabyte of duplicated
//! strings. Here a [`Chunk`] is 56 bytes -- a document id, two byte spans, and
//! its position in the tree -- and the text is borrowed from the corpus blob on
//! demand. The size is asserted by a test, not assumed: at 100M chunks every
//! 8 bytes is 800 MB, so the layout is a budget rather than a detail.
//!
//! The blob is a `String`, so UTF-8 validity is a type-level guarantee rather
//! than an invariant we maintain by hand. Spans that would split a character
//! are rejected by [`Corpus::text`] instead of panicking deep in a hot loop.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DocId(pub u64);

/// A byte range inside the corpus blob.
///
/// `start` is `u64` so a corpus may exceed 4 GB; `len` is `u32` because a
/// single chunk that large is a bug, not a use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: u64,
    pub len: u32,
}

impl Span {
    pub fn new(start: u64, len: u32) -> Self {
        Self { start, len }
    }
    pub fn end(&self) -> u64 {
        self.start + self.len as u64
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Shift a document-relative span into corpus-absolute coordinates.
    pub fn offset_by(self, base: u64) -> Self {
        Self { start: base + self.start, len: self.len }
    }
}

/// A retrieval unit: a view into one document, positioned in a tree.
///
/// `parent` indexes the slice of chunks this one was produced with, so a
/// structure-aware chunker can express a heading hierarchy without allocating
/// a graph. Flat chunkers simply leave it `None` -- flat is the degenerate
/// case of structured, which is why one type covers both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub doc: DocId,
    pub span: Span,
    pub ordinal: u32,
    pub parent: Option<u32>,
    pub depth: u16,
    /// Heading or caption, as a span -- not a copied `String`.
    ///
    /// A zero length means "no label". `Option<Span>` would be the obvious
    /// type, but `Span` has no niche, so the `Option` costs 8 bytes on every
    /// chunk for a bit of information. [`Chunk::label`] restores the ergonomics
    /// without the memory.
    pub label_span: Span,
}

impl Chunk {
    pub fn leaf(doc: DocId, span: Span, ordinal: u32) -> Self {
        Self { doc, span, ordinal, parent: None, depth: 0, label_span: Span::new(0, 0) }
    }

    /// The heading span, if this chunk has one.
    pub fn label(&self) -> Option<Span> {
        (self.label_span.len > 0).then_some(self.label_span)
    }

    /// Stable identifier, following agno's deterministic-id rule: the same
    /// input must produce the same id across runs, or caches and run
    /// comparisons silently break.
    pub fn id(&self, corpus: &Corpus) -> String {
        match corpus.doc(self.doc) {
            Some(d) => format!("{}#{}", d.uri, self.ordinal),
            None => format!("doc{}#{}", self.doc.0, self.ordinal),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Doc {
    pub id: DocId,
    pub uri: String,
    pub span: Span,
    pub meta: serde_json::Value,
}

/// Borrowed view of one document. What a [`crate::Chunker`] receives.
#[derive(Debug, Clone, Copy)]
pub struct DocView<'a> {
    pub id: DocId,
    pub uri: &'a str,
    pub text: &'a str,
    pub meta: &'a serde_json::Value,
    /// Absolute offset of `text` within the corpus blob. A chunker works in
    /// document-relative coordinates and calls [`Span::offset_by`] with this.
    pub base: u64,
}

#[derive(Debug, Default)]
pub struct Corpus {
    blob: String,
    docs: Vec<Doc>,
    by_uri: HashMap<String, DocId>,
}

impl Corpus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self { blob: String::with_capacity(bytes), ..Default::default() }
    }

    pub fn add(&mut self, uri: impl Into<String>, text: &str) -> DocId {
        self.add_with_meta(uri, text, serde_json::Value::Null)
    }

    pub fn add_with_meta(
        &mut self,
        uri: impl Into<String>,
        text: &str,
        meta: serde_json::Value,
    ) -> DocId {
        let uri = uri.into();
        let id = DocId(self.docs.len() as u64);
        let span = Span::new(self.blob.len() as u64, text.len() as u32);
        self.blob.push_str(text);
        self.by_uri.insert(uri.clone(), id);
        self.docs.push(Doc { id, uri, span, meta });
        id
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.blob.len()
    }
    pub fn doc(&self, id: DocId) -> Option<&Doc> {
        self.docs.get(id.0 as usize)
    }
    pub fn find(&self, uri: &str) -> Option<DocId> {
        self.by_uri.get(uri).copied()
    }
    pub fn docs(&self) -> impl Iterator<Item = DocView<'_>> {
        (0..self.docs.len()).filter_map(|i| self.view(DocId(i as u64)))
    }

    pub fn view(&self, id: DocId) -> Option<DocView<'_>> {
        let d = self.doc(id)?;
        Some(DocView {
            id,
            uri: &d.uri,
            text: self.text(d.span).ok()?,
            meta: &d.meta,
            base: d.span.start,
        })
    }

    /// Borrow text for a span. Rejects spans that would split a character
    /// rather than panicking, so a buggy chunker fails with a useful message.
    pub fn text(&self, span: Span) -> Result<&str> {
        let (s, e) = (span.start as usize, span.end() as usize);
        if e > self.blob.len() || !self.blob.is_char_boundary(s) || !self.blob.is_char_boundary(e) {
            return Err(Error::UnalignedSpan { doc: 0, start: span.start, end: span.end() });
        }
        Ok(&self.blob[s..e])
    }

    pub fn chunk_text(&self, chunk: &Chunk) -> Result<&str> {
        self.text(chunk.span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_views_not_copies() {
        // A chunk carries no text of its own, and its size is a budget.
        // Going below 56 would mean u32 document ids (capping a corpus at 4B
        // documents) and sentinel-encoded parents -- not worth the ergonomics
        // until something actually needs it.
        assert_eq!(
            std::mem::size_of::<Chunk>(),
            56,
            "Chunk layout changed; update the budget deliberately, not by accident"
        );
    }

    #[test]
    fn an_absent_label_costs_nothing_and_reads_as_none() {
        let k = Chunk::leaf(DocId(0), Span::new(0, 4), 0);
        assert_eq!(k.label(), None);
        let labelled = Chunk { label_span: Span::new(2, 3), ..k.clone() };
        assert_eq!(labelled.label(), Some(Span::new(2, 3)));
    }

    #[test]
    fn text_round_trips_through_spans() {
        let mut c = Corpus::new();
        let a = c.add("a.txt", "hello world");
        let b = c.add("b.txt", "second doc");
        assert_eq!(c.view(a).unwrap().text, "hello world");
        assert_eq!(c.view(b).unwrap().text, "second doc");
        assert_eq!(c.len(), 2);
        assert_eq!(c.find("b.txt"), Some(b));
    }

    #[test]
    fn document_relative_spans_offset_into_the_blob() {
        let mut c = Corpus::new();
        c.add("pad", "XXXX");
        let d = c.add("doc", "alpha beta");
        let v = c.view(d).unwrap();
        let rel = Span::new(6, 4); // "beta" within the document
        assert_eq!(c.text(rel.offset_by(v.base)).unwrap(), "beta");
    }

    #[test]
    fn a_span_splitting_a_character_is_an_error_not_a_panic() {
        let mut c = Corpus::new();
        let d = c.add("u.txt", "héllo"); // 'é' occupies two bytes
        let v = c.view(d).unwrap();
        let bad = Span::new(v.base + 1, 1);
        assert!(matches!(c.text(bad), Err(Error::UnalignedSpan { .. })));
    }

    #[test]
    fn chunk_ids_are_deterministic() {
        let mut c = Corpus::new();
        let d = c.add("book.md", "text");
        let k = Chunk::leaf(d, Span::new(0, 4), 3);
        assert_eq!(k.id(&c), "book.md#3");
        assert_eq!(k.id(&c), Chunk::leaf(d, Span::new(0, 4), 3).id(&c));
    }
}
