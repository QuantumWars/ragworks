//! The swap points.
//!
//! One trait per stage a developer replaces. Two conventions run through all of
//! them, and both exist to avoid allocation patterns that Python libraries
//! cannot avoid:
//!
//! * **Output buffers are passed in**, not returned. Indexing a million
//!   documents reuses one `Vec` instead of allocating a fresh one per document.
//! * **Batches are contiguous.** An embedder writes a row-major `Vec<f32>`,
//!   not a `Vec<Vec<f32>>`, so a batch is one allocation and can be handed to
//!   BLAS, mmapped, or sliced without a gather.

use crate::corpus::{Chunk, DocView};
use crate::error::Result;

/// Split a document into retrieval units.
///
/// Implementations work in **document-relative** byte offsets and call
/// [`crate::Span::offset_by`] with `doc.base` before emitting, so a chunker
/// never needs to know where its document sits in the corpus.
pub trait Chunker: Send + Sync {
    fn name(&self) -> &'static str;

    /// Append this document's chunks to `out`. `out` is not cleared.
    fn chunk(&self, doc: DocView<'_>, out: &mut Vec<Chunk>) -> Result<()>;

    /// Convenience for one-off use and tests.
    fn chunk_one(&self, doc: DocView<'_>) -> Result<Vec<Chunk>> {
        let mut out = Vec::new();
        self.chunk(doc, &mut out)?;
        Ok(out)
    }
}

/// Text extracted from a source file, plus whatever the format revealed.
#[derive(Debug, Clone, Default)]
pub struct Extracted {
    pub text: String,
    pub meta: serde_json::Value,
}

/// Turn bytes of some format into text. The one place copying is unavoidable.
pub trait Reader: Send + Sync {
    fn name(&self) -> &'static str;
    /// Lowercase extensions, no dot. Used for format dispatch.
    fn extensions(&self) -> &'static [&'static str];
    fn read(&self, bytes: &[u8], uri: &str) -> Result<Extracted>;
}

/// Map text to vectors.
pub trait Embedder: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;

    /// Append `texts.len() * dim()` floats to `out`, row-major.
    fn embed(&self, texts: &[&str], out: &mut Vec<f32>) -> Result<()>;

    /// Largest batch worth sending at once. The runner uses this to chunk work
    /// rather than each caller guessing a provider's limit.
    fn max_batch(&self) -> usize {
        64
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.dim());
        self.embed(&[text], &mut out)?;
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}

/// Retrieve by text -- lexical indexes such as BM25.
///
/// Separate from [`VectorStore`] on purpose: the two take genuinely different
/// queries (a string versus a vector), and collapsing them behind one enum
/// would buy nothing but a match arm at every call site. A hybrid retriever
/// composes the two rather than unifying them.
pub trait TextIndex: Send + Sync {
    fn name(&self) -> &'static str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn add(&mut self, id: u64, text: &str) -> Result<()>;

    /// Finalise after bulk loading. Collection statistics such as IDF are not
    /// knowable until every document is in, so scoring before this is called
    /// would be wrong rather than merely stale.
    fn finish(&mut self) -> Result<()>;

    /// Append the top `k` hits to `out`, best first.
    fn search(&self, query: &str, k: usize, out: &mut Vec<Hit>) -> Result<()>;
}

/// Store vectors and search them. Local or a remote service behind the trait.
pub trait VectorStore: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `vectors` is row-major and must be `ids.len() * dim()` long.
    fn add(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()>;

    /// Append the top `k` hits to `out`, best first.
    fn search(&self, query: &[f32], k: usize, out: &mut Vec<Hit>) -> Result<()>;
}

/// Reorder candidates for a query.
///
/// Wave-1 measurement: reranking moved recall@5 from 0.748 to 0.822
/// (p=0.0001) -- but it can only reorder the shortlist it is given. A gold
/// document at rank 88 is unreachable by any reranker over a top-8 window,
/// which is why `depth` belongs to the caller, not to the reranker.
pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;

    /// Append one score per candidate to `out`, in candidate order.
    fn rerank(&self, query: &str, candidates: &[&str], out: &mut Vec<f32>) -> Result<()>;
}

/// Judge whether a set of evidence supports an answer.
///
/// Set-level by construction: the whole candidate set arrives in one call,
/// because sufficiency is not decidable per passage. Wave 1 confirmed this on
/// our own data -- a missing second hop is invisible to any per-passage scorer.
pub trait Verifier: Send + Sync {
    fn name(&self) -> &'static str;
    fn verify(&self, query: &str, evidence: &[&str]) -> Result<Verdict>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    Supports,
    Refutes,
    Insufficient,
}

#[derive(Debug, Clone)]
pub struct Verdict {
    pub support: Support,
    pub confidence: f32,
    /// How many evidence items were judged. A verdict without this is
    /// uninterpretable -- "insufficient over 4" and "insufficient over 8" are
    /// different claims. (Wave-1 finding F4.)
    pub window: usize,
}

impl Verdict {
    pub fn should_answer(&self, min_confidence: f32) -> bool {
        self.support == Support::Supports && self.confidence >= min_confidence
    }
}
