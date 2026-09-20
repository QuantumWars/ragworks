//! Core types, the zero-copy corpus store, and the plugin system.
//!
//! Everything a developer swaps -- chunker, reader, embedder, vector store,
//! reranker, verifier -- is a trait here with a [`Registry`] in front of it.
//! Implementations live in sibling crates so that adding a backend never
//! touches this one.

pub mod corpus;
pub mod error;
pub mod plugin;
pub mod tokenize;
pub mod traits;

pub use corpus::{Chunk, Corpus, Doc, DocId, DocView, Span};
pub use error::{Error, ProviderFault, Result};
pub use plugin::{Component, PluginSpec, Registry};
pub use tokenize::{TokenSpan, Tokenizer};
pub use traits::{
    Chunker, Embedder, Extracted, Hit, Reader, Reranker, Support, TextIndex, VectorStore, Verdict,
    Verifier,
};
