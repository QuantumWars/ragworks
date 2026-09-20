//! Python bindings.
//!
//! The boundary follows one rule: **nothing crosses per item.** Chunking,
//! indexing and search all happen entirely in Rust, and Python hands over or
//! receives whole batches. `search_many` exists precisely so that scoring a
//! thousand queries is one crossing rather than a thousand.
//!
//! Text is copied only where Python genuinely needs a `str`. Internally a chunk
//! stays a view into the corpus blob, so chunking a large corpus never
//! materialises the text twice.

use pyo3::create_exception;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::prelude::*;

create_exception!(
    ragworks,
    RetryableError,
    PyRuntimeError,
    "A failure that could plausibly succeed if retried -- a rate limit, a timeout, a 5xx."
);

/// Newtype so the orphan rule lets us convert into `PyErr`.
struct Error(ragworks_core::Error);

impl From<ragworks_core::Error> for Error {
    fn from(e: ragworks_core::Error) -> Self {
        Self(e)
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        // The retry classification survives into Python as a distinct
        // exception type, so callers can `except RetryableError` instead of
        // matching on message text.
        if e.0.is_retryable() {
            RetryableError::new_err(e.0.to_string())
        } else {
            PyValueError::new_err(e.0.to_string())
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

fn json(obj: Option<&Bound<'_, PyAny>>) -> PyResult<serde_json::Value> {
    let Some(obj) = obj else {
        return Ok(serde_json::Value::Null);
    };
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    // Round-trip through the json module: it handles dict, list and scalars
    // uniformly, and config objects are tiny so the cost is irrelevant.
    let py = obj.py();
    let dumps = py.import("json")?.getattr("dumps")?;
    let s: String = dumps.call1((obj,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("invalid config: {e}")))
}

// ---------------------------------------------------------------- corpus

#[pyclass(module = "ragworks")]
pub struct Corpus {
    pub(crate) inner: ragworks_core::Corpus,
}

#[pymethods]
impl Corpus {
    #[new]
    fn new() -> Self {
        Self { inner: ragworks_core::Corpus::new() }
    }

    /// Add a document; returns its id.
    fn add(&mut self, uri: &str, text: &str) -> u64 {
        self.inner.add(uri, text).0
    }

    /// Add many documents in one crossing.
    fn add_many(&mut self, uris: Vec<String>, texts: Vec<String>) -> PyResult<Vec<u64>> {
        if uris.len() != texts.len() {
            return Err(PyValueError::new_err(format!(
                "uris ({}) and texts ({}) must be the same length",
                uris.len(),
                texts.len()
            )));
        }
        Ok(uris
            .iter()
            .zip(&texts)
            .map(|(u, t)| self.inner.add(u.as_str(), t.as_str()).0)
            .collect())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.bytes()
    }

    fn text(&self, doc_id: u64) -> PyResult<String> {
        let d = self
            .inner
            .view(ragworks_core::DocId(doc_id))
            .ok_or_else(|| PyValueError::new_err(format!("no document {doc_id}")))?;
        Ok(d.text.to_string())
    }

    fn __repr__(&self) -> String {
        format!("Corpus(docs={}, bytes={})", self.inner.len(), self.inner.bytes())
    }
}

// ---------------------------------------------------------------- chunks

#[pyclass(module = "ragworks", get_all, from_py_object)]
#[derive(Clone)]
pub struct Chunk {
    pub id: String,
    pub doc: u64,
    pub ordinal: u32,
    pub depth: u16,
    pub parent: Option<u32>,
    pub label: Option<String>,
    pub text: String,
}

#[pymethods]
impl Chunk {
    fn __repr__(&self) -> String {
        format!(
            "Chunk(id={:?}, depth={}, parent={:?}, label={:?}, len={})",
            self.id,
            self.depth,
            self.parent,
            self.label,
            self.text.len()
        )
    }
}

#[pyclass(module = "ragworks")]
pub struct Chunker {
    inner: Box<dyn ragworks_core::Chunker>,
}

#[pymethods]
impl Chunker {
    #[new]
    #[pyo3(signature = (name, config = None))]
    fn new(name: &str, config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let cfg = json(config)?;
        Ok(Self { inner: ragworks_chunk::registry().build(name, &cfg).map_err(Error)? })
    }

    /// Chunk an entire corpus. All splitting happens in Rust; only the
    /// resulting text crosses, once.
    fn chunk(&self, corpus: &Corpus) -> Result<Vec<Chunk>> {
        let c = &corpus.inner;
        let mut raw = Vec::new();
        for doc in c.docs() {
            self.inner.chunk(doc, &mut raw)?;
        }
        let mut out = Vec::with_capacity(raw.len());
        for k in &raw {
            out.push(Chunk {
                id: k.id(c),
                doc: k.doc.0,
                ordinal: k.ordinal,
                depth: k.depth,
                parent: k.parent,
                label: match k.label() {
                    Some(s) => Some(c.text(s)?.to_string()),
                    None => None,
                },
                text: c.chunk_text(k)?.to_string(),
            });
        }
        Ok(out)
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("Chunker({:?})", self.inner.name())
    }
}

// ---------------------------------------------------------------- bm25

#[pyclass(module = "ragworks")]
pub struct Bm25 {
    inner: Box<dyn ragworks_core::TextIndex>,
}

#[pymethods]
impl Bm25 {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let cfg = json(config)?;
        Ok(Self { inner: ragworks_index::text_registry().build("bm25", &cfg).map_err(Error)? })
    }

    /// Index a batch. One crossing for the whole corpus.
    fn add_many(&mut self, ids: Vec<u64>, texts: Vec<String>) -> PyResult<()> {
        if ids.len() != texts.len() {
            return Err(PyValueError::new_err(format!(
                "ids ({}) and texts ({}) must be the same length",
                ids.len(),
                texts.len()
            )));
        }
        for (id, t) in ids.iter().zip(&texts) {
            self.inner.add(*id, t).map_err(Error)?;
        }
        self.inner.finish().map_err(Error)?;
        Ok(())
    }

    /// Index chunks produced by a [`Chunker`], keyed by position.
    fn add_chunks(&mut self, chunks: Vec<Chunk>) -> Result<()> {
        for (i, c) in chunks.iter().enumerate() {
            self.inner.add(i as u64, &c.text)?;
        }
        self.inner.finish()?;
        Ok(())
    }

    /// Top `k` as `[(id, score), ...]`.
    #[pyo3(signature = (query, k = 10))]
    fn search(&self, query: &str, k: usize) -> Result<Vec<(u64, f32)>> {
        let mut hits = Vec::with_capacity(k);
        self.inner.search(query, k, &mut hits)?;
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }

    /// Score many queries in **one** crossing. This is the whole point of the
    /// FFI design: a thousand queries should cost one boundary transition, not
    /// a thousand.
    #[pyo3(signature = (queries, k = 10))]
    fn search_many(&self, py: Python<'_>, queries: Vec<String>, k: usize) -> Result<Vec<Vec<(u64, f32)>>> {
        // The GIL is released for the whole batch, so a caller can run this
        // alongside Python threads doing other work. (PyO3 0.29 renamed
        // `allow_threads` to `detach`.)
        py.detach(|| {
            let mut out = Vec::with_capacity(queries.len());
            let mut hits = Vec::with_capacity(k);
            for q in &queries {
                hits.clear();
                self.inner.search(q, k, &mut hits)?;
                out.push(hits.iter().map(|h| (h.id, h.score)).collect());
            }
            Ok(out)
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Bm25(docs={})", self.inner.len())
    }
}

// ------------------------------------------------------------- embedder

#[pyclass(module = "ragworks")]
pub struct Embedder {
    inner: Box<dyn ragworks_core::Embedder>,
}

#[pymethods]
impl Embedder {
    /// `name` is "hashing" (offline, deterministic) or "openai" (any
    /// OpenAI-compatible `/embeddings` endpoint).
    #[new]
    #[pyo3(signature = (name, config = None))]
    fn new(name: &str, config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let cfg = json(config)?;
        Ok(Self { inner: ragworks_embed::registry().build(name, &cfg).map_err(Error)? })
    }

    /// Embed a batch. Returns `dim * len(texts)` floats, row-major.
    ///
    /// Batching, retry and rate limiting happen inside Rust, so a long run is
    /// one crossing per provider batch rather than one per text.
    fn embed(&self, py: Python<'_>, texts: Vec<String>) -> Result<Vec<f32>> {
        py.detach(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let mut out = Vec::with_capacity(refs.len() * self.inner.dim());
            self.inner.embed(&refs, &mut out)?;
            Ok(out)
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn max_batch(&self) -> usize {
        self.inner.max_batch()
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("Embedder({:?}, dim={})", self.inner.name(), self.inner.dim())
    }
}

// ---------------------------------------------------------------- dense

#[pyclass(module = "ragworks")]
pub struct Flat {
    inner: Box<dyn ragworks_core::VectorStore>,
}

#[pymethods]
impl Flat {
    #[new]
    #[pyo3(signature = (dim, metric = "cosine"))]
    fn new(dim: usize, metric: &str) -> PyResult<Self> {
        let cfg = serde_json::json!({"dim": dim, "metric": metric});
        Ok(Self { inner: ragworks_index::vector_registry().build("flat", &cfg).map_err(Error)? })
    }

    /// `vectors` is a flat row-major sequence of `len(ids) * dim` floats.
    fn add(&mut self, ids: Vec<u64>, vectors: Vec<f32>) -> Result<()> {
        self.inner.add(&ids, &vectors)?;
        Ok(())
    }

    #[pyo3(signature = (query, k = 10))]
    fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<(u64, f32)>> {
        let mut hits = Vec::with_capacity(k);
        self.inner.search(&query, k, &mut hits)?;
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __repr__(&self) -> String {
        format!("Flat(n={}, dim={})", self.inner.len(), self.inner.dim())
    }
}

// --------------------------------------------------------------- readers

/// Read one file, dispatching on its extension. Returns `{"text", "meta"}`.
#[pyfunction]
fn read_file(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let readers = ragworks_read::Readers::standard().map_err(Error)?;
    let x = readers.read_path(path).map_err(Error)?;
    let out = serde_json::json!({"text": x.text, "meta": x.meta});
    let loads = py.import("json")?.getattr("loads")?;
    Ok(loads.call1((out.to_string(),))?.unbind())
}

/// Walk `dir` and add every readable file to `corpus`.
///
/// Files whose extension has no reader are sniffed and skipped if they look
/// binary, so compiled artefacts do not enter the index as mojibake.
#[pyfunction]
#[pyo3(signature = (corpus, dir, max_bytes = 64 << 20, skip_hidden = true))]
fn ingest(
    py: Python<'_>,
    corpus: &mut Corpus,
    dir: &str,
    max_bytes: usize,
    skip_hidden: bool,
) -> PyResult<Py<PyAny>> {
    let readers = ragworks_read::Readers::standard().map_err(Error)?;
    let opts = ragworks_read::IngestOptions {
        max_bytes,
        skip_hidden,
        ..Default::default()
    };
    let stats = ragworks_read::ingest_dir(&mut corpus.inner, dir, &readers, &opts).map_err(Error)?;
    let out = serde_json::json!({
        "files": stats.files,
        "skipped": stats.skipped,
        "bytes": stats.bytes,
        "errors": stats.errors.iter()
            .map(|(p, e)| serde_json::json!([p.to_string_lossy(), e]))
            .collect::<Vec<_>>(),
    });
    let loads = py.import("json")?.getattr("loads")?;
    Ok(loads.call1((out.to_string(),))?.unbind())
}

// ---------------------------------------------------------------- module

/// Reciprocal Rank Fusion over ranked runs of `(id, score)` pairs.
#[pyfunction]
#[pyo3(signature = (runs, k = 60.0, top = 10))]
fn rrf(runs: Vec<Vec<(u64, f32)>>, k: f32, top: usize) -> Vec<(u64, f32)> {
    let owned: Vec<Vec<ragworks_core::Hit>> = runs
        .into_iter()
        .map(|r| r.into_iter().map(|(id, score)| ragworks_core::Hit { id, score }).collect())
        .collect();
    let refs: Vec<&[ragworks_core::Hit]> = owned.iter().map(|r| r.as_slice()).collect();
    let mut out = Vec::with_capacity(top);
    ragworks_index::rrf(&refs, k, top, &mut out);
    out.into_iter().map(|h| (h.id, h.score)).collect()
}

/// Every registered plugin, with its JSON Schema. The same source of truth that
/// validates configs also documents them.
#[pyfunction]
fn catalogue(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let all = serde_json::json!({
        "chunker": ragworks_chunk::registry().describe(),
        "tokenizer": ragworks_core::tokenize::registry().describe(),
        "text_index": ragworks_index::text_registry().describe(),
        "vector_store": ragworks_index::vector_registry().describe(),
        "embedder": ragworks_embed::registry().describe(),
        "reader": ragworks_read::registry().describe(),
    });
    let loads = py.import("json")?.getattr("loads")?;
    Ok(loads.call1((all.to_string(),))?.unbind())
}

#[pymodule]
fn ragworks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust core for RAG infrastructure.")?;
    m.add("RetryableError", m.py().get_type::<RetryableError>())?;
    m.add_class::<Corpus>()?;
    m.add_class::<Chunk>()?;
    m.add_class::<Chunker>()?;
    m.add_class::<Bm25>()?;
    m.add_class::<Embedder>()?;
    m.add_class::<Flat>()?;
    m.add_function(wrap_pyfunction!(rrf, m)?)?;
    m.add_function(wrap_pyfunction!(read_file, m)?)?;
    m.add_function(wrap_pyfunction!(ingest, m)?)?;
    m.add_function(wrap_pyfunction!(catalogue, m)?)?;
    Ok(())
}
