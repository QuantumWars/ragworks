//! Exact dense search over a contiguous matrix.
//!
//! No approximation: every vector is scored. That is the right default up to
//! a few hundred thousand vectors, it has no recall cliff to tune, and it gives
//! an exact baseline to measure an approximate index against later.
//!
//! Vectors live in one row-major `Vec<f32>` rather than a vector of vectors, so
//! the whole matrix is a single allocation with contiguous rows -- which is
//! what lets the scoring loop vectorise and what would let the buffer be
//! mmapped or handed to BLAS unchanged.

use ragworks_core::{Component, Error, Hit, Result, VectorStore};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Length-invariant. Safe when vectors are not normalised.
    #[default]
    Cosine,
    /// Raw inner product. Equals cosine when inputs are already unit length,
    /// and skips the division.
    Dot,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FlatConfig {
    /// Vector dimension. Fixed for the life of the index.
    pub dim: usize,
    #[serde(default)]
    pub metric: Metric,
}

impl Default for FlatConfig {
    fn default() -> Self {
        Self { dim: 0, metric: Metric::Cosine }
    }
}

#[derive(Debug)]
pub struct Flat {
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    data: Vec<f32>,
    norms: Vec<f32>,
}

impl Component for Flat {
    type Config = FlatConfig;
    const NAME: &'static str = "flat";
    const SUMMARY: &'static str = "Exact brute-force dense search over a contiguous matrix.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.dim == 0 {
            return Err(Error::config(Self::NAME, "dim must be > 0 and is required"));
        }
        Ok(Self {
            dim: config.dim,
            metric: config.metric,
            ids: Vec::new(),
            data: Vec::new(),
            norms: Vec::new(),
        })
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Straightforward and auto-vectorised by LLVM for f32 slices of equal,
    // statically unknown length. Explicit SIMD belongs behind a benchmark.
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Flat {
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl VectorStore for Flat {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn add(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        let expected = ids.len() * self.dim;
        if vectors.len() != expected {
            return Err(Error::DimensionMismatch { expected, actual: vectors.len() });
        }
        self.data.reserve(vectors.len());
        for (i, id) in ids.iter().enumerate() {
            let v = &vectors[i * self.dim..(i + 1) * self.dim];
            let n = dot(v, v).sqrt();
            // A zero vector has no direction, so cosine against it is undefined
            // rather than zero. Storing 1.0 makes it score 0 against everything
            // instead of producing NaN and poisoning the ranking.
            self.norms.push(if n > 0.0 { n } else { 1.0 });
            self.ids.push(*id);
            self.data.extend_from_slice(v);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, out: &mut Vec<Hit>) -> Result<()> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch { expected: self.dim, actual: query.len() });
        }
        if self.ids.is_empty() || k == 0 {
            return Ok(());
        }
        let qn = match self.metric {
            Metric::Cosine => {
                let n = dot(query, query).sqrt();
                if n > 0.0 { n } else { 1.0 }
            }
            Metric::Dot => 1.0,
        };

        let mut scored: Vec<(f32, u64)> = (0..self.ids.len())
            .map(|i| {
                let raw = dot(query, self.row(i));
                let s = match self.metric {
                    Metric::Cosine => raw / (self.norms[i] * qn),
                    Metric::Dot => raw,
                };
                (s, self.ids[i])
            })
            .collect();

        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
        }
        // Ties break on id so a run is reproducible regardless of scan order.
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        out.extend(scored.into_iter().map(|(score, id)| Hit { id, score }));
        Ok(())
    }
}
