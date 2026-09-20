//! Deterministic hashing embedder -- the signed hashing trick.
//!
//! No network, no model, no download. It exists so a pipeline can be tested,
//! benchmarked and demonstrated offline, and so a change to chunking or
//! indexing can be measured without an embedding bill confounding the result.
//!
//! It is a real (weak) embedding, not a stub: lexical overlap produces genuine
//! cosine similarity, which makes it a usable floor to measure a learned
//! embedder against.
//!
//! **It has no notion of meaning, and on paraphrases it can rank backwards.**
//! Measured on three sentences, a learned embedder scored the paraphrase pair
//! 0.51 and the unrelated pair 0.07; this hasher scored them 0.13 and 0.25 --
//! the wrong way round, because the unrelated sentences happened to share more
//! function words. Use it to exercise a pipeline, never to judge retrieval
//! quality.

use ragworks_core::{Component, Embedder, Error, PluginSpec, Result, Tokenizer, tokenize};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HashingConfig {
    #[serde(default = "default_dim")]
    pub dim: usize,
    #[serde(default = "default_tokenizer")]
    pub tokenizer: PluginSpec,
}

fn default_dim() -> usize {
    256
}
fn default_tokenizer() -> PluginSpec {
    PluginSpec::named("simple")
}

impl Default for HashingConfig {
    fn default() -> Self {
        Self { dim: default_dim(), tokenizer: default_tokenizer() }
    }
}

pub struct Hashing {
    dim: usize,
    tok: Box<dyn Tokenizer>,
}

impl std::fmt::Debug for Hashing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hashing").field("dim", &self.dim).finish()
    }
}

impl Component for Hashing {
    type Config = HashingConfig;
    const NAME: &'static str = "hashing";
    const SUMMARY: &'static str = "Deterministic signed hashing embedder; offline, no model.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.dim == 0 {
            return Err(Error::config(Self::NAME, "dim must be > 0"));
        }
        Ok(Self { dim: config.dim, tok: tokenize::registry().build_spec(&config.tokenizer)? })
    }
}

/// FNV-1a. Chosen for being stable across runs and platforms -- `DefaultHasher`
/// is explicitly not, and an embedding that changes between processes would
/// silently invalidate every cached vector.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Embedder for Hashing {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_batch(&self) -> usize {
        usize::MAX // no provider, so no batch limit
    }

    fn embed(&self, texts: &[&str], out: &mut Vec<f32>) -> Result<()> {
        let mut spans = Vec::new();
        let mut buf = String::new();
        out.reserve(texts.len() * self.dim);

        for text in texts {
            let base = out.len();
            out.resize(base + self.dim, 0.0);
            spans.clear();
            self.tok.tokenize(text, &mut spans)?;

            for (s, e) in &spans {
                buf.clear();
                buf.extend(text[*s as usize..*e as usize].chars().flat_map(char::to_lowercase));
                let h = fnv1a(buf.as_bytes());
                let bucket = (h % self.dim as u64) as usize;
                // A sign bit drawn from a different part of the hash keeps
                // unrelated collisions from always adding constructively.
                let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
                out[base + bucket] += sign;
            }

            let norm = out[base..].iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut out[base..] {
                    *v /= norm;
                }
            }
        }
        Ok(())
    }
}
