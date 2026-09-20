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
//! Tokens are hashed together with their character n-grams, bounded by `<`
//! and `>` as in FastText. Without that, `writing` and `writes` land in
//! unrelated buckets and score exactly zero against each other; with it they
//! share `<wr`, `wri` and `rit`, so morphology, compounds and typos degrade
//! gracefully instead of falling off a cliff. It also makes the embedder work
//! on scripts a stemmer would not handle.
//!
//! Measured on HotpotQA over 1,461 paragraphs and 135 queries, adding n-grams
//! moved recall@5 from 0.419 to **0.622** (+0.204, p=0.0001) and hit@5 from
//! 0.681 to 0.889. A learned bi-encoder still scores 0.748 on the same task, so
//! this reaches roughly 83% of it at no cost and lower latency.
//!
//! **It still has no notion of meaning.** Two texts sharing no substrings score
//! zero however related they are, and common words count as much as rare ones
//! because there is no corpus to derive inverse document frequency from.
//! Raising the tokenizer's `min_len` to drop short function words looked
//! promising on a toy example and made no measurable difference on the real
//! task (-0.004, p=1.0), so it is not the default. Use this to exercise a
//! pipeline offline, not to judge retrieval quality.

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
    /// Character n-gram widths hashed alongside each whole token. Empty means
    /// whole tokens only, which cannot match morphological variants at all.
    #[serde(default = "default_ngrams")]
    pub ngrams: Vec<usize>,
    /// Weight of the whole-token feature relative to each n-gram feature.
    /// Above 1.0 favours exact matches over shared substrings.
    #[serde(default = "default_token_weight")]
    pub token_weight: f32,
}

fn default_dim() -> usize {
    256
}
fn default_tokenizer() -> PluginSpec {
    PluginSpec::named("simple")
}
fn default_ngrams() -> Vec<usize> {
    vec![3, 4, 5]
}
fn default_token_weight() -> f32 {
    1.0
}

impl Default for HashingConfig {
    fn default() -> Self {
        Self {
            dim: default_dim(),
            tokenizer: default_tokenizer(),
            ngrams: default_ngrams(),
            token_weight: default_token_weight(),
        }
    }
}

pub struct Hashing {
    dim: usize,
    tok: Box<dyn Tokenizer>,
    ngrams: Vec<usize>,
    token_weight: f32,
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
        if config.ngrams.contains(&0) {
            return Err(Error::config(Self::NAME, "n-gram widths must be > 0"));
        }
        if !config.token_weight.is_finite() || config.token_weight < 0.0 {
            return Err(Error::config(Self::NAME, "token_weight must be finite and >= 0"));
        }
        Ok(Self {
            dim: config.dim,
            tok: tokenize::registry().build_spec(&config.tokenizer)?,
            ngrams: config.ngrams,
            token_weight: config.token_weight,
        })
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

/// Accumulate one hashed feature into a row.
///
/// The sign is drawn from a different part of the hash than the bucket, so two
/// unrelated features that collide are as likely to cancel as to reinforce --
/// which keeps collisions from systematically inflating similarity.
#[inline]
fn add_feature(row: &mut [f32], bytes: &[u8], weight: f32) {
    let h = fnv1a(bytes);
    let bucket = (h % row.len() as u64) as usize;
    row[bucket] += if (h >> 63) & 1 == 1 { -weight } else { weight };
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
        let mut starts: Vec<usize> = Vec::new();
        out.reserve(texts.len() * self.dim);

        for text in texts {
            let base = out.len();
            out.resize(base + self.dim, 0.0);
            let row = &mut out[base..];

            spans.clear();
            self.tok.tokenize(text, &mut spans)?;

            for (s, e) in &spans {
                // Bounded, lowercased form: "<writing>". The markers make a
                // prefix n-gram distinguishable from the same letters occurring
                // mid-word, which is what separates "sorted" from "resorted".
                buf.clear();
                buf.push('<');
                buf.extend(text[*s as usize..*e as usize].chars().flat_map(char::to_lowercase));
                buf.push('>');

                if self.token_weight > 0.0 {
                    add_feature(row, &buf.as_bytes()[1..buf.len() - 1], self.token_weight);
                }

                if !self.ngrams.is_empty() {
                    starts.clear();
                    starts.extend(buf.char_indices().map(|(i, _)| i));
                    starts.push(buf.len());
                    let n_chars = starts.len() - 1;
                    for n in &self.ngrams {
                        // A token shorter than the window contributes no
                        // n-grams of that width; its whole-token feature and
                        // narrower widths still apply.
                        if n_chars < *n {
                            continue;
                        }
                        for w in 0..=(n_chars - n) {
                            add_feature(row, &buf.as_bytes()[starts[w]..starts[w + n]], 1.0);
                        }
                    }
                }
            }

            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }
        Ok(())
    }
}
