//! Offline baselines: term-overlap reranking, and term-coverage verification.
//!
//! These exist to answer a question a model-backed judge cannot answer about
//! itself: how much of its benefit is understanding, and how much is term
//! overlap? A reranker that beats no baseline has not been shown to do
//! anything, and the same evaluation discipline applies here as everywhere
//! else in this project.

use std::collections::{HashMap, HashSet};

use ragworks_core::{
    Component, Error, PluginSpec, Reranker, Result, Support, Tokenizer, Verdict, Verifier,
    tokenize,
};
use schemars::JsonSchema;
use serde::Deserialize;

fn default_tokenizer() -> PluginSpec {
    PluginSpec::named("simple")
}

fn terms(tok: &dyn Tokenizer, text: &str) -> Result<Vec<String>> {
    let mut spans = Vec::new();
    tok.tokenize(text, &mut spans)?;
    Ok(spans
        .iter()
        .map(|(s, e)| text[*s as usize..*e as usize].to_lowercase())
        .collect())
}

// ---------------------------------------------------------------- reranker

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LexicalConfig {
    #[serde(default = "default_tokenizer")]
    pub tokenizer: PluginSpec,
    #[serde(default = "default_k1")]
    pub k1: f32,
    #[serde(default = "default_b")]
    pub b: f32,
}

fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}

impl Default for LexicalConfig {
    fn default() -> Self {
        Self { tokenizer: default_tokenizer(), k1: default_k1(), b: default_b() }
    }
}

pub struct Lexical {
    tok: Box<dyn Tokenizer>,
    k1: f32,
    b: f32,
}

impl std::fmt::Debug for Lexical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lexical").finish()
    }
}

impl Component for Lexical {
    type Config = LexicalConfig;
    const NAME: &'static str = "lexical";
    const SUMMARY: &'static str = "BM25 rescoring of a shortlist; offline, no model.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.k1.is_nan() || config.k1 < 0.0 {
            return Err(Error::config(Self::NAME, "k1 must be a number >= 0"));
        }
        if config.b.is_nan() || !(0.0..=1.0).contains(&config.b) {
            return Err(Error::config(Self::NAME, "b must be a number between 0 and 1"));
        }
        Ok(Self { tok: tokenize::registry().build_spec(&config.tokenizer)?, k1: config.k1, b: config.b })
    }
}

impl Reranker for Lexical {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// BM25 with the shortlist as its own collection.
    ///
    /// IDF is computed over the candidates rather than the full corpus, which
    /// is the point: within a shortlist every document already matched, so what
    /// discriminates is which terms are *rare among the survivors*.
    fn rerank(&self, query: &str, candidates: &[&str], out: &mut Vec<f32>) -> Result<()> {
        if candidates.is_empty() {
            return Ok(());
        }
        let docs: Vec<Vec<String>> =
            candidates.iter().map(|c| terms(&*self.tok, c)).collect::<Result<_>>()?;
        let n = docs.len() as f32;
        let avg_len =
            (docs.iter().map(|d| d.len()).sum::<usize>() as f32 / n).max(1.0);

        let mut df: HashMap<&str, f32> = HashMap::new();
        for d in &docs {
            for t in d.iter().collect::<HashSet<_>>() {
                *df.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
        }

        let q = terms(&*self.tok, query)?;
        for d in &docs {
            let mut tf: HashMap<&str, f32> = HashMap::new();
            for t in d {
                *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
            let norm = 1.0 - self.b + self.b * (d.len() as f32 / avg_len);
            let mut score = 0.0;
            for term in &q {
                let f = tf.get(term.as_str()).copied().unwrap_or(0.0);
                if f == 0.0 {
                    continue;
                }
                let dfv = df.get(term.as_str()).copied().unwrap_or(0.0);
                let idf = (1.0 + (n - dfv + 0.5) / (dfv + 0.5)).ln();
                score += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * norm);
            }
            out.push(score);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- verifier

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CoverageConfig {
    #[serde(default = "default_tokenizer")]
    pub tokenizer: PluginSpec,
    /// Fraction of the query's terms that must appear somewhere in the evidence
    /// for the verdict to be `Supports`.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Terms shorter than this are ignored, which removes most English function
    /// words without a language-specific stop list.
    #[serde(default = "default_min_len")]
    pub min_term_len: usize,
}

fn default_threshold() -> f32 {
    0.6
}
fn default_min_len() -> usize {
    3
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            tokenizer: default_tokenizer(),
            threshold: default_threshold(),
            min_term_len: default_min_len(),
        }
    }
}

pub struct Coverage {
    tok: Box<dyn Tokenizer>,
    threshold: f32,
    min_term_len: usize,
}

impl std::fmt::Debug for Coverage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coverage").field("threshold", &self.threshold).finish()
    }
}

impl Component for Coverage {
    type Config = CoverageConfig;
    const NAME: &'static str = "coverage";
    const SUMMARY: &'static str =
        "Offline sufficiency by query-term coverage; never returns Refutes.";

    fn build(config: Self::Config) -> Result<Self> {
        if !(0.0..=1.0).contains(&config.threshold) {
            return Err(Error::config(Self::NAME, "threshold must be between 0 and 1"));
        }
        Ok(Self {
            tok: tokenize::registry().build_spec(&config.tokenizer)?,
            threshold: config.threshold,
            min_term_len: config.min_term_len,
        })
    }
}

impl Verifier for Coverage {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    /// Fraction of the query's content terms present anywhere in the evidence.
    ///
    /// This encodes the missing-hop intuition cheaply: a question naming an
    /// entity that appears in none of the retrieved passages cannot have been
    /// answered from them. It **never returns `Refutes`** -- detecting
    /// contradiction requires understanding the text, and claiming otherwise
    /// would be dishonest about what term matching can do.
    fn verify(&self, query: &str, evidence: &[&str]) -> Result<Verdict> {
        let q: HashSet<String> = terms(&*self.tok, query)?
            .into_iter()
            .filter(|t| t.chars().count() >= self.min_term_len)
            .collect();
        if q.is_empty() || evidence.is_empty() {
            return Ok(Verdict {
                support: Support::Insufficient,
                confidence: 1.0,
                window: evidence.len(),
            });
        }

        let mut seen: HashSet<String> = HashSet::new();
        for e in evidence {
            seen.extend(terms(&*self.tok, e)?);
        }
        let hit = q.iter().filter(|t| seen.contains(*t)).count() as f32;
        let coverage = hit / q.len() as f32;

        Ok(Verdict {
            support: if coverage >= self.threshold {
                Support::Supports
            } else {
                Support::Insufficient
            },
            // Confidence is distance from the threshold, so the risk-coverage
            // sweep has something monotone to sort by.
            confidence: (coverage - self.threshold).abs().min(1.0),
            window: evidence.len(),
        })
    }
}
