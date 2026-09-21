//! Transforms that need no model: the control, and classical relevance feedback.
//!
//! RM3-style feedback predates every LLM technique in this crate by decades and
//! is often competitive with them. Measuring against it is what distinguishes
//! "the model helped" from "any expansion would have helped".

use std::collections::{HashMap, HashSet};

use ragworks_core::{
    Component, Error, PluginSpec, QueryTransform, Result, Tokenizer, tokenize,
};
use schemars::JsonSchema;
use serde::Deserialize;

fn default_tokenizer() -> PluginSpec {
    PluginSpec::named("simple")
}

fn terms(tok: &dyn Tokenizer, text: &str) -> Result<Vec<String>> {
    let mut spans = Vec::new();
    tok.tokenize(text, &mut spans)?;
    Ok(spans.iter().map(|(s, e)| text[*s as usize..*e as usize].to_lowercase()).collect())
}

// ---------------------------------------------------------------- identity

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct IdentityConfig {}

/// Pass the query through unchanged. The control.
#[derive(Debug, Default)]
pub struct Identity;

impl Component for Identity {
    type Config = IdentityConfig;
    const NAME: &'static str = "identity";
    const SUMMARY: &'static str = "The query, unchanged. The control every other transform is measured against.";
    fn build(_config: Self::Config) -> Result<Self> {
        Ok(Self)
    }
}

impl QueryTransform for Identity {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn transform(&self, query: &str, _feedback: &[&str], out: &mut Vec<String>) -> Result<()> {
        out.push(query.to_string());
        Ok(())
    }
}

// -------------------------------------------------------------------- rm3

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rm3Config {
    #[serde(default = "default_tokenizer")]
    pub tokenizer: PluginSpec,
    /// Expansion terms appended to the query.
    #[serde(default = "default_terms")]
    pub expansion_terms: usize,
    /// Feedback documents to read. Beyond the first few, relevance decays and
    /// the expansion drifts off topic.
    #[serde(default = "default_docs")]
    pub feedback_docs: usize,
    /// Terms shorter than this are ignored. A blunt substitute for a stop list
    /// that happens to remove most English function words without being tied to
    /// one language.
    #[serde(default = "default_min_len")]
    pub min_term_len: usize,
}

fn default_terms() -> usize {
    10
}
fn default_docs() -> usize {
    3
}
fn default_min_len() -> usize {
    4
}

impl Default for Rm3Config {
    fn default() -> Self {
        Self {
            tokenizer: default_tokenizer(),
            expansion_terms: default_terms(),
            feedback_docs: default_docs(),
            min_term_len: default_min_len(),
        }
    }
}

/// Pseudo-relevance feedback: assume the top results are relevant, harvest
/// their distinctive terms, and re-query with them appended.
///
/// Simplified from textbook RM3 in one way worth stating: real RM3 weights
/// expansion terms and mixes them with the original query model. The transform
/// interface produces a query *string*, which carries no weights, so terms are
/// appended unweighted. That makes this a floor for feedback expansion rather
/// than a faithful implementation of the algorithm.
pub struct Rm3 {
    tok: Box<dyn Tokenizer>,
    expansion_terms: usize,
    feedback_docs: usize,
    min_term_len: usize,
}

impl std::fmt::Debug for Rm3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rm3").field("expansion_terms", &self.expansion_terms).finish()
    }
}

impl Component for Rm3 {
    type Config = Rm3Config;
    const NAME: &'static str = "rm3";
    const SUMMARY: &'static str =
        "Pseudo-relevance feedback: expand the query with terms from the top results.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.feedback_docs == 0 {
            return Err(Error::config(Self::NAME, "feedback_docs must be > 0"));
        }
        Ok(Self {
            tok: tokenize::registry().build_spec(&config.tokenizer)?,
            expansion_terms: config.expansion_terms,
            feedback_docs: config.feedback_docs,
            min_term_len: config.min_term_len,
        })
    }
}

impl QueryTransform for Rm3 {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn uses_feedback(&self) -> bool {
        true
    }

    fn transform(&self, query: &str, feedback: &[&str], out: &mut Vec<String>) -> Result<()> {
        if feedback.is_empty() {
            // Nothing to learn from yet. Returning the query unchanged lets the
            // same transform serve both passes of a two-pass pipeline.
            out.push(query.to_string());
            return Ok(());
        }

        let existing: HashSet<String> = terms(&*self.tok, query)?.into_iter().collect();
        let mut score: HashMap<String, f32> = HashMap::new();

        for doc in feedback.iter().take(self.feedback_docs) {
            let ts = terms(&*self.tok, doc)?;
            let seen: HashSet<&String> = ts.iter().collect();
            for t in &ts {
                if t.chars().count() < self.min_term_len || existing.contains(t) {
                    continue;
                }
                *score.entry(t.clone()).or_insert(0.0) += 1.0;
            }
            // A term appearing in several feedback documents is more likely to
            // be about the topic than about one document.
            for t in seen {
                if t.chars().count() >= self.min_term_len && !existing.contains(t) {
                    *score.entry(t.clone()).or_insert(0.0) += 2.0;
                }
            }
        }

        let mut ranked: Vec<(String, f32)> = score.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut expanded = query.to_string();
        for (term, _) in ranked.into_iter().take(self.expansion_terms) {
            expanded.push(' ');
            expanded.push_str(&term);
        }
        out.push(expanded);
        Ok(())
    }
}
