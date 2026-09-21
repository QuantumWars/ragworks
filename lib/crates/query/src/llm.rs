//! Model-backed reformulation: paraphrase, hypothetical document, decomposition.
//!
//! Each of these is a hypothesis, not a known win. Published evaluation reports
//! that **query decomposition improves structured domains but degrades ranking
//! precision on multi-hop benchmarks**, which is the opposite of the intuition
//! that makes people reach for it. All three ship so the question can be
//! settled by measurement against [`crate::Identity`] and [`crate::Rm3`] rather
//! than by assumption.

use ragworks_core::{Component, Error, QueryTransform, Result};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::chat::{Chat, ChatConfig, parse_lines};

/// Degrade to the original query when the provider is transiently unavailable.
///
/// A reformulation is an optimisation: retrieving with the plain question is
/// worse than retrieving with a good rewrite, and enormously better than
/// failing the query. But only *transient* failures degrade. A missing API key
/// or a rejected request is a misconfiguration, and silently returning
/// mediocre results would present it as poor retrieval quality — the hardest
/// kind of fault to diagnose. Those propagate.
fn degrade_if_transient(query: &str, out: &mut Vec<String>, r: Result<()>) -> Result<()> {
    match r {
        Ok(()) => Ok(()),
        Err(e) if e.is_retryable() => {
            if out.is_empty() {
                out.push(query.to_string());
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// ------------------------------------------------------------ multi-query

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MultiQueryConfig {
    #[serde(default)]
    pub chat: ChatConfig,
    /// Paraphrases requested, excluding the original.
    #[serde(default = "default_variants")]
    pub variants: usize,
    /// Keep the original query alongside the paraphrases.
    #[serde(default = "yes")]
    pub keep_original: bool,
}

fn default_variants() -> usize {
    3
}
fn yes() -> bool {
    true
}

impl Default for MultiQueryConfig {
    fn default() -> Self {
        Self {
            chat: ChatConfig::default(),
            variants: default_variants(),
            keep_original: yes(),
        }
    }
}

/// Several phrasings of one question, all retrieved and fused.
///
/// The bet is vocabulary coverage: a document phrased unlike the question is
/// reachable by a paraphrase that happens to share its wording.
pub struct MultiQuery {
    chat: Chat,
    variants: usize,
    keep_original: bool,
}

impl std::fmt::Debug for MultiQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiQuery").field("variants", &self.variants).finish()
    }
}

impl Component for MultiQuery {
    type Config = MultiQueryConfig;
    const NAME: &'static str = "multi_query";
    const SUMMARY: &'static str = "Several paraphrases of the query, to be retrieved and fused.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.variants == 0 {
            return Err(Error::config(Self::NAME, "variants must be > 0"));
        }
        Ok(Self {
            chat: Chat::new(config.chat)?,
            variants: config.variants,
            keep_original: config.keep_original,
        })
    }
}

impl MultiQuery {
    pub fn with_chat(chat: Chat, variants: usize, keep_original: bool) -> Self {
        Self { chat, variants, keep_original }
    }
}

const LIST_RULES: &str =
    "Answer with one item per line. No numbering, no bullets, no preamble, no explanation.";

impl QueryTransform for MultiQuery {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn transform(&self, query: &str, _feedback: &[&str], out: &mut Vec<String>) -> Result<()> {
        if self.keep_original {
            out.push(query.to_string());
        }
        let attempt = self
            .chat
            .complete(
                &format!(
                    "You rewrite search queries. Produce {} alternative phrasings that a \
                     relevant document might use, keeping the same meaning. {LIST_RULES}",
                    self.variants
                ),
                query,
            )
            .map(|reply| {
                for line in parse_lines(&reply, self.variants) {
                    if !line.eq_ignore_ascii_case(query) {
                        out.push(line);
                    }
                }
            });
        degrade_if_transient(query, out, attempt)
    }
}

// ------------------------------------------------------------------- hyde

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct HydeConfig {
    #[serde(default)]
    pub chat: ChatConfig,
    /// Keep the original query alongside the hypothetical passage.
    #[serde(default)]
    pub keep_original: bool,
}

/// Hypothetical Document Embeddings: answer the question, then search with the
/// answer.
///
/// A question and the passage answering it are written differently; a fabricated
/// answer is written like the real one. The fabrication may be factually wrong
/// and that is tolerable, because it is used only as a retrieval probe and never
/// shown to anyone.
pub struct Hyde {
    chat: Chat,
    keep_original: bool,
}

impl std::fmt::Debug for Hyde {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hyde").finish()
    }
}

impl Component for Hyde {
    type Config = HydeConfig;
    const NAME: &'static str = "hyde";
    const SUMMARY: &'static str =
        "Generate a hypothetical answer and retrieve with that instead of the question.";

    fn build(config: Self::Config) -> Result<Self> {
        Ok(Self { chat: Chat::new(config.chat)?, keep_original: config.keep_original })
    }
}

impl Hyde {
    pub fn with_chat(chat: Chat, keep_original: bool) -> Self {
        Self { chat, keep_original }
    }
}

impl QueryTransform for Hyde {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn transform(&self, query: &str, _feedback: &[&str], out: &mut Vec<String>) -> Result<()> {
        if self.keep_original {
            out.push(query.to_string());
        }
        let attempt = self
            .chat
            .complete(
                "Write a short, factual encyclopaedia passage that would answer the question. \
                 Two or three sentences. State facts directly; do not restate the question, \
                 and do not say you are unsure.",
                query,
            )
            .map(|passage| {
                let passage = passage.trim();
                if !passage.is_empty() {
                    out.push(passage.to_string());
                }
            });
        degrade_if_transient(query, out, attempt)
    }
}

// -------------------------------------------------------------- decompose

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecomposeConfig {
    #[serde(default)]
    pub chat: ChatConfig,
    #[serde(default = "default_max_parts")]
    pub max_parts: usize,
    #[serde(default = "yes")]
    pub keep_original: bool,
}

fn default_max_parts() -> usize {
    3
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            chat: ChatConfig::default(),
            max_parts: default_max_parts(),
            keep_original: yes(),
        }
    }
}

/// Split a question into the simpler questions it depends on.
///
/// Ships with a warning attached: published evaluation finds decomposition
/// helps in structured domains and **degrades ranking precision on multi-hop
/// benchmarks**, which is exactly where it is usually reached for. Treat a gain
/// here as a result to be measured, not expected.
pub struct Decompose {
    chat: Chat,
    max_parts: usize,
    keep_original: bool,
}

impl std::fmt::Debug for Decompose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decompose").field("max_parts", &self.max_parts).finish()
    }
}

impl Component for Decompose {
    type Config = DecomposeConfig;
    const NAME: &'static str = "decompose";
    const SUMMARY: &'static str =
        "Split a question into sub-questions; measured to hurt on multi-hop benchmarks.";

    fn build(config: Self::Config) -> Result<Self> {
        if config.max_parts == 0 {
            return Err(Error::config(Self::NAME, "max_parts must be > 0"));
        }
        Ok(Self {
            chat: Chat::new(config.chat)?,
            max_parts: config.max_parts,
            keep_original: config.keep_original,
        })
    }
}

impl Decompose {
    pub fn with_chat(chat: Chat, max_parts: usize, keep_original: bool) -> Self {
        Self { chat, max_parts, keep_original }
    }
}

impl QueryTransform for Decompose {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn transform(&self, query: &str, _feedback: &[&str], out: &mut Vec<String>) -> Result<()> {
        if self.keep_original {
            out.push(query.to_string());
        }
        let attempt = self
            .chat
            .complete(
                &format!(
                    "Break the question into at most {} simpler standalone questions that must \
                     be answered to answer it. If it is already simple, repeat it unchanged. \
                     {LIST_RULES}",
                    self.max_parts
                ),
                query,
            )
            .map(|reply| {
                for line in parse_lines(&reply, self.max_parts) {
                    if !line.eq_ignore_ascii_case(query) {
                        out.push(line);
                    }
                }
            });
        degrade_if_transient(query, out, attempt)
    }
}
