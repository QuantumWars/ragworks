//! Typed judging: reranking and verification through a schema-constrained model.
//!
//! The model answers questions drawn from a supplied schema rather than
//! generating tokens, so an out-of-schema verdict is structurally impossible --
//! which is the right property for an instrument that measures invalid output.
//!
//! Every question in a request is evaluated against the same state in parallel,
//! so fan-out is close to free: reranking a shortlist costs one round trip, not
//! one per candidate.
//!
//! Measured in this repository's wave-1 experiment on HotpotQA: reranking moved
//! recall@5 from 0.748 to 0.822 (p=0.0001), and set-level sufficiency with
//! abstention cut risk from 0.260 to 0.062, at roughly $0.09 per thousand
//! queries.

use ragworks_core::{
    Component, Error, Reranker, Result, Support, Verdict, Verifier,
};
use ragworks_net::{HttpPost, RetryPolicy};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JevConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    /// Environment variable holding the API key. Never the key itself.
    #[serde(default = "default_key_env")]
    pub api_key_env: String,
    /// Characters of each candidate sent to the model. Accuracy falls as
    /// irrelevant state grows, so truncation is a quality control, not only a
    /// cost control.
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default)]
    pub requests_per_minute: f64,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Cache namespace for successful responses. `null` disables caching.
    #[serde(default = "default_cache")]
    pub cache: Option<String>,
    #[serde(default)]
    pub retry: RetryPolicy,
}

fn default_model() -> String {
    "typesafe/jev-1.13".into()
}
fn default_endpoint() -> String {
    "https://openrouter.ai/api/v1/systemone".into()
}
fn default_key_env() -> String {
    "OPENROUTER_API_KEY".into()
}
fn default_max_chars() -> usize {
    1200
}
fn default_timeout() -> u64 {
    60
}
fn default_cache() -> Option<String> {
    Some("jev".into())
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            endpoint: default_endpoint(),
            api_key_env: default_key_env(),
            max_chars: default_max_chars(),
            requests_per_minute: 0.0,
            timeout_secs: default_timeout(),
            cache: default_cache(),
            retry: RetryPolicy::default(),
        }
    }
}

pub struct Jev {
    model: String,
    endpoint: String,
    api_key: String,
    max_chars: usize,
    retry: RetryPolicy,
    http: Box<dyn HttpPost>,
}

impl std::fmt::Debug for Jev {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jev").field("model", &self.model).finish()
    }
}

impl Component for Jev {
    type Config = JevConfig;
    const NAME: &'static str = "jev";
    const SUMMARY: &'static str =
        "Schema-constrained judging; an out-of-schema verdict cannot occur.";

    fn build(config: Self::Config) -> Result<Self> {
        let http = ragworks_net::transport(
            config.timeout_secs,
            config.cache.as_deref(),
            config.requests_per_minute,
        )?;
        Self::with_http(config, http)
    }
}

impl Jev {
    pub fn with_http(config: JevConfig, http: Box<dyn HttpPost>) -> Result<Self> {
        if config.max_chars == 0 {
            return Err(Error::config(Self::NAME, "max_chars must be > 0"));
        }
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            Error::config(
                Self::NAME,
                format!(
                    "environment variable {} is not set; keys are read from the environment, \
                     never from configuration",
                    config.api_key_env
                ),
            )
        })?;
        Ok(Self {
            model: config.model,
            endpoint: config.endpoint,
            api_key,
            max_chars: config.max_chars,
            retry: config.retry,
            http,
        })
    }

    fn clip(&self, s: &str) -> String {
        s.chars().take(self.max_chars).collect()
    }

    /// One round trip: every question is scored against the same state.
    fn ask(
        &self,
        state: serde_json::Value,
        questions: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let body =
            serde_json::json!({"model": self.model, "state": state, "questions": questions})
                .to_string();
        let headers = [
            ("Authorization", format!("Bearer {}", self.api_key)),
            ("Content-Type", "application/json".to_string()),
        ];

        let raw = self.retry.run(&|d| std::thread::sleep(d), |_| {
            let (status, text) = self.http.post(&self.endpoint, &headers, &body)?;
            if (200..300).contains(&status) {
                return Ok(text);
            }
            let msg = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or_else(|| text.chars().take(200).collect());
            Err(Error::provider(&self.model, Some(status), msg))
        })?;

        serde_json::from_str(&raw)
            .map_err(|e| Error::provider(&self.model, None, format!("malformed JSON: {e}")))
    }
}

impl Reranker for Jev {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn rerank(&self, query: &str, candidates: &[&str], out: &mut Vec<f32>) -> Result<()> {
        if candidates.is_empty() {
            return Ok(());
        }
        let state = serde_json::json!({
            "query": query,
            "candidates": candidates.iter().map(|c| self.clip(c)).collect::<Vec<_>>(),
        });
        let questions: serde_json::Map<String, serde_json::Value> = (0..candidates.len())
            .map(|i| {
                (
                    format!("c{i}"),
                    serde_json::json!({
                        "type": "noul",
                        "instructions": format!(
                            "Passage `candidates[{i}]` directly answers `query`."
                        ),
                        "criteria": {
                            "true": "The passage supplies a fact the answer depends on",
                            "false": "It is about something else, or mentions the subject only in passing"
                        }
                    }),
                )
            })
            .collect();

        let resp = self.ask(state, serde_json::Value::Object(questions))?;
        let answers = &resp["answers"];
        for i in 0..candidates.len() {
            // A missing answer scores zero rather than failing the batch: one
            // unanswered question should cost one candidate, not the query.
            out.push(answers[format!("c{i}")]["noul"].as_f64().unwrap_or(0.0) as f32);
        }
        Ok(())
    }
}

impl Verifier for Jev {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn verify(&self, query: &str, evidence: &[&str]) -> Result<Verdict> {
        if evidence.is_empty() {
            // No evidence cannot support anything, and asking costs money to
            // learn that.
            return Ok(Verdict { support: Support::Insufficient, confidence: 1.0, window: 0 });
        }
        let state = serde_json::json!({
            "question": query,
            "evidence": evidence.iter().map(|e| self.clip(e)).collect::<Vec<_>>(),
        });
        // One question over the whole set. Sufficiency is not decidable per
        // passage: a missing reasoning hop is invisible to any scorer that
        // examines passages independently.
        let questions = serde_json::json!({
            "verdict": {
                "type": "choice",
                "instructions": "Taken TOGETHER, do the passages in `evidence` establish an \
                                 answer to `question`? Judge the set as a whole: a missing \
                                 reasoning step, or a conflict between passages, means the \
                                 evidence is insufficient.",
                "criteria": {
                    "supports": "The evidence together establishes an answer",
                    "refutes": "The evidence together contradicts the question's premise",
                    "insufficient": "A step is missing, or the passages conflict, so no answer is established"
                }
            }
        });

        let resp = self.ask(state, questions)?;
        let v = &resp["answers"]["verdict"];
        let support = match v["choice"].as_str() {
            Some("supports") => Support::Supports,
            Some("refutes") => Support::Refutes,
            _ => Support::Insufficient,
        };
        Ok(Verdict {
            support,
            confidence: v["confidence"].as_f64().unwrap_or(0.0) as f32,
            // Without this the verdict is uninterpretable: "insufficient over 4"
            // and "insufficient over 8" are different claims.
            window: evidence.len(),
        })
    }
}
