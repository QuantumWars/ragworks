//! Minimal chat-completions client with a model fallback chain.
//!
//! The chain is not a nicety. Free and shared tiers rate-limit individual
//! models and take them out of service without notice, so a single pinned model
//! is not reliable enough to run an experiment against. Each model in the chain
//! is tried in turn, and only a failure of every one of them fails the call.
//!
//! Lives here rather than in a shared crate because it has exactly one consumer
//! so far. It moves when a second appears, the way the HTTP plumbing did.

use ragworks_core::{Error, Result};
use ragworks_net::{HttpPost, RetryPolicy};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatConfig {
    /// Tried in order. A later entry is used only when every earlier one fails.
    #[serde(default = "default_models")]
    pub models: Vec<String>,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    /// Environment variable holding the API key. Never the key itself.
    #[serde(default = "default_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    /// Ask the provider to skip reasoning tokens.
    ///
    /// Reasoning models can spend an entire token budget deliberating and emit
    /// no content at all. Measured on 2026-09-21: one free model produced 414
    /// reasoning tokens against a 400-token budget and returned an empty
    /// completion, which the fallback chain then treated as a model failure.
    /// Rewriting a query does not benefit from deliberation, and disabling it
    /// was 2.6x faster on a model that did work, using a tenth of the tokens.
    #[serde(default = "yes")]
    pub disable_reasoning: bool,
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

fn default_models() -> Vec<String> {
    // Verified responding on 2026-09-21. Availability changes, which is the
    // reason this is a list.
    vec!["nex-agi/nex-n2.5-mini:free".into(), "inclusionai/ling-3.0-flash-vl:free".into()]
}
fn default_endpoint() -> String {
    "https://openrouter.ai/api/v1/chat/completions".into()
}
fn default_key_env() -> String {
    "OPENROUTER_API_KEY".into()
}
fn default_max_tokens() -> usize {
    400
}
fn default_timeout() -> u64 {
    90
}
fn default_cache() -> Option<String> {
    Some("chat".into())
}
fn yes() -> bool {
    true
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            models: default_models(),
            endpoint: default_endpoint(),
            api_key_env: default_key_env(),
            max_tokens: default_max_tokens(),
            temperature: 0.0,
            disable_reasoning: yes(),
            requests_per_minute: 0.0,
            timeout_secs: default_timeout(),
            cache: default_cache(),
            retry: RetryPolicy::default(),
        }
    }
}

pub struct Chat {
    models: Vec<String>,
    endpoint: String,
    api_key: String,
    max_tokens: usize,
    temperature: f32,
    disable_reasoning: bool,
    retry: RetryPolicy,
    http: Box<dyn HttpPost>,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat").field("models", &self.models).finish()
    }
}

impl Chat {
    pub fn new(config: ChatConfig) -> Result<Self> {
        let http = ragworks_net::transport(
            config.timeout_secs,
            config.cache.as_deref(),
            config.requests_per_minute,
        )?;
        Self::with_http(config, http)
    }

    pub fn with_http(config: ChatConfig, http: Box<dyn HttpPost>) -> Result<Self> {
        if config.models.is_empty() {
            return Err(Error::config("chat", "at least one model is required"));
        }
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            Error::config(
                "chat",
                format!(
                    "environment variable {} is not set; keys are read from the environment, \
                     never from configuration",
                    config.api_key_env
                ),
            )
        })?;
        Ok(Self {
            models: config.models,
            endpoint: config.endpoint,
            api_key,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            disable_reasoning: config.disable_reasoning,
            retry: config.retry,
            http,
        })
    }

    fn call(&self, model: &str, system: &str, user: &str) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        });
        if self.disable_reasoning {
            payload["reasoning"] = serde_json::json!({"enabled": false});
        }
        let body = payload.to_string();
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
            Err(Error::provider(model, Some(status), msg))
        })?;

        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::provider(model, None, format!("malformed JSON: {e}")))?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("").trim();
        if content.is_empty() {
            // A model that deliberates past its budget returns nothing. That is
            // a failure of this model rather than of the request, so the chain
            // moves on -- and `disable_reasoning` exists to prevent it.
            let reasoning = v["usage"]["completion_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap_or(0);
            return Err(Error::provider(
                model,
                None,
                if reasoning > 0 {
                    format!("empty completion after {reasoning} reasoning tokens")
                } else {
                    "empty completion".to_string()
                },
            ));
        }
        Ok(content.to_string())
    }

    /// Complete, walking the fallback chain until one model answers.
    pub fn complete(&self, system: &str, user: &str) -> Result<String> {
        let mut last = None;
        for model in &self.models {
            match self.call(model, system, user) {
                Ok(text) => return Ok(text),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::provider("chat", None, "no models configured")))
    }
}

/// Parse a model's list reply into clean lines.
///
/// Models decorate lists however they like. Stripping numbering, bullets and
/// surrounding quotes here means every transform gets the same treatment rather
/// than each inventing its own.
pub fn parse_lines(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut t = line.trim();
        if t.is_empty() {
            continue;
        }
        t = t.trim_start_matches(['-', '*', '•']).trim();
        if let Some((head, rest)) = t.split_once(['.', ')'])
            && !head.is_empty()
            && head.len() <= 2
            && head.chars().all(|c| c.is_ascii_digit())
        {
            t = rest.trim();
        }
        let t = t.trim_matches(['"', '\'', '`']).trim();
        // A model that narrates before listing produces prose lines; keeping
        // only plausible queries is cheaper than prompting harder.
        if t.len() < 3 || t.ends_with(':') {
            continue;
        }
        out.push(t.to_string());
        if out.len() >= max {
            break;
        }
    }
    out
}
