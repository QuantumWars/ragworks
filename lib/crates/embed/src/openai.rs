//! OpenAI-compatible `/embeddings` client.
//!
//! One implementation covers OpenAI, OpenRouter, vLLM, Ollama, LM Studio and
//! anything else speaking the same shape; only the endpoint changes.
//!
//! **The API key is read from an environment variable named in the config, and
//! never from the config itself.** Configs get committed, pasted into issues and
//! printed in logs; keys must not travel with them.

use std::sync::Mutex;
use ragworks_core::{Component, Embedder, Error, Result};
use schemars::JsonSchema;
use serde::Deserialize;

use ragworks_net::{HttpPost, RetryPolicy};

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenAiConfig {
    /// Model identifier, e.g. `openai/text-embedding-3-small`.
    pub model: String,
    /// Expected vector width. Checked against the first response, so a model
    /// and dimension that disagree fail immediately instead of corrupting an
    /// index.
    pub dim: usize,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    /// Environment variable holding the API key. Never the key itself.
    #[serde(default = "default_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_max_batch")]
    pub max_batch: usize,
    #[serde(default = "default_rpm")]
    pub requests_per_minute: f64,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Cache namespace for successful responses. `null` disables caching.
    #[serde(default = "default_cache")]
    pub cache: Option<String>,
    #[serde(default)]
    pub retry: RetryPolicy,
}

fn default_endpoint() -> String {
    "https://openrouter.ai/api/v1/embeddings".into()
}
fn default_key_env() -> String {
    "OPENROUTER_API_KEY".into()
}
fn default_max_batch() -> usize {
    64
}
fn default_rpm() -> f64 {
    0.0
}
fn default_timeout() -> u64 {
    60
}
fn default_cache() -> Option<String> {
    Some("openai".into())
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            dim: 0,
            endpoint: default_endpoint(),
            api_key_env: default_key_env(),
            max_batch: default_max_batch(),
            requests_per_minute: default_rpm(),
            timeout_secs: default_timeout(),
            cache: default_cache(),
            retry: RetryPolicy::default(),
        }
    }
}

/// Cumulative spend, so a caller can enforce a budget rather than discover it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub usd: f64,
}

pub struct OpenAiCompatible {
    model: String,
    dim: usize,
    endpoint: String,
    api_key: String,
    max_batch: usize,
    retry: RetryPolicy,
    http: Box<dyn HttpPost>,
    usage: Mutex<Usage>,
}

impl std::fmt::Debug for OpenAiCompatible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatible")
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

impl Component for OpenAiCompatible {
    type Config = OpenAiConfig;
    const NAME: &'static str = "openai";
    const SUMMARY: &'static str =
        "OpenAI-compatible /embeddings endpoint, with retry, rate limiting and cost accounting.";

    fn build(config: Self::Config) -> Result<Self> {
        let http = ragworks_net::transport(
            config.timeout_secs,
            config.cache.as_deref(),
            config.requests_per_minute,
        )?;
        Self::with_http(config, http)
    }
}

impl OpenAiCompatible {
    /// Build against a supplied transport. Tests use this with a mock.
    pub fn with_http(config: OpenAiConfig, http: Box<dyn HttpPost>) -> Result<Self> {
        if config.model.is_empty() {
            return Err(Error::config(Self::NAME, "model is required"));
        }
        if config.dim == 0 {
            return Err(Error::config(Self::NAME, "dim is required and must be > 0"));
        }
        if config.max_batch == 0 {
            return Err(Error::config(Self::NAME, "max_batch must be > 0"));
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
            dim: config.dim,
            endpoint: config.endpoint,
            api_key,
            max_batch: config.max_batch,
            retry: config.retry,
            http,
            usage: Mutex::new(Usage::default()),
        })
    }

    pub fn usage(&self) -> Usage {
        *self.usage.lock().unwrap()
    }

    fn parse(&self, body: &str, expected: usize, out: &mut Vec<f32>) -> Result<()> {
        let v: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| Error::provider(&self.model, None, format!("malformed JSON: {e}")))?;

        let data = v["data"]
            .as_array()
            .ok_or_else(|| Error::provider(&self.model, None, "response has no `data` array"))?;
        if data.len() != expected {
            return Err(Error::provider(
                &self.model,
                None,
                format!("asked for {expected} embeddings, got {}", data.len()),
            ));
        }

        // Order is not guaranteed, so sort by the index the API returns rather
        // than trusting array position.
        let mut rows: Vec<(usize, &serde_json::Value)> = data
            .iter()
            .enumerate()
            .map(|(i, e)| (e["index"].as_u64().map(|x| x as usize).unwrap_or(i), e))
            .collect();
        rows.sort_by_key(|(i, _)| *i);

        for (_, e) in rows {
            let arr = e["embedding"].as_array().ok_or_else(|| {
                Error::provider(&self.model, None, "entry has no `embedding` array")
            })?;
            if arr.len() != self.dim {
                return Err(Error::DimensionMismatch { expected: self.dim, actual: arr.len() });
            }
            out.extend(arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32));
        }

        let u = &v["usage"];
        let mut acc = self.usage.lock().unwrap();
        acc.requests += 1;
        acc.prompt_tokens += u["prompt_tokens"].as_u64().unwrap_or(0);
        acc.usd += u["cost"].as_f64().unwrap_or(0.0);
        Ok(())
    }

    fn send_batch(&self, texts: &[&str], out: &mut Vec<f32>) -> Result<()> {
        let body = serde_json::json!({"model": self.model, "input": texts}).to_string();
        let headers = [
            ("Authorization", format!("Bearer {}", self.api_key)),
            ("Content-Type", "application/json".to_string()),
        ];

        let raw = self.retry.run(&|d| std::thread::sleep(d), |_| {
            let (status, text) = self.http.post(&self.endpoint, &headers, &body)?;
            if (200..300).contains(&status) {
                return Ok(text);
            }
            // Surface the provider's own message; it usually says why, and for
            // a 429 often says for how long.
            let msg = serde_json::from_str::<serde_json::Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or_else(|| text.chars().take(200).collect());
            Err(Error::provider(&self.model, Some(status), msg))
        })?;

        self.parse(&raw, texts.len(), out)
    }
}

impl Embedder for OpenAiCompatible {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn embed(&self, texts: &[&str], out: &mut Vec<f32>) -> Result<()> {
        if texts.is_empty() {
            return Ok(());
        }
        out.reserve(texts.len() * self.dim);
        // Split internally as well as exposing `max_batch`: a caller that
        // forgets to chunk should get correct results, not a 400.
        for batch in texts.chunks(self.max_batch) {
            self.send_batch(batch, out)?;
        }
        Ok(())
    }
}
