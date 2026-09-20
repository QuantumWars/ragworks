//! A minimal HTTP seam.
//!
//! Requests go through a trait rather than straight to `ureq` so that retry
//! behaviour, rate limiting and response parsing are testable without a
//! network. Getting backoff wrong is the kind of bug that only shows up against
//! a live rate limit, which is exactly when it is most expensive to discover.

use std::sync::Mutex;
use std::time::Duration;

use ragworks_core::{Error, Result};

/// One blocking POST. Returns the status and body even for 4xx and 5xx, so the
/// caller can classify the failure instead of receiving an opaque transport
/// error.
pub trait HttpPost: Send + Sync {
    fn post(&self, url: &str, headers: &[(&str, String)], body: &str) -> Result<(u16, String)>;
}

pub struct UreqClient {
    agent: ureq::Agent,
}

impl UreqClient {
    pub fn new(timeout: Duration) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            // Classify status codes ourselves; a 429 is a retry signal, not a
            // transport failure, and the body often says how long to wait.
            .http_status_as_error(false)
            .build();
        Self { agent: config.into() }
    }
}

impl Default for UreqClient {
    fn default() -> Self {
        Self::new(Duration::from_secs(60))
    }
}

impl HttpPost for UreqClient {
    fn post(&self, url: &str, headers: &[(&str, String)], body: &str) -> Result<(u16, String)> {
        let mut req = self.agent.post(url);
        for (k, v) in headers {
            req = req.header(*k, v.as_str());
        }
        let mut resp = req
            .send(body)
            .map_err(|e| Error::provider("http", None, format!("transport: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| Error::provider("http", Some(status), format!("reading body: {e}")))?;
        Ok((status, text))
    }
}

/// Canned responses, for tests. Records every request it received.
pub struct MockHttp {
    responses: Mutex<Vec<Result<(u16, String)>>>,
    pub calls: Mutex<Vec<String>>,
}

impl MockHttp {
    /// Responses are returned in order; the last one repeats once exhausted.
    pub fn new(responses: Vec<Result<(u16, String)>>) -> Self {
        Self { responses: Mutex::new(responses), calls: Mutex::new(Vec::new()) }
    }

    pub fn ok(body: &str) -> Self {
        Self::new(vec![Ok((200, body.to_string()))])
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl HttpPost for MockHttp {
    fn post(&self, _url: &str, _headers: &[(&str, String)], body: &str) -> Result<(u16, String)> {
        self.calls.lock().unwrap().push(body.to_string());
        let mut r = self.responses.lock().unwrap();
        if r.len() > 1 { r.remove(0) } else { r.first().cloned_result() }
    }
}

/// Helper so `MockHttp` can repeat its final response without `Result: Clone`.
trait ClonedResult {
    fn cloned_result(&self) -> Result<(u16, String)>;
}

impl ClonedResult for Option<&Result<(u16, String)>> {
    fn cloned_result(&self) -> Result<(u16, String)> {
        match self {
            Some(Ok((s, b))) => Ok((*s, b.clone())),
            Some(Err(e)) => Err(Error::provider("mock", None, e.to_string())),
            None => Err(Error::provider("mock", None, "no responses configured")),
        }
    }
}
