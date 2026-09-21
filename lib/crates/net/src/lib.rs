//! Provider plumbing shared by every component that calls a remote model.
//!
//! Extracted when a second component needed it. Embedders and judges are
//! unrelated, and both required the same three things: a mockable HTTP seam, a
//! retry policy driven by the error classification, and client-side rate
//! limiting. Two independent needs for identical code is what justifies a
//! shared crate -- the alternative was a judge crate depending on an embedder
//! crate, which describes no real relationship.

pub mod cache;
pub mod http;
pub mod limit;
pub mod retry;

pub use cache::{CacheStats, CachedHttp};
pub use http::{HttpPost, MockHttp, UreqClient};
pub use limit::{RateLimited, RateLimiter};
pub use retry::RetryPolicy;

use std::time::Duration;

use ragworks_core::Result;

/// Build the transport every provider-backed component uses.
///
/// `cache` names a directory under the cache root; `None` disables caching.
/// Caching is on by default in the component configs because this is research
/// infrastructure: re-running an evaluation after a metric change should not
/// re-spend the money. The cache key is the full request body, so changing a
/// prompt or a parameter is a miss, not a stale hit. The one real staleness
/// risk is a provider silently updating a model behind the same name.
/// Layered cache -> rate limit -> network, in that order. A cache hit consumed
/// no quota, so it must not wait for one.
pub fn transport(timeout_secs: u64, cache: Option<&str>, rpm: f64) -> Result<Box<dyn HttpPost>> {
    let mut client: Box<dyn HttpPost> = Box::new(UreqClient::new(Duration::from_secs(timeout_secs)));
    if rpm > 0.0 {
        client = Box::new(RateLimited::new(client, rpm));
    }
    match cache {
        Some(ns) => Ok(Box::new(CachedHttp::with_default_dir(client, ns)?)),
        None => Ok(client),
    }
}
