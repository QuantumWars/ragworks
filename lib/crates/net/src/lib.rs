//! Provider plumbing shared by every component that calls a remote model.
//!
//! Extracted when a second component needed it. Embedders and judges are
//! unrelated, and both required the same three things: a mockable HTTP seam, a
//! retry policy driven by the error classification, and client-side rate
//! limiting. Two independent needs for identical code is what justifies a
//! shared crate -- the alternative was a judge crate depending on an embedder
//! crate, which describes no real relationship.

pub mod http;
pub mod limit;
pub mod retry;

pub use http::{HttpPost, MockHttp, UreqClient};
pub use limit::RateLimiter;
pub use retry::RetryPolicy;
