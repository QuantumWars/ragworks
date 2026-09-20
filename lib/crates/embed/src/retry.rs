//! Retry with exponential backoff, driven by the error classification in
//! `ragworks_core::Error`.
//!
//! This is where `is_retryable` earns its keep. Retrying a 401 cannot succeed
//! and wastes quota; not retrying a 429 throws away a run that would have
//! completed. The policy sleeps through an injected function so the logic is
//! testable in microseconds rather than minutes.

use std::time::Duration;

use ragworks_core::{Error, Result};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RetryPolicy {
    /// Total attempts, including the first. 1 disables retrying.
    #[serde(default = "default_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_base_ms")]
    pub base_delay_ms: u64,
    #[serde(default = "default_max_ms")]
    pub max_delay_ms: u64,
}

fn default_attempts() -> u32 {
    5
}
fn default_base_ms() -> u64 {
    500
}
fn default_max_ms() -> u64 {
    30_000
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: default_attempts(),
            base_delay_ms: default_base_ms(),
            max_delay_ms: default_max_ms(),
        }
    }
}

/// Cheap jitter without a `rand` dependency. Full jitter (uniform over
/// `[0, backoff]`) rather than equal jitter, because it spreads a burst of
/// retries widest and the difference matters when several workers back off
/// together.
fn jitter(seed: u64, ceiling_ms: u64) -> u64 {
    if ceiling_ms == 0 {
        return 0;
    }
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x % ceiling_ms
}

impl RetryPolicy {
    pub fn delay_for(&self, attempt: u32, seed: u64) -> Duration {
        let exp = self.base_delay_ms.saturating_mul(1u64 << attempt.min(16));
        Duration::from_millis(jitter(seed, exp.min(self.max_delay_ms)))
    }

    /// Run `f`, retrying only failures the error model says could succeed.
    pub fn run<T>(
        &self,
        sleep: &dyn Fn(Duration),
        mut f: impl FnMut(u32) -> Result<T>,
    ) -> Result<T> {
        let attempts = self.max_attempts.max(1);
        let mut last: Option<Error> = None;
        for attempt in 0..attempts {
            match f(attempt) {
                Ok(v) => return Ok(v),
                Err(e) if e.is_retryable() && attempt + 1 < attempts => {
                    sleep(self.delay_for(attempt, attempt as u64 ^ 0x9e37_79b9));
                    last = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::provider("retry", None, "no attempts made")))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn policy() -> RetryPolicy {
        RetryPolicy { max_attempts: 4, base_delay_ms: 10, max_delay_ms: 1000 }
    }

    #[test]
    fn a_rate_limit_is_retried_until_it_succeeds() {
        let calls = RefCell::new(0);
        let slept = RefCell::new(0u32);
        let out: Result<u32> = policy().run(&|_| *slept.borrow_mut() += 1, |_| {
            let n = { *calls.borrow() + 1 };
            *calls.borrow_mut() = n;
            if n < 3 { Err(Error::provider("x", Some(429), "slow down")) } else { Ok(n) }
        });
        assert_eq!(out.unwrap(), 3);
        assert_eq!(*calls.borrow(), 3);
        assert_eq!(*slept.borrow(), 2, "one sleep between each pair of attempts");
    }

    #[test]
    fn an_auth_failure_is_not_retried() {
        let calls = RefCell::new(0);
        let out: Result<u32> = policy().run(&|_| panic!("must not sleep"), |_| {
            *calls.borrow_mut() += 1;
            Err(Error::provider("x", Some(401), "bad key"))
        });
        assert!(out.is_err());
        assert_eq!(*calls.borrow(), 1, "retrying a 401 cannot help and wastes quota");
    }

    #[test]
    fn retries_are_bounded_and_the_last_error_survives() {
        let calls = RefCell::new(0);
        let out: Result<u32> = policy().run(&|_| {}, |_| {
            *calls.borrow_mut() += 1;
            Err(Error::provider("x", Some(503), "unavailable"))
        });
        assert_eq!(*calls.borrow(), 4);
        assert!(out.err().unwrap().to_string().contains("unavailable"));
    }

    #[test]
    fn backoff_grows_and_is_capped() {
        let p = RetryPolicy { max_attempts: 10, base_delay_ms: 100, max_delay_ms: 1000 };
        // Jitter is uniform over [0, ceiling], so compare ceilings, not samples.
        for attempt in 0..8 {
            assert!(p.delay_for(attempt, 7).as_millis() <= 1000, "cap must hold at {attempt}");
        }
    }
}
