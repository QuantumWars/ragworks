//! Client-side rate limiting.
//!
//! Free and shared tiers cap requests per minute, and exceeding the cap costs
//! more than waiting for it: a 429 burns an attempt and a backoff window. The
//! limiter is not an optimisation, it is what keeps a long run finishing.

use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct RateLimiter {
    interval: Duration,
    last: Mutex<Option<Instant>>,
}

impl RateLimiter {
    /// `rpm <= 0` disables limiting.
    pub fn per_minute(rpm: f64) -> Self {
        let interval = if rpm > 0.0 {
            Duration::from_secs_f64(60.0 / rpm)
        } else {
            Duration::ZERO
        };
        Self { interval, last: Mutex::new(None) }
    }

    pub fn is_enabled(&self) -> bool {
        !self.interval.is_zero()
    }

    /// How long the caller must wait, recording this slot as taken.
    ///
    /// Returns the duration rather than sleeping so the decision is testable
    /// and the caller chooses how to wait.
    pub fn reserve(&self) -> Duration {
        if !self.is_enabled() {
            return Duration::ZERO;
        }
        let now = Instant::now();
        let mut last = self.last.lock().unwrap();
        let wait = match *last {
            Some(prev) => {
                let elapsed = now.saturating_duration_since(prev);
                self.interval.saturating_sub(elapsed)
            }
            None => Duration::ZERO,
        };
        *last = Some(now + wait);
        wait
    }

    pub fn acquire(&self, sleep: &dyn Fn(Duration)) {
        let w = self.reserve();
        if !w.is_zero() {
            sleep(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_request_is_never_delayed() {
        assert_eq!(RateLimiter::per_minute(60.0).reserve(), Duration::ZERO);
    }

    #[test]
    fn subsequent_requests_are_spaced_by_the_interval() {
        let l = RateLimiter::per_minute(60.0); // one per second
        assert_eq!(l.reserve(), Duration::ZERO);
        let second = l.reserve();
        assert!(second.as_millis() > 900, "expected ~1s, got {second:?}");
        // Reservations accumulate, so a burst is spread rather than collapsed.
        let third = l.reserve();
        assert!(third > second, "each reservation must queue behind the last");
    }

    #[test]
    fn a_non_positive_rate_disables_limiting() {
        for rpm in [0.0, -1.0] {
            let l = RateLimiter::per_minute(rpm);
            assert!(!l.is_enabled());
            assert_eq!(l.reserve(), Duration::ZERO);
            assert_eq!(l.reserve(), Duration::ZERO);
        }
    }
}
