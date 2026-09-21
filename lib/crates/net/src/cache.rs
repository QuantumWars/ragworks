//! A content-addressed response cache, as a transport decorator.
//!
//! This wraps any [`HttpPost`], so embedders, judges and query transforms all
//! gain caching without a line of change in any of them. That is the payoff of
//! having put HTTP behind a trait rather than calling `ureq` directly.
//!
//! **Why this matters more than any retrieval optimisation here.** Measured p95
//! in this repository: BM25 search 0.1 ms, dense retrieval 8 ms, typed
//! reranking 614 ms, LLM query rewriting 4.3 s. A full evaluation of one LLM
//! transform took 436 s of wall clock against 13 s for the baseline. Re-running
//! it after a metric change currently pays that again in full, in both time and
//! money. Cached, it costs nothing.
//!
//! Measured on 40 HotpotQA queries through a HyDE transform: median per-query
//! latency **1520 ms cold, 7 ms warm** -- a 217x reduction, with the cached
//! call itself costing 0.0 ms and the remaining 7 ms being dense retrieval.
//! Total query time fell from 101.9 s to 11.8 s.
//!
//! ## A fallback chain leaves gaps in the cache
//!
//! The key includes the model name, because a different model is a different
//! response. So a request that the first model failed and the second answered
//! caches only under the *second* model. The next run tries the first model
//! again, misses, and goes to the network -- correctly, but it means a warm run
//! is not always a 100% hit rate. Measured: 37 of 40 queries hit; the three
//! misses were queries the chain had fallen through on.
//!
//! Keying on the logical request instead would be wrong: it would serve a
//! response from a model the caller did not ask for.
//!
//! ## The key deliberately excludes headers
//!
//! Headers carry the `Authorization` bearer token. Keying on them would put a
//! secret into a filename, and rotating a key would silently invalidate every
//! entry for no reason -- the same request returns the same response whoever
//! asked. The key is `(url, body)` and nothing else.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ragworks_core::Result;

use crate::http::HttpPost;

/// What a cache did, for reporting.
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bytes_served: AtomicU64,
    pub writes: AtomicU64,
}

impl CacheStats {
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.bytes_served.load(Ordering::Relaxed),
            self.writes.load(Ordering::Relaxed),
        )
    }

    pub fn hit_rate(&self) -> f64 {
        let (h, m, _, _) = self.snapshot();
        if h + m == 0 { 0.0 } else { h as f64 / (h + m) as f64 }
    }
}

pub struct CachedHttp {
    inner: Box<dyn HttpPost>,
    dir: PathBuf,
    pub stats: CacheStats,
}

impl CachedHttp {
    /// Wrap `inner`, storing entries under `dir`.
    pub fn new(inner: Box<dyn HttpPost>, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { inner, dir, stats: CacheStats::default() })
    }

    /// Cache under the OS cache directory, or a local fallback.
    pub fn with_default_dir(inner: Box<dyn HttpPost>, namespace: &str) -> Result<Self> {
        let base = std::env::var("RAGWORKS_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".cache"))
                    .unwrap_or_else(|_| std::env::temp_dir())
                    .join("ragworks")
            });
        Self::new(inner, base.join(namespace))
    }

    /// Key on request identity only. See the module docs on headers.
    fn key(url: &str, body: &str) -> String {
        let mut h = blake3::Hasher::new();
        h.update(url.as_bytes());
        h.update(b"\0");
        h.update(body.as_bytes());
        h.finalize().to_hex().to_string()
    }

    /// Two hex characters of fan-out. A flat directory with a hundred thousand
    /// entries is slow to list and unpleasant on some filesystems.
    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(&key[..2]).join(&key[2..])
    }

    pub fn read(&self, key: &str) -> Option<String> {
        fs::read_to_string(self.path_for(key)).ok()
    }

    pub fn write(&self, key: &str, body: &str) -> Result<()> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Write to a unique temporary file and rename, so a crash or a
        // concurrent writer can never leave a half-written entry that later
        // reads as a valid response.
        let tmp = path.with_extension(format!("tmp{}", unique_suffix(key)));
        fs::write(&tmp, body)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

fn unique_suffix(key: &str) -> u64 {
    let mut h = DefaultHasher::new();
    std::process::id().hash(&mut h);
    std::thread::current().id().hash(&mut h);
    key.hash(&mut h);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
        .hash(&mut h);
    h.finish()
}

impl HttpPost for CachedHttp {
    fn post(&self, url: &str, headers: &[(&str, String)], body: &str) -> Result<(u16, String)> {
        let key = Self::key(url, body);

        if let Some(hit) = self.read(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes_served.fetch_add(hit.len() as u64, Ordering::Relaxed);
            return Ok((200, hit));
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);

        let (status, text) = self.inner.post(url, headers, body)?;

        // Only successes are stored. Caching a 429 would make a transient rate
        // limit permanent, and caching a 401 would survive fixing the key.
        if (200..300).contains(&status) {
            // A cache that cannot write is slow, not broken. Failing the
            // request would turn a full disk into a failed experiment.
            if self.write(&key, &text).is_ok() {
                self.stats.writes.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok((status, text))
    }
}

/// Remove every entry. Returns how many were deleted.
pub fn clear(dir: impl AsRef<Path>) -> Result<usize> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(0);
    }
    let mut n = 0;
    for shard in fs::read_dir(dir)? {
        let shard = shard?;
        if shard.file_type()?.is_dir() {
            for entry in fs::read_dir(shard.path())? {
                fs::remove_file(entry?.path())?;
                n += 1;
            }
        }
    }
    Ok(n)
}

/// Total bytes held under `dir`.
pub fn size_bytes(dir: impl AsRef<Path>) -> Result<u64> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(0);
    }
    let mut total = 0;
    for shard in fs::read_dir(dir)? {
        let shard = shard?;
        if shard.file_type()?.is_dir() {
            for entry in fs::read_dir(shard.path())? {
                total += entry?.metadata()?.len();
            }
        }
    }
    Ok(total)
}

impl std::fmt::Debug for CachedHttp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (h, m, b, w) = self.stats.snapshot();
        f.debug_struct("CachedHttp")
            .field("dir", &self.dir)
            .field("hits", &h)
            .field("misses", &m)
            .field("writes", &w)
            .field("bytes_served", &b)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use ragworks_core::Error;

    use super::*;
    use crate::http::MockHttp;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("ragworks_cache_{name}_{}_{:?}", std::process::id(), std::thread::current().id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_repeated_request_is_served_without_touching_the_transport() {
        let dir = tmpdir("hit");
        let inner = MockHttp::new(vec![Ok((200, "first".into()))]);
        let c = CachedHttp::new(Box::new(inner), &dir).unwrap();

        assert_eq!(c.post("u", &[], "body").unwrap(), (200, "first".to_string()));
        assert_eq!(c.post("u", &[], "body").unwrap(), (200, "first".to_string()));

        let (hits, misses, _, writes) = c.stats.snapshot();
        assert_eq!((hits, misses, writes), (1, 1, 1));
        assert!((c.stats.hit_rate() - 0.5).abs() < 1e-9);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_different_body_is_a_different_entry() {
        let dir = tmpdir("body");
        let c = CachedHttp::new(
            Box::new(MockHttp::new(vec![Ok((200, "a".into())), Ok((200, "b".into()))])),
            &dir,
        )
        .unwrap();
        assert_eq!(c.post("u", &[], "one").unwrap().1, "a");
        assert_eq!(c.post("u", &[], "two").unwrap().1, "b");
        assert_eq!(c.stats.snapshot().1, 2, "both should miss");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rotating_an_api_key_does_not_invalidate_the_cache() {
        // The key excludes headers: the same request returns the same response
        // whoever asked, and a bearer token must never reach a filename.
        let dir = tmpdir("headers");
        let c = CachedHttp::new(Box::new(MockHttp::new(vec![Ok((200, "shared".into()))])), &dir)
            .unwrap();
        c.post("u", &[("Authorization", "Bearer OLD".into())], "b").unwrap();
        let second = c.post("u", &[("Authorization", "Bearer NEW".into())], "b").unwrap();
        assert_eq!(second.1, "shared");
        assert_eq!(c.stats.snapshot().0, 1, "the second call must hit");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_secret_reaches_the_cache_directory() {
        let dir = tmpdir("secret");
        let c = CachedHttp::new(Box::new(MockHttp::ok("response body")), &dir).unwrap();
        c.post("https://api/x", &[("Authorization", "Bearer sk-SECRET-VALUE".into())], "b")
            .unwrap();

        let mut inspected = 0;
        for shard in fs::read_dir(&dir).unwrap() {
            let shard = shard.unwrap();
            assert!(!shard.file_name().to_string_lossy().contains("SECRET"));
            for entry in fs::read_dir(shard.path()).unwrap() {
                let entry = entry.unwrap();
                assert!(!entry.file_name().to_string_lossy().contains("SECRET"));
                let content = fs::read_to_string(entry.path()).unwrap();
                assert!(!content.contains("SECRET"), "a key leaked into a cache entry");
                inspected += 1;
            }
        }
        assert_eq!(inspected, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failures_are_never_cached() {
        // Caching a 429 would make a transient rate limit permanent; caching a
        // 401 would survive fixing the key.
        for status in [429, 401, 500] {
            let dir = tmpdir(&format!("fail{status}"));
            let c = CachedHttp::new(
                Box::new(MockHttp::new(vec![
                    Ok((status, "nope".into())),
                    Ok((200, "recovered".into())),
                ])),
                &dir,
            )
            .unwrap();
            assert_eq!(c.post("u", &[], "b").unwrap().0, status);
            assert_eq!(c.post("u", &[], "b").unwrap().1, "recovered");
            assert_eq!(c.stats.snapshot().0, 0, "a {status} must not be served from cache");
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn entries_survive_a_new_process_and_can_be_cleared() {
        let dir = tmpdir("persist");
        {
            let c = CachedHttp::new(Box::new(MockHttp::ok("persisted")), &dir).unwrap();
            c.post("u", &[], "b").unwrap();
        }
        // A fresh wrapper over a transport that would fail if consulted.
        let c2 = CachedHttp::new(
            Box::new(MockHttp::new(vec![Err(Error::provider("mock", Some(500), "must not call"))])),
            &dir,
        )
        .unwrap();
        assert_eq!(c2.post("u", &[], "b").unwrap().1, "persisted");
        assert!(size_bytes(&dir).unwrap() > 0);
        assert_eq!(clear(&dir).unwrap(), 1);
        assert_eq!(size_bytes(&dir).unwrap(), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
