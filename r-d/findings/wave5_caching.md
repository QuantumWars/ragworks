# Wave 5 — response caching, and a bug the measurement exposed

The first item from `CLAUDE.md`'s optimization list, plus a correction to the
list itself.

## The priority was wrong as written

`CLAUDE.md` originally listed "parallelize `search_many`" first. Its own section 3
contradicted that: BM25 search has a p95 of **0.15 ms** against 614 ms for typed
reranking and 4.3 s for LLM query rewriting — a 43,000x spread. Parallelizing a
0.15 ms stage saves nothing that the next stage does not immediately dwarf.

Caching and provider concurrency are the real first two. The list has been
corrected and the reasoning recorded in it, so the same mistake is not made again.

## What was built

A content-addressed cache that decorates the `HttpPost` seam. Because HTTP was
already behind a trait, embedders, judges and query transforms all gained
caching with **no change to any of them**.

Design decisions worth keeping:

- **The key excludes headers.** Headers carry the bearer token; keying on them
  would put a secret in a filename and make key rotation silently invalidate
  every entry. A test asserts no key material reaches the cache directory.
- **Failures are never cached.** Caching a 429 makes a transient rate limit
  permanent; caching a 401 survives fixing the key.
- **Writes are atomic** via a unique temp file and rename, so a crash cannot
  leave a half-written entry that later reads as a valid response.

## F15 — the limiter was throttling the cache

The first measurement showed only a 2.6x speedup, with warm p95 at 1207 ms —
suspiciously close to the 1 s interval implied by `requests_per_minute: 60`.

The cause: all three components called `limiter.acquire()` **before**
`http.post()`. A cache hit consumed no quota and still waited for one.

Rate limiting is now a transport layer *below* the cache:

```
CachedHttp  ->  RateLimited  ->  UreqClient
```

That doubled the benefit, removed duplicated limiter logic from three
components, and is pinned by a test that fails if the layers are ever reordered.

## Measured

40 HotpotQA queries through a HyDE transform, identical work both times:

| | cold | warm |
|---|---|---|
| **median per-query latency** | 1520 ms | **7 ms** |
| p95 | 7332 ms | 3246 ms |
| total query time | 101.9 s | **11.8 s** |
| first three queries | 2551, 1045, 1348 ms | **8, 7, 8 ms** |

**217x at the median.** Isolating the transform from dense retrieval, a cached
call costs **0.0 ms** — the remaining 7 ms is the embedding lookup. Cache size
for the whole run: 46 entries, 44 KB.

## F16 — a fallback chain leaves gaps in the cache

Warm p95 stayed at 3246 ms while p50 was 7 ms. Three of forty queries missed.

The key includes the model name, because a different model is a different
response. A request the first model failed and the second answered is cached
only under the *second* model. The next run tries the first model again, misses,
and goes to the network.

This is correct rather than broken — keying on the logical request would serve a
response from a model the caller did not ask for — but it means **a warm run is
not automatically a 100% hit rate**, and a p95 can stay high while a p50
collapses. Recorded so the next person reading these numbers is not confused by
them.

## Caveat

Wall-clock comparisons here include sentence-transformers model loading and
corpus encoding (6-10 s), which caching does not touch. The per-query medians
are the honest number; the totals flatter the cache slightly less than the
speedup ratio suggests.
