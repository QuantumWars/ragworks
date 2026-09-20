//! Embedders, with the plumbing a real provider needs.
//!
//! Two implementations: [`Hashing`] runs offline and deterministically, and
//! [`OpenAiCompatible`] talks to any endpoint speaking OpenAI's `/embeddings`
//! shape. The parts that usually get written badly -- retry, rate limiting,
//! batching and cost accounting -- live here rather than in each caller.

pub mod hash;
pub mod http;
pub mod limit;
pub mod openai;
pub mod retry;

pub use hash::{Hashing, HashingConfig};
pub use limit::RateLimiter;
pub use openai::{OpenAiCompatible, OpenAiConfig, Usage};
use ragworks_core::{Embedder, Registry};
pub use retry::RetryPolicy;

pub fn registry() -> Registry<dyn Embedder> {
    let mut r = Registry::<dyn Embedder>::new("embedder");
    r.register::<Hashing>(|c| Box::new(c)).expect("builtin");
    r.register::<OpenAiCompatible>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use ragworks_core::Embedder;

    use super::*;
    use crate::http::MockHttp;

    // ------------------------------------------------------------- hashing

    fn hashing(dim: usize) -> Box<dyn Embedder> {
        registry().build("hashing", &serde_json::json!({"dim": dim})).unwrap()
    }

    fn vec_of(e: &dyn Embedder, text: &str) -> Vec<f32> {
        let mut v = Vec::new();
        e.embed(&[text], &mut v).unwrap();
        v
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn registry_exposes_both_embedders_with_schemas() {
        let r = registry();
        assert_eq!(r.names(), vec!["hashing", "openai"]);
        for n in r.names() {
            assert!(r.schema(n).unwrap()["properties"].is_object(), "{n} has no schema");
        }
    }

    #[test]
    fn hashing_is_deterministic_across_instances() {
        // Not merely stable within one process: `DefaultHasher` is explicitly
        // not portable, and an embedding that changed between runs would
        // silently invalidate every cached vector.
        assert_eq!(vec_of(&*hashing(64), "hello world"), vec_of(&*hashing(64), "hello world"));
    }

    #[test]
    fn vectors_are_unit_length_and_the_right_width() {
        let e = hashing(128);
        let v = vec_of(&*e, "the quick brown fox");
        assert_eq!(v.len(), 128);
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5, "not normalised: {}", cosine(&v, &v));
    }

    #[test]
    fn lexical_overlap_produces_real_similarity() {
        // Note the premise: these differ by one literal token. Hashing has no
        // notion of meaning, and on paraphrases that share few words it can
        // rank a pair *below* an unrelated one -- see the module docs.
        let e = hashing(512);
        let a = vec_of(&*e, "database index btree sorted column");
        let near = vec_of(&*e, "database index btree sorted rows");
        let far = vec_of(&*e, "volcanic activity iceland glacier");
        assert!(cosine(&a, &near) > cosine(&a, &far), "overlap must beat unrelated text");
    }

    #[test]
    fn an_empty_batch_and_empty_text_are_handled() {
        let e = hashing(32);
        let mut out = Vec::new();
        e.embed(&[], &mut out).unwrap();
        assert!(out.is_empty());
        e.embed(&[""], &mut out).unwrap();
        assert_eq!(out.len(), 32, "an empty string still occupies a row");
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn batches_are_written_contiguously_row_major() {
        let e = hashing(16);
        let mut out = Vec::new();
        e.embed(&["alpha", "beta", "gamma"], &mut out).unwrap();
        assert_eq!(out.len(), 48);
        assert_eq!(&out[0..16], vec_of(&*e, "alpha").as_slice());
        assert_eq!(&out[32..48], vec_of(&*e, "gamma").as_slice());
    }

    // -------------------------------------------------------------- openai

    fn body(vectors: &[&[f32]]) -> String {
        let data: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| serde_json::json!({"index": i, "embedding": v}))
            .collect();
        serde_json::json!({
            "data": data,
            "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.00000008}
        })
        .to_string()
    }

    fn cfg(env: &str, dim: usize, max_batch: usize) -> OpenAiConfig {
        OpenAiConfig {
            model: "test/model".into(),
            dim,
            api_key_env: env.into(),
            max_batch,
            retry: RetryPolicy { max_attempts: 3, base_delay_ms: 0, max_delay_ms: 0 },
            ..Default::default()
        }
    }

    #[test]
    fn a_missing_api_key_names_the_variable_and_refuses_to_take_one_from_config() {
        let err = OpenAiCompatible::with_http(
            cfg("RAGWORKS_ABSENT_KEY_1", 3, 8),
            Box::new(MockHttp::ok("{}")),
        )
        .err()
        .unwrap();
        let m = err.to_string();
        assert!(m.contains("RAGWORKS_ABSENT_KEY_1"), "{m}");
        assert!(m.contains("never from configuration"), "{m}");
    }

    #[test]
    fn embeddings_are_parsed_and_usage_accumulated() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_2", "k") };
        let http = MockHttp::ok(&body(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]));
        let e = OpenAiCompatible::with_http(cfg("RAGWORKS_TEST_KEY_2", 3, 8), Box::new(http)).unwrap();
        let mut out = Vec::new();
        e.embed(&["a", "b"], &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let u = e.usage();
        assert_eq!((u.requests, u.prompt_tokens), (1, 4));
        assert!((u.usd - 0.00000008).abs() < 1e-12);
    }

    #[test]
    fn out_of_order_responses_are_reordered_by_index() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_3", "k") };
        let raw = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [9.0, 9.0]},
                {"index": 0, "embedding": [1.0, 1.0]}
            ],
            "usage": {}
        })
        .to_string();
        let e = OpenAiCompatible::with_http(
            cfg("RAGWORKS_TEST_KEY_3", 2, 8),
            Box::new(MockHttp::ok(&raw)),
        )
        .unwrap();
        let mut out = Vec::new();
        e.embed(&["first", "second"], &mut out).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 9.0, 9.0], "array order must not be trusted");
    }

    #[test]
    fn a_dimension_mismatch_is_caught_before_it_reaches_an_index() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_4", "k") };
        let e = OpenAiCompatible::with_http(
            cfg("RAGWORKS_TEST_KEY_4", 1536, 8),
            Box::new(MockHttp::ok(&body(&[&[1.0, 2.0, 3.0]]))),
        )
        .unwrap();
        let err = e.embed(&["a"], &mut Vec::new()).err().unwrap();
        assert!(err.to_string().contains("1536"), "{err}");
    }

    #[test]
    fn a_rate_limit_is_retried_and_then_succeeds() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_5", "k") };
        let http = MockHttp::new(vec![
            Ok((429, r#"{"error":{"message":"slow down"}}"#.into())),
            Ok((200, body(&[&[1.0, 1.0]]))),
        ]);
        let e = OpenAiCompatible::with_http(cfg("RAGWORKS_TEST_KEY_5", 2, 8), Box::new(http)).unwrap();
        let mut out = Vec::new();
        e.embed(&["a"], &mut out).unwrap();
        assert_eq!(out, vec![1.0, 1.0]);
    }

    #[test]
    fn an_auth_failure_fails_immediately_with_the_providers_message() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_6", "k") };
        let http = MockHttp::new(vec![Ok((401, r#"{"error":{"message":"invalid key"}}"#.into()))]);
        let e = OpenAiCompatible::with_http(cfg("RAGWORKS_TEST_KEY_6", 2, 8), Box::new(http)).unwrap();
        let err = e.embed(&["a"], &mut Vec::new()).err().unwrap();
        assert!(err.to_string().contains("invalid key"), "{err}");
        assert!(!err.is_retryable());
    }

    #[test]
    fn oversized_input_is_split_into_batches_automatically() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_7", "k") };
        // The mock replies with two vectors each time, so a batch size of 2
        // over 6 inputs must produce exactly three requests.
        let http = MockHttp::new(vec![Ok((200, body(&[&[1.0], &[2.0]])))]);
        let e = OpenAiCompatible::with_http(cfg("RAGWORKS_TEST_KEY_7", 1, 2), Box::new(http)).unwrap();
        let mut out = Vec::new();
        e.embed(&["a", "b", "c", "d", "e", "f"], &mut out).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(e.usage().requests, 3, "a caller that forgets to chunk must still work");
    }

    #[test]
    fn a_short_response_is_an_error_not_a_silent_truncation() {
        unsafe { std::env::set_var("RAGWORKS_TEST_KEY_8", "k") };
        let e = OpenAiCompatible::with_http(
            cfg("RAGWORKS_TEST_KEY_8", 1, 8),
            Box::new(MockHttp::ok(&body(&[&[1.0]]))),
        )
        .unwrap();
        let err = e.embed(&["a", "b"], &mut Vec::new()).err().unwrap();
        assert!(err.to_string().contains("asked for 2"), "{err}");
    }
}
