//! Query construction and reformulation.
//!
//! Retrieval fails in two ways, and reranking only fixes one of them. If the
//! right document is in the shortlist but badly ordered, a reranker fixes it.
//! If it is not in the shortlist at all, nothing downstream can help — the
//! query has to change.
//!
//! Wave-1 measurement on multi-hop questions: the first supporting document was
//! retrieved at rank 0, the second at rank 88. That is the failure this crate
//! exists for.

pub mod chat;
pub mod llm;
pub mod offline;

pub use chat::{Chat, ChatConfig};
pub use llm::{Decompose, DecomposeConfig, Hyde, HydeConfig, MultiQuery, MultiQueryConfig};
pub use offline::{Identity, IdentityConfig, Rm3, Rm3Config};
use ragworks_core::{QueryTransform, Registry};

pub fn registry() -> Registry<dyn QueryTransform> {
    let mut r = Registry::<dyn QueryTransform>::new("query_transform");
    r.register::<Identity>(|c| Box::new(c)).expect("builtin");
    r.register::<Rm3>(|c| Box::new(c)).expect("builtin");
    r.register::<MultiQuery>(|c| Box::new(c)).expect("builtin");
    r.register::<Hyde>(|c| Box::new(c)).expect("builtin");
    r.register::<Decompose>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use ragworks_core::QueryTransform;
    use ragworks_net::MockHttp;

    use super::*;
    use crate::chat::parse_lines;

    fn apply(t: &dyn QueryTransform, q: &str, fb: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        t.transform(q, fb, &mut out).unwrap();
        out
    }

    #[test]
    fn registry_exposes_every_transform_with_a_schema() {
        let r = registry();
        assert_eq!(r.names(), vec!["decompose", "hyde", "identity", "multi_query", "rm3"]);
        for n in r.names() {
            assert!(r.schema(n).is_ok(), "{n} has no schema");
        }
    }

    #[test]
    fn identity_returns_the_query_unchanged() {
        let t = registry().default_build("identity").unwrap();
        assert_eq!(apply(&*t, "why are writes slow", &["ignored"]), vec!["why are writes slow"]);
        assert!(!t.uses_feedback());
    }

    // -------------------------------------------------------------- rm3

    #[test]
    fn rm3_appends_distinctive_terms_from_the_feedback() {
        let t = registry().default_build("rm3").unwrap();
        assert!(t.uses_feedback());
        let out = apply(
            &*t,
            "slow writes",
            &[
                "Indexes are btree structures that must be updated on every insert.",
                "Each btree insert rebalances nodes, which costs additional work.",
            ],
        );
        assert_eq!(out.len(), 1);
        assert!(out[0].starts_with("slow writes"), "the original must survive: {:?}", out[0]);
        assert!(out[0].contains("btree"), "a term in both feedback docs should win: {:?}", out[0]);
    }

    #[test]
    fn rm3_without_feedback_is_the_identity() {
        // The same transform has to serve both passes of a two-pass pipeline.
        let t = registry().default_build("rm3").unwrap();
        assert_eq!(apply(&*t, "slow writes", &[]), vec!["slow writes"]);
    }

    #[test]
    fn rm3_does_not_repeat_query_terms_or_add_short_ones() {
        let t = registry().default_build("rm3").unwrap();
        let out = apply(&*t, "btree", &["the btree is in a table and of the rows"]);
        let added: Vec<&str> = out[0].split_whitespace().skip(1).collect();
        assert!(!added.contains(&"btree"), "query terms must not be re-added: {added:?}");
        for stop in ["the", "is", "in", "a", "of"] {
            assert!(!added.contains(&stop), "{stop:?} should be below the length floor: {added:?}");
        }
    }

    #[test]
    fn rm3_rejects_a_zero_feedback_window() {
        assert!(registry().build("rm3", &serde_json::json!({"feedback_docs": 0})).is_err());
    }

    // ------------------------------------------------------- line parsing

    #[test]
    fn list_replies_are_cleaned_of_decoration() {
        let reply = "Here are the rewrites:\n1. why writes are slow\n- indexed write cost\n* \"btree update overhead\"\n\n3) insert latency causes\n";
        assert_eq!(
            parse_lines(reply, 10),
            vec![
                "why writes are slow",
                "indexed write cost",
                "btree update overhead",
                "insert latency causes",
            ],
            "numbering, bullets, quotes and the preamble line should all go"
        );
    }

    #[test]
    fn line_parsing_respects_the_requested_maximum() {
        assert_eq!(parse_lines("one\ntwo\nthree\nfour", 2).len(), 2);
    }

    // ---------------------------------------------------- chat + llm paths

    fn chat_with(responses: Vec<ragworks_core::Result<(u16, String)>>) -> Chat {
        unsafe { std::env::set_var("RAGWORKS_QUERY_KEY", "k") };
        Chat::with_http(
            ChatConfig {
                api_key_env: "RAGWORKS_QUERY_KEY".into(),
                models: vec!["model-a".into(), "model-b".into()],
                retry: ragworks_net::RetryPolicy {
                    max_attempts: 1,
                    base_delay_ms: 0,
                    max_delay_ms: 0,
                },
                ..Default::default()
            },
            Box::new(MockHttp::new(responses)),
        )
        .unwrap()
    }

    fn completion(text: &str) -> String {
        serde_json::json!({"choices": [{"message": {"content": text}}]}).to_string()
    }

    #[test]
    fn the_fallback_chain_moves_on_when_a_model_fails() {
        // Free tiers take individual models out of service without notice, so a
        // single pinned model is not reliable enough to run an experiment on.
        let chat = chat_with(vec![
            Ok((503, r#"{"error":{"message":"model offline"}}"#.into())),
            Ok((200, completion("second model answered"))),
        ]);
        assert_eq!(chat.complete("s", "u").unwrap(), "second model answered");
    }

    #[test]
    fn an_empty_completion_counts_as_a_failure_of_that_model() {
        // Reasoning models sometimes spend the whole budget before emitting
        // content; the chain should move on rather than return nothing.
        let chat = chat_with(vec![
            Ok((200, completion("   "))),
            Ok((200, completion("real answer"))),
        ]);
        assert_eq!(chat.complete("s", "u").unwrap(), "real answer");
    }

    #[test]
    fn every_model_failing_surfaces_the_last_error() {
        let chat = chat_with(vec![Ok((401, r#"{"error":{"message":"bad key"}}"#.into()))]);
        assert!(chat.complete("s", "u").err().unwrap().to_string().contains("bad key"));
    }

    #[test]
    fn multi_query_keeps_the_original_and_adds_paraphrases() {
        let chat = chat_with(vec![Ok((
            200,
            completion("1. indexed write cost\n2. btree update overhead"),
        ))]);
        let t = MultiQuery::with_chat(chat, 3, true);
        let out = apply(&t, "why are writes slow", &[]);
        assert_eq!(out[0], "why are writes slow", "the original must be retrieved too");
        assert_eq!(out.len(), 3);
        assert!(out.contains(&"btree update overhead".to_string()));
    }

    #[test]
    fn a_paraphrase_identical_to_the_query_is_not_duplicated() {
        let chat = chat_with(vec![Ok((200, completion("Why Are Writes Slow\nindexed write cost")))]);
        let out = apply(&MultiQuery::with_chat(chat, 3, true), "why are writes slow", &[]);
        assert_eq!(out.len(), 2, "case-insensitive duplicate should be dropped: {out:?}");
    }

    #[test]
    fn hyde_searches_with_the_passage_not_the_question() {
        let chat = chat_with(vec![Ok((
            200,
            completion("An index is a btree. Every insert must update it, so writes cost more."),
        ))]);
        let out = apply(&Hyde::with_chat(chat, false), "why are writes slow", &[]);
        assert_eq!(out.len(), 1);
        assert!(out[0].contains("btree"), "{out:?}");
        assert!(!out[0].contains("why are writes slow"));
    }

    #[test]
    fn a_transient_outage_degrades_to_the_original_query() {
        // A reformulation is an optimisation. Retrieving with the plain
        // question is worse than a good rewrite and far better than failing.
        let chat = chat_with(vec![Ok((503, r#"{"error":{"message":"overloaded"}}"#.into()))]);
        let out = apply(&Hyde::with_chat(chat, false), "why are writes slow", &[]);
        assert_eq!(out, vec!["why are writes slow"]);
    }

    #[test]
    fn a_misconfiguration_is_not_hidden_behind_degraded_results() {
        // A bad key must not present itself as poor retrieval quality; that is
        // the hardest class of fault to diagnose.
        let chat = chat_with(vec![Ok((401, r#"{"error":{"message":"bad key"}}"#.into()))]);
        let mut out = Vec::new();
        let err = Hyde::with_chat(chat, false)
            .transform("why are writes slow", &[], &mut out)
            .err()
            .unwrap();
        assert!(err.to_string().contains("bad key"), "{err}");
    }

    #[test]
    fn decompose_emits_sub_questions_alongside_the_original() {
        let chat = chat_with(vec![Ok((
            200,
            completion("What is a btree?\nHow does an insert update a btree?"),
        ))]);
        let out = apply(&Decompose::with_chat(chat, 3, true), "why are writes slow", &[]);
        assert_eq!(out.len(), 3);
        assert!(out[1].starts_with("What is a btree"));
    }
}
