//! Rerankers and verifiers.
//!
//! Each trait has a model-backed implementation and an offline baseline. The
//! baselines are not filler: a judge that beats no baseline has not been shown
//! to do anything, and both of these run with no network and no cost.

pub mod jev;
pub mod lexical;

pub use jev::{Jev, JevConfig};
pub use lexical::{Coverage, CoverageConfig, Lexical, LexicalConfig};
use ragworks_core::{Registry, Reranker, Verifier};

pub fn reranker_registry() -> Registry<dyn Reranker> {
    let mut r = Registry::<dyn Reranker>::new("reranker");
    r.register::<Lexical>(|c| Box::new(c)).expect("builtin");
    r.register::<Jev>(|c| Box::new(c)).expect("builtin");
    r
}

pub fn verifier_registry() -> Registry<dyn Verifier> {
    let mut r = Registry::<dyn Verifier>::new("verifier");
    r.register::<Coverage>(|c| Box::new(c)).expect("builtin");
    r.register::<Jev>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use ragworks_core::{Reranker, Support, Verifier};
    use ragworks_net::MockHttp;

    use super::*;

    fn scores(r: &dyn Reranker, q: &str, c: &[&str]) -> Vec<f32> {
        let mut out = Vec::new();
        r.rerank(q, c, &mut out).unwrap();
        out
    }

    #[test]
    fn registries_expose_both_implementations_with_schemas() {
        assert_eq!(reranker_registry().names(), vec!["jev", "lexical"]);
        assert_eq!(verifier_registry().names(), vec!["coverage", "jev"]);
        assert!(reranker_registry().schema("lexical").unwrap()["properties"].is_object());
        assert!(verifier_registry().schema("coverage").unwrap()["properties"].is_object());
    }

    // --------------------------------------------------- lexical reranker

    #[test]
    fn a_candidate_matching_a_rare_term_outranks_one_matching_a_common_one() {
        let r = reranker_registry().default_build("lexical").unwrap();
        let cands = [
            "the system stores the data in the table",
            "the aardvark stores the data in the table",
            "the system stores the data in the table",
        ];
        let s = scores(&*r, "aardvark system", &cands);
        assert!(s[1] > s[0], "the rare term should dominate: {s:?}");
    }

    #[test]
    fn one_score_is_produced_per_candidate_in_order() {
        let r = reranker_registry().default_build("lexical").unwrap();
        assert_eq!(scores(&*r, "q", &["a b", "c d", "e f"]).len(), 3);
        assert!(scores(&*r, "q", &[]).is_empty());
    }

    #[test]
    fn a_candidate_sharing_nothing_with_the_query_scores_zero() {
        let r = reranker_registry().default_build("lexical").unwrap();
        let s = scores(&*r, "volcano", &["database index btree", "volcano eruption"]);
        assert_eq!(s[0], 0.0);
        assert!(s[1] > 0.0);
    }

    // -------------------------------------------------- coverage verifier

    fn verdict(v: &dyn Verifier, q: &str, e: &[&str]) -> ragworks_core::Verdict {
        v.verify(q, e).unwrap()
    }

    #[test]
    fn full_term_coverage_supports_and_a_missing_entity_does_not() {
        let v = verifier_registry().default_build("coverage").unwrap();
        let covered = verdict(&*v, "who directed Sinister", &["Sinister was directed by Scott Derrickson"]);
        assert_eq!(covered.support, Support::Supports);

        // The second hop is absent from the evidence, so the question cannot
        // have been answered from it.
        let missing = verdict(&*v, "what nationality was Ed Wood", &["Sinister is a horror film"]);
        assert_eq!(missing.support, Support::Insufficient);
    }

    #[test]
    fn coverage_never_claims_to_detect_contradiction() {
        // Term matching cannot see that evidence refutes a premise, and saying
        // otherwise would misreport what the baseline does.
        let v = verifier_registry().default_build("coverage").unwrap();
        for (q, e) in [
            ("is the sky green", &["the sky is green"][..]),
            ("is the sky green", &["the sky is blue not green"][..]),
        ] {
            assert_ne!(verdict(&*v, q, e).support, Support::Refutes);
        }
    }

    #[test]
    fn every_verdict_records_the_window_it_judged() {
        // "insufficient over 1" and "insufficient over 8" are different claims.
        let v = verifier_registry().default_build("coverage").unwrap();
        assert_eq!(verdict(&*v, "anything", &["a", "b", "c"]).window, 3);
        assert_eq!(verdict(&*v, "anything", &[]).window, 0);
    }

    #[test]
    fn no_evidence_is_insufficient_without_asking_anything() {
        let v = verifier_registry().default_build("coverage").unwrap();
        assert_eq!(verdict(&*v, "q", &[]).support, Support::Insufficient);
    }

    // ---------------------------------------------------------------- jev

    fn jev_with(body: &str) -> Jev {
        unsafe { std::env::set_var("RAGWORKS_JUDGE_KEY", "k") };
        Jev::with_http(
            JevConfig {
                api_key_env: "RAGWORKS_JUDGE_KEY".into(),
                retry: ragworks_net::RetryPolicy {
                    max_attempts: 2,
                    base_delay_ms: 0,
                    max_delay_ms: 0,
                },
                ..Default::default()
            },
            Box::new(MockHttp::ok(body)),
        )
        .unwrap()
    }

    #[test]
    fn jev_reranking_reads_one_probability_per_candidate() {
        let j = jev_with(r#"{"answers":{"c0":{"noul":0.11},"c1":{"noul":0.93}},"usage":{}}"#);
        assert_eq!(scores(&j, "q", &["first", "second"]), vec![0.11, 0.93]);
    }

    #[test]
    fn a_missing_answer_costs_one_candidate_not_the_query() {
        let j = jev_with(r#"{"answers":{"c0":{"noul":0.5}},"usage":{}}"#);
        assert_eq!(scores(&j, "q", &["a", "b"]), vec![0.5, 0.0]);
    }

    #[test]
    fn jev_verification_parses_all_three_outcomes() {
        for (choice, expected) in [
            ("supports", Support::Supports),
            ("refutes", Support::Refutes),
            ("insufficient", Support::Insufficient),
        ] {
            let body = format!(
                r#"{{"answers":{{"verdict":{{"choice":"{choice}","confidence":0.87}}}},"usage":{{}}}}"#
            );
            let v = verdict(&jev_with(&body), "q", &["e1", "e2"]);
            assert_eq!(v.support, expected);
            assert!((v.confidence - 0.87).abs() < 1e-6);
            assert_eq!(v.window, 2);
        }
    }

    #[test]
    fn an_unrecognised_choice_falls_back_to_insufficient() {
        // The schema makes this impossible from the model's side, but a proxy
        // or a version change could still produce it, and guessing "supports"
        // would be the dangerous direction to be wrong in.
        let j = jev_with(r#"{"answers":{"verdict":{"choice":"maybe","confidence":0.5}}}"#);
        assert_eq!(verdict(&j, "q", &["e"]).support, Support::Insufficient);
    }

    #[test]
    fn empty_input_short_circuits_before_spending_anything() {
        unsafe { std::env::set_var("RAGWORKS_JUDGE_KEY_2", "k") };
        // A transport that fails on any call: reaching it would be the bug.
        let http = MockHttp::new(vec![Err(ragworks_core::Error::provider(
            "mock",
            Some(500),
            "must not be called",
        ))]);
        let j = Jev::with_http(
            JevConfig { api_key_env: "RAGWORKS_JUDGE_KEY_2".into(), ..Default::default() },
            Box::new(http),
        )
        .unwrap();
        assert!(scores(&j, "q", &[]).is_empty());
        assert_eq!(verdict(&j, "q", &[]).support, Support::Insufficient);
    }

    #[test]
    fn a_missing_api_key_names_the_variable() {
        let err = Jev::with_http(
            JevConfig { api_key_env: "RAGWORKS_JUDGE_ABSENT".into(), ..Default::default() },
            Box::new(MockHttp::ok("{}")),
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("RAGWORKS_JUDGE_ABSENT"), "{err}");
    }
}
