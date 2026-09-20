//! Indexes and rank fusion.
//!
//! Two registries, because lexical and dense retrieval take different queries:
//! [`text_registry`] builds a [`TextIndex`] searched by string, and
//! [`vector_registry`] builds a [`VectorStore`] searched by vector.

pub mod bm25;
pub mod flat;
pub mod fuse;

pub use bm25::{Bm25, Bm25Config};
pub use flat::{Flat, FlatConfig, Metric};
pub use fuse::{DEFAULT_K, rrf};
use ragworks_core::{Registry, TextIndex, VectorStore};

pub fn text_registry() -> Registry<dyn TextIndex> {
    let mut r = Registry::<dyn TextIndex>::new("text_index");
    r.register::<Bm25>(|c| Box::new(c)).expect("builtin");
    r
}

pub fn vector_registry() -> Registry<dyn VectorStore> {
    let mut r = Registry::<dyn VectorStore>::new("vector_store");
    r.register::<Flat>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use ragworks_core::Hit;

    use super::*;

    fn bm25_with(docs: &[(u64, &str)]) -> Box<dyn TextIndex> {
        let mut idx = text_registry().default_build("bm25").unwrap();
        for (id, text) in docs {
            idx.add(*id, text).unwrap();
        }
        idx.finish().unwrap();
        idx
    }

    fn search(idx: &dyn TextIndex, q: &str, k: usize) -> Vec<u64> {
        let mut out = Vec::new();
        idx.search(q, k, &mut out).unwrap();
        out.into_iter().map(|h| h.id).collect()
    }

    #[test]
    fn registries_expose_schemas() {
        assert_eq!(text_registry().names(), vec!["bm25"]);
        assert_eq!(vector_registry().names(), vec!["flat"]);
        let s = text_registry().schema("bm25").unwrap();
        for f in ["k1", "b", "tokenizer"] {
            assert!(s["properties"].get(f).is_some(), "bm25 schema lacks {f}");
        }
    }

    #[test]
    fn a_rare_term_outweighs_a_common_one() {
        // "the" is in every document and carries no information; "aardvark" is
        // in one. IDF is what makes the difference, so this is the test that
        // BM25 is actually BM25 and not term counting.
        let idx = bm25_with(&[
            (1, "the the the the cat"),
            (2, "the aardvark sat"),
            (3, "the the dog"),
        ]);
        assert_eq!(search(&*idx, "the aardvark", 3)[0], 2);
    }

    #[test]
    fn shorter_documents_win_on_equal_term_frequency() {
        let idx = bm25_with(&[
            (1, "quantum"),
            (2, "quantum padding padding padding padding padding padding padding"),
        ]);
        assert_eq!(search(&*idx, "quantum", 2), vec![1, 2], "b=0.75 must penalise length");
    }

    #[test]
    fn unknown_query_terms_are_ignored_not_fatal() {
        let idx = bm25_with(&[(1, "alpha beta"), (2, "beta gamma")]);
        assert_eq!(search(&*idx, "beta zzzzz", 2).len(), 2);
        assert!(search(&*idx, "zzzzz", 2).is_empty(), "a query of only unknown terms matches nothing");
    }

    #[test]
    fn search_is_case_insensitive() {
        let idx = bm25_with(&[(1, "Quantum Mechanics")]);
        assert_eq!(search(&*idx, "quantum", 1), vec![1]);
        assert_eq!(search(&*idx, "QUANTUM", 1), vec![1]);
    }

    #[test]
    fn k_truncates_the_result() {
        let idx = bm25_with(&[(1, "a x"), (2, "a y"), (3, "a z")]);
        assert_eq!(search(&*idx, "a", 2).len(), 2);
        assert_eq!(search(&*idx, "a", 0).len(), 0);
    }

    #[test]
    fn an_empty_index_returns_nothing_rather_than_erroring() {
        let idx = text_registry().default_build("bm25").unwrap();
        assert!(search(&*idx, "anything", 5).is_empty());
    }

    #[test]
    fn the_tokenizer_is_swappable_through_the_config() {
        // min_len 4 drops "cat", so a search for it finds nothing.
        let cfg = serde_json::json!({
            "tokenizer": {"name": "simple", "config": {"min_len": 4}}
        });
        let mut idx = text_registry().build("bm25", &cfg).unwrap();
        idx.add(1, "cat elephant").unwrap();
        assert!(search(&*idx, "cat", 5).is_empty(), "short tokens should have been dropped");
        assert_eq!(search(&*idx, "elephant", 5), vec![1]);
    }

    #[test]
    fn an_invalid_bm25_parameter_is_rejected_at_build_time() {
        let err = text_registry().build("bm25", &serde_json::json!({"b": 2.0})).err().unwrap();
        assert!(err.to_string().contains("b must be"), "{err}");
    }

    // ---------------------------------------------------------------- dense

    fn flat_with(dim: usize, rows: &[(u64, Vec<f32>)]) -> Box<dyn VectorStore> {
        let mut v = vector_registry().build("flat", &serde_json::json!({"dim": dim})).unwrap();
        let ids: Vec<u64> = rows.iter().map(|(i, _)| *i).collect();
        let data: Vec<f32> = rows.iter().flat_map(|(_, r)| r.clone()).collect();
        v.add(&ids, &data).unwrap();
        v
    }

    fn vsearch(v: &dyn VectorStore, q: &[f32], k: usize) -> Vec<u64> {
        let mut out = Vec::new();
        v.search(q, k, &mut out).unwrap();
        out.into_iter().map(|h| h.id).collect()
    }

    #[test]
    fn cosine_ranks_by_direction_not_magnitude() {
        let v = flat_with(2, &[(1, vec![10.0, 0.0]), (2, vec![0.0, 0.1])]);
        assert_eq!(vsearch(&*v, &[1.0, 0.0], 2), vec![1, 2]);
        // Doc 2 is tiny but points the right way, so it must still win.
        assert_eq!(vsearch(&*v, &[0.0, 1.0], 1), vec![2]);
    }

    #[test]
    fn dot_metric_rewards_magnitude_where_cosine_does_not() {
        let rows = [(1u64, vec![10.0f32, 0.0]), (2, vec![1.0, 0.0])];
        let ids: Vec<u64> = rows.iter().map(|(i, _)| *i).collect();
        let data: Vec<f32> = rows.iter().flat_map(|(_, r)| r.clone()).collect();

        let mut cos = vector_registry().build("flat", &serde_json::json!({"dim": 2})).unwrap();
        cos.add(&ids, &data).unwrap();
        let mut out = Vec::new();
        cos.search(&[1.0, 0.0], 2, &mut out).unwrap();
        assert!((out[0].score - out[1].score).abs() < 1e-6, "cosine: both are parallel, so equal");

        let mut d =
            vector_registry().build("flat", &serde_json::json!({"dim": 2, "metric": "dot"})).unwrap();
        d.add(&ids, &data).unwrap();
        let mut out = Vec::new();
        d.search(&[1.0, 0.0], 2, &mut out).unwrap();
        assert!(out[0].score > out[1].score, "dot: magnitude must matter");
    }

    #[test]
    fn a_zero_vector_scores_zero_instead_of_nan() {
        let v = flat_with(2, &[(1, vec![0.0, 0.0]), (2, vec![1.0, 0.0])]);
        let mut out = Vec::new();
        v.search(&[1.0, 0.0], 2, &mut out).unwrap();
        assert!(out.iter().all(|h| h.score.is_finite()), "NaN would poison the ranking: {out:?}");
        assert_eq!(out[0].id, 2);
    }

    #[test]
    fn dimension_mismatches_are_caught_on_both_sides() {
        let mut v = vector_registry().build("flat", &serde_json::json!({"dim": 3})).unwrap();
        assert!(v.add(&[1], &[1.0, 2.0]).is_err(), "row shorter than dim");
        v.add(&[1], &[1.0, 2.0, 3.0]).unwrap();
        let mut out = Vec::new();
        assert!(v.search(&[1.0, 2.0], 1, &mut out).is_err(), "query of wrong dim");
    }

    #[test]
    fn flat_requires_a_dimension() {
        let err = vector_registry().default_build("flat").err().unwrap();
        assert!(err.to_string().contains("dim"), "{err}");
    }

    #[test]
    fn lexical_and_dense_runs_fuse() {
        let lex: Vec<Hit> = vec![Hit { id: 1, score: 9.0 }, Hit { id: 2, score: 8.0 }];
        let den: Vec<Hit> = vec![Hit { id: 2, score: 0.9 }, Hit { id: 3, score: 0.8 }];
        let mut out = Vec::new();
        rrf(&[&lex, &den], DEFAULT_K, 10, &mut out);
        assert_eq!(out[0].id, 2, "appears in both runs, so it should lead");
        assert_eq!(out.len(), 3);
    }
}
