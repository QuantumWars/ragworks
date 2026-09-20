//! Rank fusion.
//!
//! Reciprocal Rank Fusion (Cormack et al., 2009) combines runs by **rank**
//! rather than score, so legs with incomparable score scales -- BM25 against
//! cosine similarity -- need no calibration:
//!
//! ```text
//! score(d) = sum over runs of  1 / (k + rank(d))
//! ```
//!
//! Measured caveat from this repository's wave-1 experiment: fusing BM25 with
//! dense retrieval on HotpotQA moved recall@5 by **+0.019 at p=0.43** -- that
//! is, not at all. Fusion is not a free win; it pays only when the legs fail
//! on different queries. Measure it on your own task before adopting it.

use std::collections::HashMap;

use ragworks_core::Hit;

/// The constant that damps the contribution of top ranks. 60 is the value from
/// the original paper and the near-universal default.
pub const DEFAULT_K: f32 = 60.0;

/// Fuse ranked runs. Each run must be ordered best-first; scores are ignored.
pub fn rrf(runs: &[&[Hit]], k: f32, top: usize, out: &mut Vec<Hit>) {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for run in runs {
        for (rank, hit) in run.iter().enumerate() {
            *acc.entry(hit.id).or_insert(0.0) += 1.0 / (k + (rank + 1) as f32);
        }
    }
    let mut scored: Vec<(f32, u64)> = acc.into_iter().map(|(id, s)| (s, id)).collect();
    let top = top.min(scored.len());
    if top < scored.len() {
        scored.select_nth_unstable_by(top, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(top);
    }
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    out.extend(scored.into_iter().map(|(score, id)| Hit { id, score }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(ids: &[u64]) -> Vec<Hit> {
        ids.iter().map(|&id| Hit { id, score: 0.0 }).collect()
    }

    #[test]
    fn agreement_between_legs_wins() {
        let a = hits(&[1, 2, 3]);
        let b = hits(&[1, 3, 2]);
        let mut out = Vec::new();
        rrf(&[&a, &b], DEFAULT_K, 10, &mut out);
        assert_eq!(out[0].id, 1, "first in both runs must lead");
    }

    #[test]
    fn being_first_somewhere_beats_being_middling_everywhere() {
        // A genuinely counter-intuitive property, and the reason RRF surfaces
        // specialist results. 1/(k + rank) is convex, so
        //     1/61 + 1/63  =  0.032266  >  2 x 1/62  =  0.032258
        // A document ranked first by one leg and last by the other outranks a
        // document both legs put in the middle. If you wanted the consensus
        // pick instead, RRF is the wrong fusion.
        let a = hits(&[1, 2, 3]);
        let b = hits(&[3, 2, 1]);
        let mut out = Vec::new();
        rrf(&[&a, &b], DEFAULT_K, 10, &mut out);
        assert_eq!(
            out.iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![1, 3, 2],
            "the consistently-second document comes last; ties break on id"
        );
        assert!(out[0].score > out[2].score);
    }

    #[test]
    fn score_scales_are_irrelevant_only_ranks_count() {
        let a: Vec<Hit> = vec![Hit { id: 7, score: 1e6 }, Hit { id: 8, score: 1e-6 }];
        let b: Vec<Hit> = vec![Hit { id: 7, score: 0.01 }, Hit { id: 8, score: 0.009 }];
        let mut out = Vec::new();
        rrf(&[&a, &b], DEFAULT_K, 10, &mut out);
        assert_eq!(out[0].id, 7);
        assert!((out[0].score - 2.0 / 61.0).abs() < 1e-6, "got {}", out[0].score);
    }

    #[test]
    fn documents_in_only_one_run_still_appear() {
        let a = hits(&[1]);
        let b = hits(&[2]);
        let mut out = Vec::new();
        rrf(&[&a, &b], DEFAULT_K, 10, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn top_truncates_and_ties_break_deterministically() {
        let a = hits(&[1, 2, 3, 4]);
        let b = hits(&[1, 2, 3, 4]);
        let mut out = Vec::new();
        rrf(&[&a, &b], DEFAULT_K, 2, &mut out);
        assert_eq!(out.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2]);
    }
}
