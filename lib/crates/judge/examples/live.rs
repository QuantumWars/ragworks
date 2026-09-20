//! Live check: the typed judge against its offline baseline.
//!
//!     OPENROUTER_API_KEY=... cargo run -p ragworks-judge --example live

use std::time::Instant;

const QUERY: &str = "Why does a database index make reads faster but writes slower?";

const PASSAGES: [&str; 5] = [
    "A database index is a separate B-tree keeping a sorted copy of one or more columns. Lookups descend the tree in logarithmic time instead of scanning every row, but each insert must also modify the tree, so writes pay extra work.",
    "Database indexes in PostgreSQL are created with the CREATE INDEX statement. The database chooses a name automatically if you do not supply one.",
    "Faster reads are the main reason teams add caching layers such as Redis in front of a database. A cache does not slow writes directly.",
    "The write-ahead log records every change before it is applied to the heap, making crash recovery possible. It is unrelated to how many indexes a table carries.",
    "Maintaining auxiliary structures alongside a table is a space-time tradeoff: you duplicate part of the data in a form that answers queries quickly, and every mutation must be applied in two places.",
];

fn order(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    idx
}

fn main() -> Result<(), ragworks_core::Error> {
    let lexical = ragworks_judge::reranker_registry().default_build("lexical")?;
    let jev = ragworks_judge::reranker_registry().default_build("jev")?;

    let mut lex = Vec::new();
    lexical.rerank(QUERY, &PASSAGES, &mut lex)?;

    let t = Instant::now();
    let mut typed = Vec::new();
    jev.rerank(QUERY, &PASSAGES, &mut typed)?;
    let ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("query: {QUERY}\n");
    println!("{:<6}{:>10}{:>8}{:>10}{:>8}  passage", "", "lexical", "rank", "jev", "rank");
    let (lo, jo) = (order(&lex), order(&typed));
    for i in 0..PASSAGES.len() {
        println!(
            "{:<6}{:>10.3}{:>8}{:>10.3}{:>8}  {}...",
            i,
            lex[i],
            lo.iter().position(|x| *x == i).unwrap(),
            typed[i],
            jo.iter().position(|x| *x == i).unwrap(),
            &PASSAGES[i][..52]
        );
    }
    println!("\n  lexical top-1 : passage {}", lo[0]);
    println!("  jev top-1     : passage {}  ({ms:.0} ms for all {} candidates, one round trip)",
             jo[0], PASSAGES.len());

    // Sufficiency: the full set, then a set with the answer removed.
    let verifier = ragworks_judge::verifier_registry().default_build("jev")?;
    let coverage = ragworks_judge::verifier_registry().default_build("coverage")?;
    println!("\nsufficiency");
    for (label, ev) in [("all passages", &PASSAGES[..]), ("answer removed", &PASSAGES[1..4])] {
        let j = verifier.verify(QUERY, ev)?;
        let c = coverage.verify(QUERY, ev)?;
        println!(
            "  {label:<16} jev={:?} ({:.2}, window {})   coverage={:?}",
            j.support, j.confidence, j.window, c.support
        );
    }
    Ok(())
}
