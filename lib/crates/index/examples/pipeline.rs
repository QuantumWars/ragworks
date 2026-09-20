//! Chunk -> index -> search, with a throughput measurement.
//!
//!     cargo run --release -p ragworks-index --example pipeline

use std::time::Instant;

use ragworks_core::Corpus;

const HANDBOOK: &str = "# Handbook\n\n## Indexes\n\nA database index is a B-tree keeping a sorted copy of one or more columns, so lookups descend in logarithmic time instead of scanning every row.\n\n## Writes\n\nEvery insert must also modify each index, so writes pay extra work for the reads they accelerate.\n\n## Caching\n\nA Redis cache in front of a database speeds reads without slowing writes, but introduces invalidation problems.\n";

fn main() -> Result<(), ragworks_core::Error> {
    // 1. Corpus, chunked by structure.
    let mut corpus = Corpus::new();
    corpus.add("handbook.md", HANDBOOK);
    let chunker = ragworks_chunk::registry().build(
        "markdown",
        &serde_json::json!({"leaves_only": true}),
    )?;
    let mut chunks = Vec::new();
    for doc in corpus.docs() {
        chunker.chunk(doc, &mut chunks)?;
    }

    // 2. Index those chunks.
    let mut idx = ragworks_index::text_registry().default_build("bm25")?;
    for (i, c) in chunks.iter().enumerate() {
        idx.add(i as u64, corpus.chunk_text(c)?)?;
    }
    idx.finish()?;

    println!("{} chunks indexed\n", idx.len());

    for query in ["why are writes slower", "sorted copy of a column", "redis"] {
        let mut hits = Vec::new();
        idx.search(query, 2, &mut hits)?;
        println!("  {query:?}");
        for h in &hits {
            let label = chunks[h.id as usize]
                .label()
                .map(|s| corpus.text(s).unwrap())
                .unwrap_or("-");
            println!("      {:.3}  {}", h.score, label);
        }
    }

    // 3. Throughput on synthetic volume.
    const N: usize = 50_000;
    let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
    let docs: Vec<String> = (0..N)
        .map(|i| {
            (0..40).map(|j| words[(i * 7 + j * 3) % words.len()]).collect::<Vec<_>>().join(" ")
        })
        .collect();

    let mut big = ragworks_index::text_registry().default_build("bm25")?;
    let t = Instant::now();
    for (i, d) in docs.iter().enumerate() {
        big.add(i as u64, d)?;
    }
    big.finish()?;
    let index_s = t.elapsed().as_secs_f64();

    const Q: usize = 1000;
    let t = Instant::now();
    let mut hits = Vec::new();
    for i in 0..Q {
        hits.clear();
        big.search(words[i % words.len()], 10, &mut hits)?;
    }
    let ms_per_query = t.elapsed().as_secs_f64() * 1000.0 / Q as f64;

    println!(
        "\n{N} docs x 40 tokens\n  index  {index_s:.2}s  ({:.0} docs/s)\n  \
         search {ms_per_query:.2} ms/query, averaged over {Q}\n  \
         (worst case: each query is a single term present in ~1/8 of the corpus)",
        N as f64 / index_s,
    );
    Ok(())
}
