//! Walk a directory into a corpus, then chunk and index it.
//!
//!     cargo run -p ragworks-read --example ingest -- <dir>
//!     cargo run -p ragworks-read --features pdf --example ingest -- <dir>

use std::time::Instant;

use ragworks_core::Corpus;
use ragworks_read::{IngestOptions, Readers, ingest_dir};

fn main() -> Result<(), ragworks_core::Error> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let readers = Readers::standard()?;
    println!("readers handle: {}", readers.extensions().join(" "));

    let mut corpus = Corpus::new();
    let t = Instant::now();
    let stats = ingest_dir(&mut corpus, &dir, &readers, &IngestOptions::default())?;
    let took = t.elapsed().as_secs_f64();

    println!(
        "\n{} files -> {} documents, {} KB of text in {:.2}s ({} skipped, {} errors)",
        stats.files,
        corpus.len(),
        stats.bytes / 1024,
        took,
        stats.skipped,
        stats.errors.len()
    );
    for (p, e) in stats.errors.iter().take(3) {
        println!("  error: {} -- {e}", p.display());
    }

    if corpus.is_empty() {
        return Ok(());
    }

    let chunker = ragworks_chunk::registry().default_build("markdown")?;
    let mut chunks = Vec::new();
    for doc in corpus.docs() {
        chunker.chunk(doc, &mut chunks)?;
    }

    let mut idx = ragworks_index::text_registry().default_build("bm25")?;
    for (i, c) in chunks.iter().enumerate() {
        idx.add(i as u64, corpus.chunk_text(c)?)?;
    }
    println!("{} chunks indexed", chunks.len());

    for q in ["retrieval augmented generation", "table of contents structure"] {
        let mut hits = Vec::new();
        idx.search(q, 3, &mut hits)?;
        println!("\n  {q:?}");
        for h in &hits {
            let c = &chunks[h.id as usize];
            let label = c.label().map(|s| corpus.text(s).unwrap()).unwrap_or("-");
            let uri = &corpus.doc(c.doc).unwrap().uri;
            println!("     {:.3}  {uri}  {label}", h.score);
        }
    }
    Ok(())
}
