//! Live check against a real OpenAI-compatible endpoint.
//!
//!     OPENROUTER_API_KEY=... cargo run -p ragworks-embed --example live
//!
//! Mocks prove the retry and parsing logic; only this proves the integration.

use std::time::Instant;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> Result<(), ragworks_core::Error> {
    let texts = [
        "A database index is a B-tree that keeps a sorted copy of one or more columns.",
        "Indexes speed up reads by storing columns in sorted order.",
        "Volcanic activity in Iceland is driven by the Mid-Atlantic Ridge.",
    ];

    let cfg = serde_json::json!({
        "model": "openai/text-embedding-3-small",
        "dim": 1536,
        "requests_per_minute": 20.0,
        "max_batch": 32
    });

    let e = ragworks_embed::registry().build("openai", &cfg)?;
    println!("embedder: {} dim={} max_batch={}", e.name(), e.dim(), e.max_batch());

    let t = Instant::now();
    let mut out = Vec::new();
    e.embed(&texts, &mut out)?;
    let elapsed = t.elapsed();

    println!("{} texts -> {} floats in {:.0} ms", texts.len(), out.len(), elapsed.as_secs_f64() * 1000.0);
    let rows: Vec<&[f32]> = out.chunks(e.dim()).collect();
    println!("  similar pair   cos = {:.4}", cosine(rows[0], rows[1]));
    println!("  unrelated pair cos = {:.4}", cosine(rows[0], rows[2]));

    // Compare against the offline hasher on the same texts.
    let h = ragworks_embed::registry().build("hashing", &serde_json::json!({"dim": 512}))?;
    let mut hv = Vec::new();
    h.embed(&texts, &mut hv)?;
    let hr: Vec<&[f32]> = hv.chunks(h.dim()).collect();
    println!("  hashing baseline: similar {:.4}, unrelated {:.4}",
             cosine(hr[0], hr[1]), cosine(hr[0], hr[2]));
    Ok(())
}
