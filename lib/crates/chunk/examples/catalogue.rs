//! What the plugin system buys: a catalogue, and config-driven swapping.
//!
//!     cargo run -p ragworks-chunk --example catalogue

use ragworks_core::Corpus;

const DOC: &str = "# Handbook\n\nPreamble.\n\n## Safety\n\nWear boots.\n\n### Boots\n\nSteel toe.\n\n## Tools\n\nUse a wrench.\n";

fn main() -> Result<(), ragworks_core::Error> {
    let reg = ragworks_chunk::registry();

    println!("== catalogue ==");
    for p in reg.describe()["plugins"].as_array().unwrap() {
        let fields: Vec<_> = p["config_schema"]["properties"]
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        println!("  {:<10} {}", p["name"].as_str().unwrap(), p["summary"].as_str().unwrap());
        println!("  {:<10} config: {}", "", fields.join(", "));
    }

    let mut corpus = Corpus::new();
    corpus.add("handbook.md", DOC);
    let doc = corpus.view(ragworks_core::DocId(0)).unwrap();

    // The whole point: the strategy is a string plus JSON. No code change.
    let configs = [
        ("fixed", serde_json::json!({"size": 40})),
        ("recursive", serde_json::json!({"size": 40})),
        ("markdown", serde_json::json!({})),
        ("markdown", serde_json::json!({"leaves_only": true})),
    ];

    for (name, cfg) in &configs {
        let chunker = reg.build(name, cfg)?;
        let chunks = chunker.chunk_one(doc)?;
        println!("\n== {name} {cfg} -> {} chunks ==", chunks.len());
        for k in &chunks {
            let label = k
                .label()
                .map(|s| corpus.text(s).unwrap().to_string())
                .unwrap_or_else(|| "-".into());
            let body = corpus.chunk_text(k)?.replace('\n', "\\n");
            let preview: String = body.chars().take(46).collect();
            println!(
                "  d{} parent={:<5} {:<10} {}",
                k.depth,
                k.parent.map(|p| p.to_string()).unwrap_or_else(|| "-".into()),
                label,
                preview
            );
        }
    }

    // A bad config fails immediately, naming the plugin.
    match reg.build("fixed", &serde_json::json!({"size": 10, "overlap": 10})) {
        Err(e) => println!("\n== rejected at build time ==\n  {e}"),
        Ok(_) => unreachable!("overlap >= size must be rejected"),
    }

    println!(
        "\ncorpus: {} bytes, {} chunk = {} bytes (text is borrowed, never copied)",
        corpus.bytes(),
        1,
        std::mem::size_of::<ragworks_core::Chunk>()
    );
    Ok(())
}
