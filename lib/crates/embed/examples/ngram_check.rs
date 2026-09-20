//! Did character n-grams fix the reported failure? Old behaviour is
//! reproduced with `ngrams: []`, which is exactly what the embedder did before.
//!
//!     cargo run -p ragworks-embed --example ngram_check

use ragworks_core::Embedder;

fn build(ngrams: &str, min_len: usize) -> Box<dyn Embedder> {
    let cfg = serde_json::json!({
        "dim": 512,
        "ngrams": serde_json::from_str::<Vec<usize>>(ngrams).unwrap(),
        "tokenizer": {"name": "simple", "config": {"min_len": min_len}}
    });
    ragworks_embed::registry().build("hashing", &cfg).unwrap()
}

fn cos(e: &dyn Embedder, a: &str, b: &str) -> f32 {
    let mut v = Vec::new();
    e.embed(&[a, b], &mut v).unwrap();
    let (x, y) = v.split_at(e.dim());
    x.iter().zip(y).map(|(p, q)| p * q).sum()
}

fn main() {
    let old = build("[]", 1);
    let new = build("[3,4,5]", 1);
    let filtered = build("[3,4,5]", 3);

    println!("{:<44}{:>9}{:>9}{:>11}", "pair", "tokens", "ngrams", "+min_len3");
    println!("{:-<73}", "");
    let pairs = [
        ("writing", "writes"),
        ("index", "indexes"),
        ("sorted column", "sorting columns"),
        ("volcano", "database"),
    ];
    for (a, b) in pairs {
        println!(
            "{:<44}{:>9.4}{:>9.4}{:>11.4}",
            format!("{a:?} vs {b:?}"),
            cos(&*old, a, b),
            cos(&*new, a, b),
            cos(&*filtered, a, b)
        );
    }

    println!("\nThe three-sentence case from the live run:");
    let s = [
        "A database index is a B-tree that keeps a sorted copy of one or more columns.",
        "Indexes speed up reads by storing columns in sorted order.",
        "Volcanic activity in Iceland is driven by the Mid-Atlantic Ridge.",
    ];
    for (label, e) in [("tokens only", &old), ("+ n-grams", &new), ("+ min_len 3", &filtered)] {
        let rel = cos(&**e, s[0], s[1]);
        let unrel = cos(&**e, s[0], s[2]);
        let verdict = if rel > unrel { "correct" } else { "BACKWARDS" };
        println!("  {label:<14} related {rel:.4}   unrelated {unrel:.4}   {verdict}");
    }
    println!("  {:<14} related 0.5096   unrelated 0.0660   (learned embedder, for scale)", "openai");
}
