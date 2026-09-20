//! Chunking strategies, and the registry that makes them swappable by name.
//!
//! ```no_run
//! let reg = ragworks_chunk::registry();
//! let chunker = reg.build("markdown", &serde_json::json!({"leaves_only": true}))?;
//! # Ok::<(), ragworks_core::Error>(())
//! ```

pub mod fixed;
pub mod markdown;
pub mod recursive;

pub use fixed::{Fixed, FixedConfig};
pub use markdown::{Markdown, MarkdownConfig};
use ragworks_core::{Chunker, Registry};
pub use recursive::{Recursive, RecursiveConfig};

/// Every built-in chunking strategy, ready to build by name.
pub fn registry() -> Registry<dyn Chunker> {
    let mut r = Registry::<dyn Chunker>::new("chunker");
    r.register::<Fixed>(|c| Box::new(c)).expect("builtin");
    r.register::<Recursive>(|c| Box::new(c)).expect("builtin");
    r.register::<Markdown>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use ragworks_core::Corpus;

    use super::*;

    const DOC: &str = "# Book\n\nIntro text.\n\n## One\n\nFirst body.\n\n### One A\n\nNested body.\n\n## Two\n\nSecond body.\n";

    fn corpus_with(text: &str) -> Corpus {
        let mut c = Corpus::new();
        c.add("d.md", text);
        c
    }

    #[test]
    fn registry_exposes_every_builtin_with_a_schema() {
        let r = registry();
        assert_eq!(r.names(), vec!["fixed", "markdown", "recursive"]);
        for name in r.names() {
            let s = r.schema(name).unwrap();
            assert!(s["properties"].is_object(), "{name} has no config schema");
        }
    }

    #[test]
    fn fixed_covers_the_document_exactly_once_without_overlap() {
        let c = corpus_with("abcdefghij");
        let ch = registry().build("fixed", &serde_json::json!({"size": 4})).unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        let texts: Vec<_> = chunks.iter().map(|k| c.chunk_text(k).unwrap()).collect();
        assert_eq!(texts, vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn fixed_overlap_repeats_the_tail_of_the_previous_chunk() {
        let c = corpus_with("abcdefghij");
        let ch = registry().build("fixed", &serde_json::json!({"size": 4, "overlap": 2})).unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        let texts: Vec<_> = chunks.iter().map(|k| c.chunk_text(k).unwrap()).collect();
        assert_eq!(texts[0], "abcd");
        assert_eq!(texts[1], "cdef", "overlap must repeat the previous tail");
    }

    #[test]
    fn overlap_at_least_size_is_rejected_at_build_time() {
        let err = registry()
            .build("fixed", &serde_json::json!({"size": 4, "overlap": 4}))
            .err().unwrap();
        assert!(err.to_string().contains("overlap"), "{err}");
    }

    #[test]
    fn multibyte_text_never_splits_a_character() {
        let c = corpus_with("héllo wörld ünïcode");
        for size in 2..12 {
            let ch = registry().build("fixed", &serde_json::json!({"size": size})).unwrap();
            let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
            let joined: String =
                chunks.iter().map(|k| c.chunk_text(k).unwrap()).collect::<Vec<_>>().concat();
            assert_eq!(joined, "héllo wörld ünïcode", "size {size} corrupted the text");
        }
    }

    #[test]
    fn recursive_prefers_paragraph_boundaries() {
        let c = corpus_with("one one one\n\ntwo two two\n\nthree three");
        let ch = registry().build("recursive", &serde_json::json!({"size": 14})).unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        for k in &chunks {
            let t = c.chunk_text(k).unwrap();
            assert!(!t.trim().is_empty());
        }
        let joined: String =
            chunks.iter().map(|k| c.chunk_text(k).unwrap()).collect::<Vec<_>>().concat();
        assert_eq!(joined, "one one one\n\ntwo two two\n\nthree three");
    }

    #[test]
    fn markdown_builds_a_heading_hierarchy() {
        let c = corpus_with(DOC);
        let ch = registry().default_build("markdown").unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();

        let labels: Vec<_> = chunks
            .iter()
            .map(|k| k.label().map(|s| c.text(s).unwrap().to_string()))
            .collect();
        assert_eq!(
            labels,
            vec![
                Some("Book".into()),
                Some("One".into()),
                Some("One A".into()),
                Some("Two".into())
            ]
        );

        // "One A" (###) nests inside "One" (##), which nests inside "Book" (#).
        let one_a = &chunks[2];
        assert_eq!(one_a.depth, 3);
        assert_eq!(one_a.parent, Some(1));
        assert_eq!(chunks[1].parent, Some(0));
        assert_eq!(chunks[0].parent, None);
    }

    #[test]
    fn a_parent_section_spans_its_children() {
        let c = corpus_with(DOC);
        let ch = registry().default_build("markdown").unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        let one = c.chunk_text(&chunks[1]).unwrap();
        assert!(one.contains("First body"));
        assert!(one.contains("Nested body"), "parent must span its subsections");
        assert!(!one.contains("Second body"), "and stop at the next sibling");
    }

    #[test]
    fn leaves_only_drops_sections_that_have_subsections() {
        let c = corpus_with(DOC);
        let ch = registry().build("markdown", &serde_json::json!({"leaves_only": true})).unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        let labels: Vec<_> = chunks
            .iter()
            .filter_map(|k| k.label().map(|s| c.text(s).unwrap().to_string()))
            .collect();
        assert_eq!(labels, vec!["One A", "Two"], "only childless sections are leaves");
    }

    #[test]
    fn a_hash_without_a_space_is_not_a_heading() {
        let c = corpus_with("# Real\n\n#notaheading still body\n");
        let ch = registry().default_build("markdown").unwrap();
        let chunks = ch.chunk_one(c.view(ragworks_core::DocId(0)).unwrap()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(c.chunk_text(&chunks[0]).unwrap().contains("#notaheading"));
    }

    #[test]
    fn chunking_appends_so_one_buffer_serves_a_whole_corpus() {
        let mut c = Corpus::new();
        c.add("a.md", "# A\n\nbody a\n");
        c.add("b.md", "# B\n\nbody b\n");
        let ch = registry().default_build("markdown").unwrap();
        let mut out = Vec::new();
        for doc in c.docs() {
            ch.chunk(doc, &mut out).unwrap();
        }
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].doc, ragworks_core::DocId(0));
        assert_eq!(out[1].doc, ragworks_core::DocId(1));
        assert!(c.chunk_text(&out[1]).unwrap().contains("body b"));
    }
}
