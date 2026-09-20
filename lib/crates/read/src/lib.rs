//! Readers: bytes of some format, into text.
//!
//! The one place in the pipeline where copying is unavoidable -- parsing a PDF
//! or rendering HTML has to produce a new string. Everything downstream works
//! on views into the corpus blob.
//!
//! [`Readers`] dispatches on file extension, and [`ingest_dir`] walks a
//! directory into a [`Corpus`], so the practical entry point is one call.

pub mod html;
pub mod json;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod markdown;
pub mod tabular;
pub mod text;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use html::{Html, HtmlConfig};
pub use json::{Json, JsonConfig};
pub use markdown::{Markdown, MarkdownConfig};
use ragworks_core::{Corpus, Error, Extracted, Reader, Registry, Result};
pub use tabular::{Csv, CsvConfig};
pub use text::{Text, TextConfig};

pub fn registry() -> Registry<dyn Reader> {
    let mut r = Registry::<dyn Reader>::new("reader");
    r.register::<Text>(|c| Box::new(c)).expect("builtin");
    r.register::<Markdown>(|c| Box::new(c)).expect("builtin");
    r.register::<Html>(|c| Box::new(c)).expect("builtin");
    r.register::<Csv>(|c| Box::new(c)).expect("builtin");
    r.register::<Json>(|c| Box::new(c)).expect("builtin");
    #[cfg(feature = "pdf")]
    r.register::<pdf::Pdf>(|c| Box::new(c)).expect("builtin");
    r
}

/// Format dispatch by file extension.
pub struct Readers {
    readers: Vec<Box<dyn Reader>>,
    by_ext: HashMap<String, usize>,
    fallback: Option<usize>,
}

impl Readers {
    /// Every built-in reader at its defaults, with `text` as the fallback for
    /// unknown extensions. A file that is probably text is better read as text
    /// than skipped.
    pub fn standard() -> Result<Self> {
        let reg = registry();
        let mut s = Self { readers: Vec::new(), by_ext: HashMap::new(), fallback: None };
        for name in reg.names() {
            s.add(reg.default_build(name)?);
        }
        s.fallback = s.by_ext.get("txt").copied();
        Ok(s)
    }

    pub fn add(&mut self, reader: Box<dyn Reader>) -> &mut Self {
        let idx = self.readers.len();
        for ext in reader.extensions() {
            // Last registration wins, so a caller can override a built-in by
            // adding their own reader for the same extension.
            self.by_ext.insert(ext.to_ascii_lowercase(), idx);
        }
        self.readers.push(reader);
        self
    }

    /// Read unknown extensions with this reader instead of skipping them.
    pub fn with_fallback(&mut self, reader: Box<dyn Reader>) -> &mut Self {
        self.add(reader);
        self.fallback = Some(self.readers.len() - 1);
        self
    }

    pub fn for_extension(&self, ext: &str) -> Option<&dyn Reader> {
        self.by_ext
            .get(&ext.to_ascii_lowercase())
            .or(self.fallback.as_ref())
            .map(|i| self.readers[*i].as_ref())
    }

    /// Whether this extension has a reader of its own, as opposed to landing on
    /// the fallback. Ingestion uses this to decide when a file is worth
    /// sniffing before trusting it to the text reader.
    pub fn has_extension(&self, ext: &str) -> bool {
        self.by_ext.contains_key(&ext.to_ascii_lowercase())
    }

    pub fn for_uri(&self, uri: &str) -> Option<&dyn Reader> {
        let ext = Path::new(uri).extension().and_then(|e| e.to_str()).unwrap_or("");
        self.for_extension(ext)
    }

    pub fn extensions(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.by_ext.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    pub fn read_bytes(&self, bytes: &[u8], uri: &str) -> Result<Extracted> {
        let r = self.for_uri(uri).ok_or_else(|| {
            Error::UnknownPlugin {
                kind: "reader",
                name: uri.to_string(),
                available: self.extensions().join(", "),
            }
        })?;
        r.read(bytes, uri)
    }

    pub fn read_path(&self, path: impl AsRef<Path>) -> Result<Extracted> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        self.read_bytes(&bytes, &path.to_string_lossy())
    }
}

#[derive(Debug, Default)]
pub struct IngestStats {
    pub files: usize,
    pub skipped: usize,
    pub bytes: usize,
    /// Per-file failures. Collected rather than propagated: one unreadable file
    /// in ten thousand must not discard the other 9,999.
    pub errors: Vec<(PathBuf, String)>,
}

#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub max_depth: usize,
    /// Skip dot-files and dot-directories.
    pub skip_hidden: bool,
    /// Skip files larger than this. 0 means no limit.
    pub max_bytes: usize,
    /// Skip files that produced no text, rather than adding empty documents.
    pub skip_empty: bool,
    /// Sniff files with an unregistered extension and skip those that look
    /// binary.
    ///
    /// Without this the text fallback happily ingests `.pyc`, `.so` and image
    /// files as mojibake, which then pollutes the index with tokens no query
    /// will ever match. Files whose extension has a real reader are never
    /// sniffed -- a PDF is binary and is supposed to be.
    pub skip_binary: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            max_depth: 32,
            skip_hidden: true,
            max_bytes: 64 << 20,
            skip_empty: true,
            skip_binary: true,
        }
    }
}

fn is_hidden(e: &walkdir::DirEntry) -> bool {
    e.file_name().to_str().is_some_and(|s| s.starts_with('.') && s != ".")
}

/// A NUL byte in the first few KB means binary. The same heuristic git uses:
/// cheap, and text files essentially never contain one.
pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

/// Walk `dir`, read every file a reader recognises, and add it to `corpus`.
///
/// The document URI is the path relative to `dir`, so a corpus is reproducible
/// regardless of where it was built.
pub fn ingest_dir(
    corpus: &mut Corpus,
    dir: impl AsRef<Path>,
    readers: &Readers,
    opts: &IngestOptions,
) -> Result<IngestStats> {
    let dir = dir.as_ref();
    let mut stats = IngestStats::default();

    let walker = walkdir::WalkDir::new(dir)
        .max_depth(opts.max_depth)
        .into_iter()
        .filter_entry(|e| !(opts.skip_hidden && e.depth() > 0 && is_hidden(e)));

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                stats.errors.push((dir.to_path_buf(), e.to_string()));
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if readers.for_extension(ext).is_none() {
            stats.skipped += 1;
            continue;
        }
        let len = entry.metadata().map(|m| m.len() as usize).unwrap_or(0);
        if opts.max_bytes > 0 && len > opts.max_bytes {
            stats.skipped += 1;
            continue;
        }

        if opts.skip_binary && !readers.has_extension(ext) {
            match std::fs::read(path) {
                Ok(b) if looks_binary(&b) => {
                    stats.skipped += 1;
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    stats.errors.push((path.to_path_buf(), e.to_string()));
                    continue;
                }
            }
        }

        match readers.read_path(path) {
            Ok(x) => {
                if opts.skip_empty && x.text.trim().is_empty() {
                    stats.skipped += 1;
                    continue;
                }
                let uri = path.strip_prefix(dir).unwrap_or(path).to_string_lossy().to_string();
                stats.bytes += x.text.len();
                stats.files += 1;
                corpus.add_with_meta(uri, &x.text, x.meta);
            }
            Err(e) => stats.errors.push((path.to_path_buf(), e.to_string())),
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(name: &str, cfg: serde_json::Value, bytes: &[u8]) -> Extracted {
        registry().build(name, &cfg).unwrap().read(bytes, "t").unwrap()
    }

    #[test]
    fn registry_exposes_every_reader_with_a_schema() {
        let r = registry();
        for name in ["text", "markdown", "html", "csv", "json"] {
            assert!(r.contains(name), "{name} is not registered");
            assert!(r.schema(name).is_ok());
        }
    }

    // ----------------------------------------------------------------- text

    #[test]
    fn utf8_is_passed_through_and_reported_as_such() {
        let x = read("text", serde_json::json!({}), "héllo wörld".as_bytes());
        assert_eq!(x.text, "héllo wörld");
        assert_eq!(x.meta["encoding"], "utf-8");
        assert_eq!(x.meta["lossy"], false);
    }

    #[test]
    fn invalid_utf8_decodes_through_the_fallback_and_says_so() {
        // 0xE9 is 'é' in windows-1252 and invalid on its own in UTF-8.
        let bytes = [b'c', b'a', b'f', 0xE9];
        let x = read("text", serde_json::json!({"fallback_encoding": "windows-1252"}), &bytes);
        assert_eq!(x.text, "café");
        assert_eq!(x.meta["encoding"], "windows-1252");
    }

    #[test]
    fn an_unknown_fallback_encoding_is_rejected_at_build_time() {
        let err = registry()
            .build("text", &serde_json::json!({"fallback_encoding": "klingon-8"}))
            .err()
            .unwrap();
        assert!(err.to_string().contains("klingon-8"), "{err}");
    }

    // ------------------------------------------------------------- markdown

    #[test]
    fn front_matter_is_lifted_and_the_body_keeps_its_headings() {
        let src = "---\ntitle: Handbook\nauthor: \"A. Writer\"\n---\n# Handbook\n\n## Part\n\nBody.\n";
        let x = read("markdown", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.meta["front_matter"]["title"], "Handbook");
        assert_eq!(x.meta["front_matter"]["author"], "A. Writer");
        assert_eq!(x.meta["title"], "Handbook");
        assert!(x.text.starts_with("# Handbook"), "front matter must be stripped: {:?}", x.text);
        assert!(x.text.contains("## Part"), "heading structure must survive");
    }

    #[test]
    fn a_document_without_front_matter_is_untouched() {
        let src = "# Title\n\nBody with --- a dash line.\n";
        let x = read("markdown", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.text, src);
        assert!(x.meta.get("front_matter").is_none());
    }

    // ----------------------------------------------------------------- html

    #[test]
    fn html_becomes_text_with_its_title_and_structure() {
        let src = r#"<html><head><title>Storage</title></head>
            <body><h1>Indexes</h1><p>A B-tree keeps a <b>sorted</b> copy.</p>
            <script>ignore_me()</script></body></html>"#;
        let x = read("html", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.meta["title"], "Storage");
        assert!(x.text.contains("Indexes"));
        assert!(x.text.contains("sorted"));
        assert!(!x.text.contains("<p>"), "tags must be gone: {:?}", x.text);
        assert!(!x.text.contains("ignore_me"), "script contents must not leak into the text");
    }

    // ------------------------------------------------------------------ csv

    #[test]
    fn csv_rows_become_labelled_paragraphs() {
        let src = "name,role\nAda,engineer\nGrace,admiral\n";
        let x = read("csv", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.meta["rows"], 2);
        assert!(x.text.contains("name: Ada"));
        assert!(x.text.contains("role: admiral"));
        // Blank-line separation is what makes a row a paragraph downstream.
        assert!(x.text.contains("\n\n"));
    }

    #[test]
    fn selected_columns_are_kept_and_a_missing_one_is_an_error() {
        let src = "name,role,secret\nAda,engineer,xyzzy\n";
        let x = read("csv", serde_json::json!({"columns": ["name"]}), src.as_bytes());
        assert!(x.text.contains("Ada"));
        assert!(!x.text.contains("xyzzy"), "unselected columns must not leak");

        let err = registry()
            .build("csv", &serde_json::json!({"columns": ["nope"]}))
            .unwrap()
            .read(src.as_bytes(), "t")
            .err()
            .unwrap();
        assert!(err.to_string().contains("nope"), "{err}");
    }

    #[test]
    fn a_tab_delimiter_makes_it_a_tsv_reader() {
        let x = read("csv", serde_json::json!({"delimiter": "\t"}), b"a\tb\n1\t2\n");
        assert!(x.text.contains("a: 1"), "{:?}", x.text);
    }

    // ----------------------------------------------------------------- json

    #[test]
    fn a_json_array_becomes_one_paragraph_per_record() {
        let src = r#"[{"title":"One","body":"First"},{"title":"Two","body":"Second"}]"#;
        let x = read("json", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.meta["records"], 2);
        assert!(x.text.contains("First") && x.text.contains("Second"));
    }

    #[test]
    fn line_delimited_json_is_detected_without_relying_on_the_extension() {
        let src = "{\"t\":\"one\"}\n{\"t\":\"two\"}\n";
        let x = read("json", serde_json::json!({}), src.as_bytes());
        assert_eq!(x.meta["records"], 2, "whole-file parse must fall back to per-line");
    }

    #[test]
    fn pointers_select_fields_and_numbers_are_skipped() {
        let src = r#"{"title":"Keep","meta":{"id":42,"note":"Also"},"drop":"Gone"}"#;
        let x = read("json", serde_json::json!({"fields": ["/title", "/meta/note"]}), src.as_bytes());
        assert!(x.text.contains("Keep") && x.text.contains("Also"));
        assert!(!x.text.contains("Gone"));
        assert!(!x.text.contains("42"), "numbers carry no retrievable language");
    }

    #[test]
    fn a_pointer_without_a_leading_slash_is_rejected() {
        let err = registry()
            .build("json", &serde_json::json!({"fields": ["title"]}))
            .err()
            .unwrap();
        assert!(err.to_string().contains("JSON Pointer"), "{err}");
    }

    #[test]
    fn text_that_is_neither_json_nor_jsonl_names_both_failures() {
        let err = registry()
            .default_build("json")
            .unwrap()
            .read(b"not json at all", "t")
            .err()
            .unwrap();
        let m = err.to_string();
        assert!(m.contains("JSON Lines") && m.contains("line 1"), "{m}");
    }

    // ------------------------------------------------------------- dispatch

    #[test]
    fn dispatch_picks_a_reader_by_extension() {
        let r = Readers::standard().unwrap();
        assert_eq!(r.for_uri("a/b/notes.md").unwrap().name(), "markdown");
        assert_eq!(r.for_uri("page.HTML").unwrap().name(), "html", "extensions are case-insensitive");
        assert_eq!(r.for_uri("data.csv").unwrap().name(), "csv");
    }

    #[test]
    fn an_unknown_extension_falls_back_to_text_rather_than_being_dropped() {
        let r = Readers::standard().unwrap();
        assert_eq!(r.for_uri("LICENSE").unwrap().name(), "text");
    }

    // --------------------------------------------------------------- ingest

    fn fixture(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ragworks_read_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("a.md"), "# Alpha\n\nFirst doc.\n").unwrap();
        std::fs::write(dir.join("nested/b.csv"), "k,v\nx,y\n").unwrap();
        std::fs::write(dir.join(".hidden.md"), "# Secret\n").unwrap();
        std::fs::write(dir.join("empty.md"), "   \n").unwrap();
        dir
    }

    #[test]
    fn ingest_walks_a_directory_into_a_corpus() {
        let dir = fixture("walk");
        let mut corpus = Corpus::new();
        let stats =
            ingest_dir(&mut corpus, &dir, &Readers::standard().unwrap(), &IngestOptions::default())
                .unwrap();

        assert_eq!(stats.files, 2, "a.md and nested/b.csv");
        assert!(stats.errors.is_empty(), "{:?}", stats.errors);
        assert_eq!(corpus.len(), 2);

        // URIs are relative to the ingest root, so a corpus is reproducible
        // wherever it was built.
        let uris: Vec<String> = (0..corpus.len())
            .map(|i| corpus.doc(ragworks_core::DocId(i as u64)).unwrap().uri.clone())
            .collect();
        assert!(uris.iter().any(|u| u == "a.md"), "{uris:?}");
        assert!(uris.iter().any(|u| u.ends_with("b.csv")), "{uris:?}");
        assert!(!uris.iter().any(|u| u.contains("hidden")), "dot-files must be skipped");
        assert!(!uris.iter().any(|u| u.contains("empty")), "blank documents must be skipped");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binary_files_with_unknown_extensions_are_not_ingested_as_text() {
        // Found by running ingestion over a real project: compiled Python
        // landed in the index as mojibake via the text fallback and started
        // appearing in search results.
        let dir = std::env::temp_dir()
            .join(format!("ragworks_read_bin_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("real.md"), "# Real\n\nText.\n").unwrap();
        std::fs::write(dir.join("compiled.pyc"), [0xcb, 0x0d, 0x0d, 0x0a, 0x00, 0x00, 0x41]).unwrap();
        std::fs::write(dir.join("LICENSE"), "MIT License\n\nPermission is granted.\n").unwrap();

        let mut corpus = Corpus::new();
        let stats =
            ingest_dir(&mut corpus, &dir, &Readers::standard().unwrap(), &IngestOptions::default())
                .unwrap();

        let uris: Vec<String> = (0..corpus.len())
            .map(|i| corpus.doc(ragworks_core::DocId(i as u64)).unwrap().uri.clone())
            .collect();
        assert!(!uris.iter().any(|u| u.ends_with(".pyc")), "binary was ingested: {uris:?}");
        // An extensionless text file must still be read, so the sniff has to
        // be a content check rather than an extension allow-list.
        assert!(uris.iter().any(|u| u == "LICENSE"), "{uris:?}");
        assert!(uris.iter().any(|u| u == "real.md"), "{uris:?}");
        assert_eq!(stats.files, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_registered_binary_format_is_never_sniffed_away() {
        // PDFs are binary and are supposed to be.
        assert!(looks_binary(&[0x25, 0x50, 0x44, 0x46, 0x00, 0x01]));
        assert!(Readers::standard().unwrap().has_extension("csv"));
        assert!(!Readers::standard().unwrap().has_extension("pyc"));
    }

    #[test]
    fn metadata_from_the_reader_reaches_the_corpus() {
        let dir = fixture("meta");
        let mut corpus = Corpus::new();
        ingest_dir(&mut corpus, &dir, &Readers::standard().unwrap(), &IngestOptions::default())
            .unwrap();
        let md = (0..corpus.len())
            .map(|i| corpus.doc(ragworks_core::DocId(i as u64)).unwrap())
            .find(|d| d.uri == "a.md")
            .unwrap();
        assert_eq!(md.meta["title"], "Alpha");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
