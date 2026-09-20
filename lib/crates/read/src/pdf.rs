//! PDF text extraction. Behind the `pdf` feature.
//!
//! Uses `pdf-extract`, which is pure Rust and therefore needs no system
//! library, keeping wheels to one file per platform. The trade is quality: on
//! multi-column layouts, tables and anything scanned, a native engine such as
//! PDFium or Poppler does materially better. If extraction quality matters more
//! than installation simplicity, implement [`Reader`] over one of those and
//! register it in place of this.
//!
//! Scanned PDFs contain no text layer at all, and no extractor will find one;
//! that needs OCR, which is out of scope here.

use ragworks_core::{Component, Error, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct PdfConfig {}

#[derive(Debug, Default)]
pub struct Pdf;

impl Component for Pdf {
    type Config = PdfConfig;
    const NAME: &'static str = "pdf";
    const SUMMARY: &'static str = "PDF text extraction, pure Rust; no system library required.";

    fn build(_config: Self::Config) -> Result<Self> {
        Ok(Self)
    }
}

impl Reader for Pdf {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["pdf"]
    }

    fn read(&self, bytes: &[u8], uri: &str) -> Result<Extracted> {
        let text = pdf_extract::extract_text_from_mem(bytes)
            .map_err(|e| Error::provider("pdf", None, format!("{uri}: {e}")))?;
        let empty = text.trim().is_empty();
        Ok(Extracted {
            text,
            // A PDF with no text layer is a scan. Flagging it is the difference
            // between a silently empty document and an actionable diagnosis.
            meta: serde_json::json!({"no_text_layer": empty}),
        })
    }
}
