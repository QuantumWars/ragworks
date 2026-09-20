//! CSV and TSV.
//!
//! Each row becomes a labelled block separated by a blank line, so a row is a
//! paragraph as far as every chunker is concerned. Producing one document per
//! row is deliberately not done here: a reader turns bytes into text, and
//! splitting text into units is the chunker's job.

use ragworks_core::{Component, Error, Extracted, Reader, Result};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CsvConfig {
    /// Field separator. A tab makes this a TSV reader.
    #[serde(default = "default_delimiter")]
    pub delimiter: char,
    #[serde(default = "yes")]
    pub has_headers: bool,
    /// Columns to emit, by header name. Empty means all of them.
    #[serde(default)]
    pub columns: Vec<String>,
    /// Stop after this many rows. 0 means no limit.
    #[serde(default)]
    pub max_rows: usize,
}

fn default_delimiter() -> char {
    ','
}
fn yes() -> bool {
    true
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: default_delimiter(),
            has_headers: yes(),
            columns: Vec::new(),
            max_rows: 0,
        }
    }
}

#[derive(Debug)]
pub struct Csv {
    delimiter: u8,
    has_headers: bool,
    columns: Vec<String>,
    max_rows: usize,
}

impl Component for Csv {
    type Config = CsvConfig;
    const NAME: &'static str = "csv";
    const SUMMARY: &'static str = "CSV or TSV; each row becomes a labelled paragraph.";

    fn build(config: Self::Config) -> Result<Self> {
        if !config.delimiter.is_ascii() {
            return Err(Error::config(Self::NAME, "delimiter must be a single ASCII character"));
        }
        Ok(Self {
            delimiter: config.delimiter as u8,
            has_headers: config.has_headers,
            columns: config.columns,
            max_rows: config.max_rows,
        })
    }
}

impl Reader for Csv {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["csv", "tsv"]
    }

    fn read(&self, bytes: &[u8], _uri: &str) -> Result<Extracted> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_headers)
            .flexible(true)
            .from_reader(bytes);

        let headers: Vec<String> = if self.has_headers {
            rdr.headers()
                .map_err(|e| Error::provider("csv", None, format!("header row: {e}")))?
                .iter()
                .map(str::to_string)
                .collect()
        } else {
            Vec::new()
        };

        let wanted: Vec<usize> = if self.columns.is_empty() {
            (0..headers.len().max(1)).collect()
        } else {
            self.columns
                .iter()
                .filter_map(|c| headers.iter().position(|h| h == c))
                .collect()
        };
        if !self.columns.is_empty() && wanted.is_empty() {
            return Err(Error::config(
                Self::NAME,
                format!("none of {:?} are present; file has {:?}", self.columns, headers),
            ));
        }

        let mut text = String::new();
        let mut rows = 0usize;
        for record in rdr.records() {
            let record =
                record.map_err(|e| Error::provider("csv", None, format!("row {rows}: {e}")))?;
            for (i, field) in record.iter().enumerate() {
                if !self.columns.is_empty() && !wanted.contains(&i) {
                    continue;
                }
                if field.trim().is_empty() {
                    continue;
                }
                match headers.get(i) {
                    Some(h) => text.push_str(&format!("{h}: {field}\n")),
                    None => text.push_str(&format!("{field}\n")),
                }
            }
            text.push('\n');
            rows += 1;
            if self.max_rows > 0 && rows >= self.max_rows {
                break;
            }
        }

        Ok(Extracted {
            text,
            meta: serde_json::json!({"rows": rows, "columns": headers}),
        })
    }
}
