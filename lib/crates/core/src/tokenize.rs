//! Tokenization -- a swap point, because one tokenizer never fits every corpus.
//!
//! English prose, CJK text, source code and chemical formulae all disagree
//! about what a word is. The trait emits **byte ranges** rather than `String`s,
//! so a caller that only needs to look a term up in a dictionary never
//! allocates.

use schemars::JsonSchema;
use serde::Deserialize;

use crate::error::Result;
use crate::plugin::{Component, Registry};
use unicode_segmentation::UnicodeSegmentation;

/// A token's byte range within the text it came from.
pub type TokenSpan = (u32, u32);

pub trait Tokenizer: Send + Sync {
    fn name(&self) -> &'static str;

    /// Append token spans to `out`. `out` is not cleared, so one buffer can be
    /// reused across an entire corpus.
    fn tokenize(&self, text: &str, out: &mut Vec<TokenSpan>) -> Result<()>;
}

/// Runs of alphanumeric characters, plus any configured continuation
/// characters.
///
/// With the default `keep = ["_"]` this reproduces the token stream of
/// `re.findall(r"\w+", text.lower())`, which is what most BM25 implementations
/// use and what makes results comparable against them.
///
/// Measured on 2,964 HotpotQA paragraphs, the only remaining divergence from
/// Python is `İ` (U+0130): Python lowercases the whole string *before*
/// tokenizing, so the combining dot its lowercase produces splits the word,
/// while this tokenizer lowercases each token afterwards. One document in
/// 2,964. Matching it exactly would mean allocating a lowercased copy of every
/// document, which is not worth it -- but see [`Unicode`] for a tokenizer that
/// handles marks properly by construction.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimpleConfig {
    /// Drop tokens shorter than this, in bytes.
    #[serde(default = "one")]
    pub min_len: usize,
    /// Characters that continue a token although they are not alphanumeric.
    #[serde(default = "default_keep")]
    pub keep: Vec<char>,
}

fn one() -> usize {
    1
}
fn default_keep() -> Vec<char> {
    vec!['_']
}

impl Default for SimpleConfig {
    fn default() -> Self {
        Self { min_len: one(), keep: default_keep() }
    }
}

#[derive(Debug, Clone)]
pub struct Simple {
    pub min_len: usize,
    pub keep: Vec<char>,
}

impl Default for Simple {
    fn default() -> Self {
        Self { min_len: 1, keep: default_keep() }
    }
}

impl Component for Simple {
    type Config = SimpleConfig;
    const NAME: &'static str = "simple";
    const SUMMARY: &'static str =
        "Alphanumeric runs plus configured characters; matches the conventional \\w+ tokenizer.";
    fn build(config: Self::Config) -> Result<Self> {
        Ok(Self { min_len: config.min_len, keep: config.keep })
    }
}

impl Simple {
    fn is_word(&self, c: char) -> bool {
        c.is_alphanumeric() || self.keep.contains(&c)
    }
}

impl Tokenizer for Simple {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn tokenize(&self, text: &str, out: &mut Vec<TokenSpan>) -> Result<()> {
        let mut start: Option<usize> = None;
        for (i, ch) in text.char_indices() {
            if self.is_word(ch) {
                start.get_or_insert(i);
            } else if let Some(s) = start.take()
                && i - s >= self.min_len
            {
                out.push((s as u32, i as u32));
            }
        }
        if let Some(s) = start
            && text.len() - s >= self.min_len
        {
            out.push((s as u32, text.len() as u32));
        }
        Ok(())
    }
}

/// Unicode word segmentation, UAX #29.
///
/// Handles combining marks, CJK and apostrophes by the standard's rules rather
/// than by a character-class approximation: `don't` stays one token, and
/// decomposed accents never split a word. It segments differently enough from
/// [`Simple`] that scores are not comparable between the two -- which is a
/// reason to measure, not a reason to prefer one blindly.
#[derive(Debug, Clone, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct UnicodeConfig {
    #[serde(default)]
    pub min_len: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Unicode {
    pub min_len: usize,
}

impl Component for Unicode {
    type Config = UnicodeConfig;
    const NAME: &'static str = "unicode";
    const SUMMARY: &'static str = "UAX #29 word segmentation; correct for marks, CJK and apostrophes.";
    fn build(config: Self::Config) -> Result<Self> {
        Ok(Self { min_len: config.min_len })
    }
}

impl Tokenizer for Unicode {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn tokenize(&self, text: &str, out: &mut Vec<TokenSpan>) -> Result<()> {
        for (i, word) in text.split_word_bound_indices() {
            // UAX #29 emits punctuation and whitespace as their own bounds;
            // keep only segments carrying a letter or a digit.
            if word.len() >= self.min_len && word.chars().any(char::is_alphanumeric) {
                out.push((i as u32, (i + word.len()) as u32));
            }
        }
        Ok(())
    }
}

/// Every built-in tokenizer.
pub fn registry() -> Registry<dyn Tokenizer> {
    let mut r = Registry::<dyn Tokenizer>::new("tokenizer");
    r.register::<Simple>(|c| Box::new(c)).expect("builtin");
    r.register::<Unicode>(|c| Box::new(c)).expect("builtin");
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks<'a>(t: &impl Tokenizer, s: &'a str) -> Vec<&'a str> {
        let mut v = Vec::new();
        t.tokenize(s, &mut v).unwrap();
        v.into_iter().map(|(a, b)| &s[a as usize..b as usize]).collect()
    }

    #[test]
    fn splits_on_punctuation_and_whitespace() {
        let t = Simple::default();
        assert_eq!(toks(&t, "the quick, brown fox!"), ["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn keeps_digits_and_accented_letters_whole() {
        let t = Simple::default();
        assert_eq!(toks(&t, "café 2026 naïve"), ["café", "2026", "naïve"]);
    }

    #[test]
    fn min_len_drops_short_tokens() {
        let t = Simple { min_len: 3, keep: vec![] };
        assert_eq!(toks(&t, "a an and ands"), ["and", "ands"]);
    }

    #[test]
    fn spans_are_reusable_across_texts() {
        let t = Simple::default();
        let mut out = Vec::new();
        t.tokenize("one two", &mut out).unwrap();
        let n = out.len();
        t.tokenize("three", &mut out).unwrap();
        assert_eq!(out.len(), n + 1, "tokenize must append, not clear");
    }

    #[test]
    fn simple_keeps_underscores_so_it_matches_the_conventional_tokenizer() {
        let t = Simple::default();
        assert_eq!(toks(&t, "united_states formula_1"), ["united_states", "formula_1"]);
        // Measured: underscores were 4 of the 5 divergences from Python's
        // \w+ across 2,964 HotpotQA paragraphs, and were enough to shift
        // document frequencies and therefore every IDF that used them.
        let bare = Simple { min_len: 1, keep: vec![] };
        assert_eq!(toks(&bare, "united_states"), ["united", "states"]);
    }

    #[test]
    fn unicode_tokenizer_keeps_marks_and_apostrophes_together() {
        // Regression: Hebrew with niqqud. The vowel points are category Mn and
        // are not alphanumeric, so the previous implementation split this into
        // several tokens, corrupting the document length and every BM25 score
        // that depended on it.
        let t = Unicode::default();
        // Hebrew with niqqud: the vowel points are category Mn.
        assert_eq!(toks(&t, "\u{5de}\u{5b0}\u{5e9}\u{5c1}\u{5b4}\u{5db}\u{5bc}\u{5b8}\u{5df}").len(), 1);
        // Decomposed Latin: "e" + combining acute stays one token.
        assert_eq!(toks(&t, "cafe\u{301} bar"), ["cafe\u{301}", "bar"]);
        // And the difference from `simple`, which splits both.
        assert_eq!(toks(&t, "don't"), ["don't"]);
        assert_eq!(toks(&Simple::default(), "don't"), ["don", "t"]);
    }

    #[test]
    fn empty_and_punctuation_only_text_yields_nothing() {
        let t = Simple::default();
        assert!(toks(&t, "").is_empty());
        assert!(toks(&t, " ,.;! ").is_empty());
    }
}
