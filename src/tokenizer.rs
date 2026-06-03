//! Minimal WordPiece tokenizer with Unicode support.
//!
//! The tokenizer keeps the runtime small by loading `vocab.txt` from caller
//! bytes instead of embedding a vocabulary in the WASM module. It implements
//! the BERT-style path needed by small encoder models:
//!
//! 1. Replace Unicode control characters with spaces.
//! 2. Insert spaces around CJK characters so each becomes its own token.
//! 3. Split on Unicode whitespace.
//! 4. For each whitespace-delimited word: optional Unicode lowercasing,
//!    then split off Unicode punctuation as individual tokens.
//! 5. Greedy longest-match WordPiece against the loaded vocabulary.
//! 6. Wrap with `[CLS]` / `[SEP]` and optionally pad to a fixed length.
//!
//! The character-class checks rely on `std::char` rather than a Unicode
//! database crate. This keeps the WASM bundle a few kilobytes lighter at
//! the cost of a small approximation of HuggingFace's Unicode punctuation
//! category — see `is_punctuation` for details. Accent stripping (NFD +
//! combining-mark removal) is intentionally not implemented; pick a
//! `*-cased` multilingual vocabulary if your text has accents.

use crate::error::{Error, Result};

/// Token ids and masks produced by [`WordPieceTokenizer`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedInput {
    /// Token ids, including `[CLS]` and `[SEP]`.
    pub input_ids: Vec<u32>,
    /// BERT token type ids. Single-sentence inputs are all zeros.
    pub token_type_ids: Vec<u32>,
    /// Attention mask: `1` for real tokens, `0` for padding.
    pub attention_mask: Vec<u32>,
}

/// Options for [`WordPieceTokenizer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordPieceOptions {
    /// Lowercase via Unicode rules before WordPiece matching.
    pub lowercase: bool,
    /// Maximum number of chars in one basic token before it becomes `[UNK]`.
    pub max_input_chars_per_word: usize,
}

impl Default for WordPieceOptions {
    fn default() -> Self {
        Self {
            lowercase: true,
            max_input_chars_per_word: 100,
        }
    }
}

/// A compact BERT WordPiece tokenizer backed by a sorted vocabulary.
///
/// Construct from a HuggingFace-style `vocab.txt` file. The first line has id
/// `0`, the second line has id `1`, and so on.
#[derive(Clone, Debug)]
pub struct WordPieceTokenizer {
    vocab: Vec<(String, u32)>,
    options: WordPieceOptions,
    pad_id: u32,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
}

impl WordPieceTokenizer {
    /// Builds a tokenizer from UTF-8 `vocab.txt` bytes using default options.
    pub fn from_vocab_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_vocab_bytes_with_options(bytes, WordPieceOptions::default())
    }

    /// Builds a tokenizer from UTF-8 `vocab.txt` bytes and explicit options.
    pub fn from_vocab_bytes_with_options(bytes: &[u8], options: WordPieceOptions) -> Result<Self> {
        let text = core::str::from_utf8(bytes)
            .map_err(|_| Error::InvalidTokenizer("vocab is not valid UTF-8"))?;

        let mut vocab = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let token = line.trim_end_matches('\r');
            if token.is_empty() {
                return Err(Error::InvalidTokenizer("vocab contains an empty token"));
            }
            let id = u32::try_from(i)
                .map_err(|_| Error::InvalidTokenizer("vocab has too many tokens"))?;
            vocab.push((token.to_string(), id));
        }

        if vocab.is_empty() {
            return Err(Error::InvalidTokenizer("vocab is empty"));
        }

        vocab.sort_by(|a, b| a.0.cmp(&b.0));
        for pair in vocab.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(Error::InvalidTokenizer("vocab contains duplicate tokens"));
            }
        }

        let pad_id =
            find_id_in(&vocab, "[PAD]").ok_or(Error::InvalidTokenizer("vocab is missing [PAD]"))?;
        let unk_id =
            find_id_in(&vocab, "[UNK]").ok_or(Error::InvalidTokenizer("vocab is missing [UNK]"))?;
        let cls_id =
            find_id_in(&vocab, "[CLS]").ok_or(Error::InvalidTokenizer("vocab is missing [CLS]"))?;
        let sep_id =
            find_id_in(&vocab, "[SEP]").ok_or(Error::InvalidTokenizer("vocab is missing [SEP]"))?;
        let mask_id = find_id_in(&vocab, "[MASK]")
            .ok_or(Error::InvalidTokenizer("vocab is missing [MASK]"))?;

        Ok(Self {
            vocab,
            options,
            pad_id,
            unk_id,
            cls_id,
            sep_id,
            mask_id,
        })
    }

    /// Number of entries in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Returns `true` if the vocabulary has no entries.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Returns the id for a token string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        find_id_in(&self.vocab, token)
    }

    /// Id of `[PAD]`.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Id of `[UNK]`.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Id of `[CLS]`.
    pub fn cls_id(&self) -> u32 {
        self.cls_id
    }

    /// Id of `[SEP]`.
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }

    /// Id of `[MASK]`.
    pub fn mask_id(&self) -> u32 {
        self.mask_id
    }

    /// Encodes one text into BERT input arrays without padding.
    ///
    /// `max_len` includes `[CLS]` and `[SEP]`; excess WordPiece tokens are
    /// truncated before the final `[SEP]`.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<EncodedInput> {
        if max_len < 2 {
            return Err(Error::InvalidTokenizer("max_len must be at least 2"));
        }

        let mut input_ids = Vec::new();
        input_ids.push(self.cls_id);
        let token_budget = max_len - 2;

        for token in basic_tokens(text, self.options.lowercase) {
            if input_ids.len() > token_budget {
                break;
            }
            let remaining = token_budget - (input_ids.len() - 1);
            self.push_wordpiece_ids(&token, remaining, &mut input_ids);
        }

        input_ids.truncate(max_len - 1);
        input_ids.push(self.sep_id);
        let token_type_ids = vec![0u32; input_ids.len()];
        let attention_mask = vec![1u32; input_ids.len()];

        Ok(EncodedInput {
            input_ids,
            token_type_ids,
            attention_mask,
        })
    }

    /// Encodes one text into BERT input arrays padded to exactly `max_len`.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<EncodedInput> {
        let mut encoded = self.encode(text, max_len)?;
        while encoded.input_ids.len() < max_len {
            encoded.input_ids.push(self.pad_id);
            encoded.token_type_ids.push(0);
            encoded.attention_mask.push(0);
        }
        Ok(encoded)
    }

    fn push_wordpiece_ids(&self, token: &str, limit: usize, out: &mut Vec<u32>) {
        if limit == 0 {
            return;
        }

        let chars: Vec<char> = token.chars().collect();
        if chars.len() > self.options.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        let mut start = 0;
        let mut pieces = Vec::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched: Option<(String, u32)> = None;

            while start < end {
                let mut piece = String::new();
                if start > 0 {
                    piece.push_str("##");
                }
                for ch in &chars[start..end] {
                    piece.push(*ch);
                }

                if let Some(id) = self.token_id(&piece) {
                    matched = Some((piece, id));
                    break;
                }
                end -= 1;
            }

            match matched {
                Some((piece, id)) => {
                    pieces.push((piece, id));
                    start = end;
                }
                None => {
                    out.push(self.unk_id);
                    return;
                }
            }
        }

        for (_, id) in pieces.into_iter().take(limit) {
            out.push(id);
        }
    }
}

fn find_id_in(vocab: &[(String, u32)], token: &str) -> Option<u32> {
    vocab
        .binary_search_by(|(candidate, _)| candidate.as_str().cmp(token))
        .ok()
        .map(|idx| vocab[idx].1)
}

/// Performs BERT's basic tokenization step.
///
/// 1. Control characters become spaces.
/// 2. CJK characters are surrounded by spaces (one token per char).
/// 3. Whitespace splits words.
/// 4. Within each word: optional Unicode lowercase, then punctuation is
///    split out as individual tokens.
fn basic_tokens(text: &str, lowercase: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut after_space = true;

    for raw_ch in text.chars() {
        let ch = if is_control(raw_ch) { ' ' } else { raw_ch };

        if is_cjk(ch) {
            flush_token(&mut current, &mut out);
            out.push(ch.to_string());
            after_space = true;
            continue;
        }

        if ch.is_whitespace() {
            flush_token(&mut current, &mut out);
            after_space = true;
            continue;
        }

        after_space = false;
        if lowercase {
            for lower in ch.to_lowercase() {
                emit_normalized(lower, &mut current, &mut out);
            }
        } else {
            emit_normalized(ch, &mut current, &mut out);
        }
    }

    flush_token(&mut current, &mut out);
    let _ = after_space;
    out
}

fn emit_normalized(ch: char, current: &mut String, out: &mut Vec<String>) {
    if is_punctuation(ch) {
        flush_token(current, out);
        out.push(ch.to_string());
    } else {
        current.push(ch);
    }
}

fn flush_token(current: &mut String, out: &mut Vec<String>) {
    if !current.is_empty() {
        out.push(core::mem::take(current));
    }
}

fn is_control(ch: char) -> bool {
    if matches!(ch, '\t' | '\n' | '\r') {
        return false;
    }
    ch.is_control()
}

/// Matches HuggingFace's `_is_chinese_char`. Hiragana, Katakana, Hangul,
/// and CJK punctuation are intentionally NOT included — BERT treats those
/// as regular characters or punctuation, not per-character tokens.
fn is_cjk(ch: char) -> bool {
    let cp = ch as u32;
    matches!(
        cp,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// HuggingFace-compatible punctuation predicate without a Unicode database.
///
/// True for: ASCII punctuation ranges (`!`–`/`, `:`–`@`, `[`–`\``, `{`–`~`)
/// and any non-ASCII character that is neither alphanumeric nor whitespace
/// nor a control character. This approximates HF's Unicode category-`P*`
/// check; it covers every real punctuation glyph we have observed (CJK
/// punctuation, Arabic comma, Spanish `¿`/`¡`, French guillemets) at the
/// cost of also marking some symbol-class characters (currency signs,
/// arrows, emoji) as punctuation, which BERT treats as its own tokens too.
fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    if cp < 128 {
        return false;
    }
    !ch.is_alphanumeric() && !ch.is_whitespace() && !ch.is_control() && !is_cjk(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> WordPieceTokenizer {
        WordPieceTokenizer::from_vocab_bytes(
            b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n##s\n,\nun\n##aff\n##able\n",
        )
        .unwrap()
    }

    #[test]
    fn parses_vocab_and_special_ids() {
        let tokenizer = test_tokenizer();
        assert_eq!(tokenizer.len(), 12);
        assert_eq!(tokenizer.pad_id(), 0);
        assert_eq!(tokenizer.unk_id(), 1);
        assert_eq!(tokenizer.cls_id(), 2);
        assert_eq!(tokenizer.sep_id(), 3);
        assert_eq!(tokenizer.mask_id(), 4);
        assert_eq!(tokenizer.token_id("hello"), Some(5));
    }

    #[test]
    fn encodes_with_basic_split_and_wordpieces() {
        let tokenizer = test_tokenizer();
        let encoded = tokenizer.encode("Hello, worlds", 8).unwrap();
        assert_eq!(encoded.input_ids, vec![2, 5, 8, 6, 7, 3]);
        assert_eq!(encoded.token_type_ids, vec![0; 6]);
        assert_eq!(encoded.attention_mask, vec![1; 6]);
    }

    #[test]
    fn encodes_unknown_token() {
        let tokenizer = test_tokenizer();
        let encoded = tokenizer.encode("missing", 8).unwrap();
        assert_eq!(encoded.input_ids, vec![2, 1, 3]);
    }

    #[test]
    fn pads_to_requested_length() {
        let tokenizer = test_tokenizer();
        let encoded = tokenizer.encode_padded("hello", 6).unwrap();
        assert_eq!(encoded.input_ids, vec![2, 5, 3, 0, 0, 0]);
        assert_eq!(encoded.attention_mask, vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn truncates_before_sep() {
        let tokenizer = test_tokenizer();
        let encoded = tokenizer.encode("hello hello hello", 4).unwrap();
        assert_eq!(encoded.input_ids, vec![2, 5, 5, 3]);
    }

    #[test]
    fn basic_tokens_unicode_lowercases_cyrillic() {
        // Russian word "Привет" must lowercase to "привет".
        let tokens = basic_tokens("Привет", true);
        assert_eq!(tokens, vec!["привет"]);
    }

    #[test]
    fn basic_tokens_unicode_lowercases_spanish_n_tilde() {
        // Spanish "ESPAÑOL" -> "español".
        let tokens = basic_tokens("ESPAÑOL", true);
        assert_eq!(tokens, vec!["español"]);
    }

    #[test]
    fn basic_tokens_splits_cjk_chars() {
        // Each CJK char becomes its own token.
        let tokens = basic_tokens("你好世界", true);
        assert_eq!(tokens, vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn basic_tokens_splits_mixed_cjk_and_latin() {
        let tokens = basic_tokens("hello 你好 world", true);
        assert_eq!(tokens, vec!["hello", "你", "好", "world"]);
    }

    #[test]
    fn basic_tokens_treats_unicode_whitespace_as_break() {
        // U+00A0 NO-BREAK SPACE and U+3000 IDEOGRAPHIC SPACE.
        let tokens = basic_tokens("hello\u{00A0}world\u{3000}foo", true);
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn basic_tokens_splits_non_ascii_punctuation() {
        // Chinese full-width comma is treated as its own token.
        let tokens = basic_tokens("hello，world", true);
        assert_eq!(tokens, vec!["hello", "\u{ff0c}", "world"]);
    }

    #[test]
    fn basic_tokens_keeps_spanish_inverted_marks_separate() {
        // Inverted question/exclamation marks are punctuation.
        let tokens = basic_tokens("¿hola?", true);
        assert_eq!(tokens, vec!["¿", "hola", "?"]);
    }
}
