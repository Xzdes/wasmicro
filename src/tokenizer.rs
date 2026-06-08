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
/// Returns `true` if this byte is directly representable as itself in GPT-2's
/// byte-level vocabulary (printable ASCII + printable Latin-1).
fn is_printable_byte(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Byte-level BPE tokenizer (GPT-2 / RoBERTa compatible)
// ─────────────────────────────────────────────────────────────────────────────

/// Byte-level BPE tokenizer compatible with GPT-2 and RoBERTa.
///
/// Load from the two files that HuggingFace distributes with every GPT-2
/// model: `vocab.json` (token-string → id mapping) and `merges.txt`
/// (ordered BPE merge rules).
///
/// # Byte-level encoding
///
/// GPT-2 maps each input byte to a unique Unicode character before applying
/// BPE. This means every byte sequence is representable without an `[UNK]`
/// token — the vocabulary only needs to contain the 256 single-byte tokens
/// plus the merged tokens.
///
/// # Limitations
///
/// The pre-tokenization uses a simple whitespace-split approach rather than
/// GPT-2's full regex pattern. This covers the common case well; contraction
/// splitting (`it's` → `it` + `'s`) is not performed.
pub mod bpe {
    use crate::error::{Error, Result};
    use std::collections::HashMap;

    /// Token ids and masks produced by [`BpeTokenizer`].
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct BpeEncodedInput {
        /// Token ids including BOS (if present).
        pub input_ids: Vec<u32>,
        /// `1` for real tokens, `0` for padding.
        pub attention_mask: Vec<u32>,
    }

    /// Byte-level BPE tokenizer.
    #[derive(Clone, Debug)]
    pub struct BpeTokenizer {
        vocab: HashMap<String, u32>,
        decoder: Vec<String>,
        merges: HashMap<(String, String), u32>,
        byte_encoder: [char; 256],
        byte_decoder: HashMap<char, u8>,
        /// BOS / EOS token id (GPT-2 uses `<|endoftext|>` for both).
        pub eos_token_id: Option<u32>,
        /// BOS token id.
        pub bos_token_id: Option<u32>,
        /// PAD token id (not all models have one).
        pub pad_token_id: Option<u32>,
    }

    impl BpeTokenizer {
        /// Constructs a tokenizer from `vocab.json` and `merges.txt` bytes.
        ///
        /// `vocab_bytes` must be a UTF-8 JSON object `{"token": id, ...}`.
        /// `merges_bytes` must be the `merges.txt` format: one `"left right"` merge per line,
        /// comment lines starting with `#` are ignored.
        pub fn from_bytes(vocab_bytes: &[u8], merges_bytes: &[u8]) -> Result<Self> {
            let vocab = parse_vocab_json(vocab_bytes)?;
            let merges = parse_merges_txt(merges_bytes)?;
            let byte_encoder = make_byte_encoder();
            let byte_decoder = make_byte_decoder(&byte_encoder);

            let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
            let mut decoder = vec![String::new(); max_id + 1];
            for (token, &id) in &vocab {
                decoder[id as usize] = token.clone();
            }

            let eos_token_id = vocab.get("<|endoftext|>").copied();
            let bos_token_id = eos_token_id;
            let pad_token_id = vocab.get("<|padding|>").copied();

            Ok(Self {
                vocab,
                decoder,
                merges,
                byte_encoder,
                byte_decoder,
                eos_token_id,
                bos_token_id,
                pad_token_id,
            })
        }

        /// Vocabulary size.
        pub fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        /// Encodes `text` into token ids. Appends EOS if present in vocab.
        /// Truncates to `max_len` (including EOS).
        pub fn encode(&self, text: &str, max_len: usize) -> Result<BpeEncodedInput> {
            if max_len == 0 {
                return Err(Error::InvalidTokenizer("max_len must be > 0"));
            }
            let mut ids: Vec<u32> = Vec::new();

            for word in pre_tokenize(text) {
                let byte_chars: Vec<char> =
                    word.bytes().map(|b| self.byte_encoder[b as usize]).collect();
                let tokens = self.bpe_merge(byte_chars);
                for tok in &tokens {
                    if ids.len() >= max_len.saturating_sub(self.eos_token_id.is_some() as usize) {
                        break;
                    }
                    let id = self.vocab.get(tok.as_str()).copied().unwrap_or_else(|| {
                        // Per-char fallback: encode each char separately
                        self.vocab
                            .get(&tok.chars().next().unwrap_or(' ').to_string())
                            .copied()
                            .unwrap_or(0)
                    });
                    ids.push(id);
                }
            }

            if let Some(eos) = self.eos_token_id {
                if ids.len() < max_len {
                    ids.push(eos);
                }
            }

            let len = ids.len();
            Ok(BpeEncodedInput {
                attention_mask: vec![1u32; len],
                input_ids: ids,
            })
        }

        /// Encodes and pads to exactly `max_len` using the pad token (0 if absent).
        pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<BpeEncodedInput> {
            let mut enc = self.encode(text, max_len)?;
            let pad = self.pad_token_id.unwrap_or(0);
            while enc.input_ids.len() < max_len {
                enc.input_ids.push(pad);
                enc.attention_mask.push(0);
            }
            Ok(enc)
        }

        /// Decodes token ids back to a UTF-8 string. Invalid byte sequences
        /// are replaced with the Unicode replacement character.
        pub fn decode(&self, ids: &[u32]) -> String {
            let mut bytes: Vec<u8> = Vec::new();
            for &id in ids {
                let Some(token) = self.decoder.get(id as usize) else {
                    continue;
                };
                // Skip special tokens (they don't round-trip through byte_decoder)
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                for c in token.chars() {
                    if let Some(&b) = self.byte_decoder.get(&c) {
                        bytes.push(b);
                    }
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }

        /// Returns the id for a token string, or `None` if not in vocab.
        pub fn token_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }

        /// Applies BPE merges to a sequence of single-character tokens.
        fn bpe_merge(&self, chars: Vec<char>) -> Vec<String> {
            if chars.is_empty() {
                return vec![];
            }
            let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();

            loop {
                let mut best_rank = u32::MAX;
                let mut best_idx = usize::MAX;

                for i in 0..tokens.len().saturating_sub(1) {
                    if let Some(&rank) =
                        self.merges.get(&(tokens[i].clone(), tokens[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }

                if best_idx == usize::MAX {
                    break;
                }

                let left = tokens[best_idx].clone();
                let right = tokens[best_idx + 1].clone();
                let merged = format!("{left}{right}");

                let mut new_tokens = Vec::with_capacity(tokens.len());
                let mut i = 0;
                while i < tokens.len() {
                    if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                        new_tokens.push(merged.clone());
                        i += 2;
                    } else {
                        new_tokens.push(tokens[i].clone());
                        i += 1;
                    }
                }
                tokens = new_tokens;
            }

            tokens
        }
    }

    /// Splits text into pre-tokenization units, keeping a leading space
    /// attached to the following word (GPT-2 / Ġ convention).
    fn pre_tokenize(text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }
        let mut words = Vec::new();
        let mut current = String::new();
        let mut pending_space = false;

        for c in text.chars() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                if !current.is_empty() {
                    words.push(core::mem::take(&mut current));
                }
                pending_space = true;
            } else {
                if pending_space {
                    current.push(' ');
                    pending_space = false;
                }
                current.push(c);
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    /// Builds the byte → Unicode character table used by GPT-2.
    ///
    /// Printable bytes (33–126, 161–172, 174–255) map to themselves.
    /// The remaining 68 bytes map to U+0100–U+0143 in byte order.
    fn make_byte_encoder() -> [char; 256] {
        let mut encoder = ['\0'; 256];
        let mut extra = 256u32;
        for b in 0u8..=255 {
            if super::is_printable_byte(b) {
                encoder[b as usize] = char::from_u32(b as u32).unwrap_or('\0');
            } else {
                encoder[b as usize] = char::from_u32(extra).unwrap_or('\0');
                extra += 1;
            }
        }
        encoder
    }

    fn make_byte_decoder(encoder: &[char; 256]) -> HashMap<char, u8> {
        encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect()
    }

    /// Minimal parser for `{"token": id, ...}` JSON (vocab.json format).
    fn parse_vocab_json(bytes: &[u8]) -> Result<HashMap<String, u32>> {
        let text = core::str::from_utf8(bytes)
            .map_err(|_| Error::InvalidTokenizer("vocab.json is not valid UTF-8"))?;
        let text = text.trim();
        if !text.starts_with('{') {
            return Err(Error::InvalidTokenizer("vocab.json: expected '{'"));
        }

        let mut map = HashMap::new();
        let tb = text.as_bytes();
        let mut pos = 1usize; // skip '{'

        loop {
            // Skip whitespace and commas
            while pos < tb.len()
                && matches!(tb[pos], b' ' | b'\t' | b'\n' | b'\r' | b',')
            {
                pos += 1;
            }
            if pos >= tb.len() || tb[pos] == b'}' {
                break;
            }

            // Parse string key
            if tb[pos] != b'"' {
                return Err(Error::InvalidTokenizer("vocab.json: expected string key"));
            }
            pos += 1; // skip opening "
            let (key, consumed) = parse_json_string(&text[pos..])?;
            pos += consumed;

            // Skip whitespace, then ':'
            while pos < tb.len() && matches!(tb[pos], b' ' | b'\t') {
                pos += 1;
            }
            if pos >= tb.len() || tb[pos] != b':' {
                return Err(Error::InvalidTokenizer("vocab.json: expected ':'"));
            }
            pos += 1;

            // Skip whitespace
            while pos < tb.len() && matches!(tb[pos], b' ' | b'\t') {
                pos += 1;
            }

            // Parse non-negative integer
            let start = pos;
            while pos < tb.len() && tb[pos].is_ascii_digit() {
                pos += 1;
            }
            if start == pos {
                return Err(Error::InvalidTokenizer("vocab.json: expected integer value"));
            }
            let id: u32 = text[start..pos]
                .parse()
                .map_err(|_| Error::InvalidTokenizer("vocab.json: token id overflow"))?;

            map.insert(key, id);
        }

        Ok(map)
    }

    /// Parses a JSON string starting *after* the opening `"`.
    /// Returns `(string, bytes_consumed_including_closing_quote)`.
    fn parse_json_string(s: &str) -> Result<(String, usize)> {
        let mut result = String::new();
        let mut iter = s.char_indices();
        loop {
            match iter.next() {
                None => return Err(Error::InvalidTokenizer("vocab.json: unterminated string")),
                Some((idx, '"')) => return Ok((result, idx + 1)),
                Some((_, '\\')) => match iter.next() {
                    None => return Err(Error::InvalidTokenizer("vocab.json: bad escape")),
                    Some((_, 'n')) => result.push('\n'),
                    Some((_, 't')) => result.push('\t'),
                    Some((_, 'r')) => result.push('\r'),
                    Some((_, '"')) => result.push('"'),
                    Some((_, '\\')) => result.push('\\'),
                    Some((_, '/')) => result.push('/'),
                    Some((_, 'u')) => {
                        let hex: String = iter.by_ref().take(4).map(|(_, c)| c).collect();
                        if hex.len() != 4 {
                            return Err(Error::InvalidTokenizer("vocab.json: short \\u escape"));
                        }
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| Error::InvalidTokenizer("vocab.json: bad \\u hex"))?;
                        result.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    Some((_, c)) => result.push(c),
                },
                Some((_, c)) => result.push(c),
            }
        }
    }

    /// Parses `merges.txt`: one `"left right"` merge per line, `#` = comment.
    fn parse_merges_txt(bytes: &[u8]) -> Result<HashMap<(String, String), u32>> {
        let text = core::str::from_utf8(bytes)
            .map_err(|_| Error::InvalidTokenizer("merges.txt is not valid UTF-8"))?;
        let mut merges = HashMap::new();
        let mut rank = 0u32;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let sep = line
                .find(' ')
                .ok_or(Error::InvalidTokenizer("merges.txt: line missing space"))?;
            let left = line[..sep].to_string();
            let right = line[sep + 1..].to_string();
            merges.insert((left, right), rank);
            rank += 1;
        }
        Ok(merges)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn tiny_vocab_and_merges() -> (Vec<u8>, Vec<u8>) {
            // Minimal vocab.json covering letters h, e, l, o, space (Ġ = byte 32 → U+0120)
            let vocab = r#"{"h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "he": 5, "hel": 6, "hell": 7, "hello": 8, "Ġworld": 9, "w": 10, "r": 11, "d": 12, "wo": 13, "wor": 14, "worl": 15, "world": 16, "<|endoftext|>": 50256}"#;
            let merges = b"#version: 0\nh e\nhe l\nhel l\nhell o\n\xc4\xa0 w\nwo r\nwor l\nworl d\n";
            // Note: Ġ = U+0120 = 0xC4 0xA0 in UTF-8
            (vocab.as_bytes().to_vec(), merges.to_vec())
        }

        #[test]
        fn byte_encoder_roundtrip() {
            let enc = make_byte_encoder();
            let dec = make_byte_decoder(&enc);
            for b in 0u8..=255 {
                let c = enc[b as usize];
                assert_eq!(dec[&c], b, "byte {b} roundtrip failed");
            }
        }

        #[test]
        fn pre_tokenize_attaches_space_to_next_word() {
            let words = pre_tokenize("hello world");
            assert_eq!(words, vec!["hello", " world"]);
        }

        #[test]
        fn pre_tokenize_single_word() {
            let words = pre_tokenize("hello");
            assert_eq!(words, vec!["hello"]);
        }

        #[test]
        fn encode_hello_merges_to_single_token() {
            let (vocab, merges) = tiny_vocab_and_merges();
            let tok = BpeTokenizer::from_bytes(&vocab, &merges).unwrap();
            // h+e→he, he+l→hel, hel+l→hell, hell+o→hello → id 8, then EOS 50256 is appended
            let enc = tok.encode("hello", 128).unwrap();
            assert_eq!(enc.input_ids[0], 8);
            assert_eq!(*enc.input_ids.last().unwrap(), 50256); // <|endoftext|>
        }

        #[test]
        fn parse_merges_counts_ranks() {
            let txt = b"#version: gpt2\na b\nc d\ne f\n";
            let m = parse_merges_txt(txt).unwrap();
            assert_eq!(m[&("a".to_string(), "b".to_string())], 0);
            assert_eq!(m[&("c".to_string(), "d".to_string())], 1);
            assert_eq!(m[&("e".to_string(), "f".to_string())], 2);
        }
    }
}
