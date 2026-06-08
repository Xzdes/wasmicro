//! High-level pipeline API — one call to load and run any supported model.
//!
//! [`Pipeline`] wraps a model and its tokenizer under a single unified interface.
//! The model type is auto-detected from `config.json` via the `"model_type"` field.
//!
//! ## Supported model types
//!
//! | `model_type` | Tokenizer | Methods |
//! |---|---|---|
//! | `bert`, `roberta`, `distilbert`, `electra` | `vocab.txt` (WordPiece) | `embed`, `embed_batch` |
//! | `gpt2`, `gpt_neo`, `gpt_neox` | `vocab.json` + `merges.txt` combined | `generate` |
//! | `t5`, `mt5`, `longt5` | `vocab.json` + `merges.txt` combined | `encode`, `generate` |
//!
//! ## Quick start
//!
//! ```no_run
//! use std::fs;
//! use wasmicro::pipeline::Pipeline;
//!
//! // BERT embedding
//! let model_bytes = fs::read("model.safetensors").unwrap();
//! let tokenizer_bytes = fs::read("vocab.txt").unwrap();
//! let config_json = fs::read_to_string("config.json").unwrap();
//!
//! let pipeline = Pipeline::from_bytes(&model_bytes, &tokenizer_bytes, &config_json, None).unwrap();
//! let embedding = pipeline.embed("Hello world", 128).unwrap();
//!
//! // GPT-2 generation
//! let vocab_json = fs::read("vocab.json").unwrap();
//! let merges_txt = fs::read("merges.txt").unwrap();
//! let config_json = fs::read_to_string("config.json").unwrap();
//!
//! let gpt2_pipeline = Pipeline::from_bytes(&model_bytes, &vocab_json, &config_json, Some(&merges_txt)).unwrap();
//! let text = gpt2_pipeline.generate("Once upon a time", 50).unwrap();
//! ```

use crate::error::{Error, Result};
use crate::loader::ModelFile;
use crate::models::bert::{BertConfig, BertModel};
use crate::models::gpt2::{Gpt2Config, Gpt2Model};
use crate::models::t5::{T5Config, T5Model};
use crate::tensor::Tensor;
use crate::tokenizer::{WordPieceOptions, WordPieceTokenizer};
use crate::tokenizer::bpe::BpeTokenizer;

enum Inner {
    Bert {
        model: BertModel,
        tokenizer: WordPieceTokenizer,
    },
    Gpt2 {
        model: Gpt2Model,
        tokenizer: BpeTokenizer,
    },
    T5 {
        model: T5Model,
        tokenizer: BpeTokenizer,
    },
}

/// A ready-to-use model+tokenizer pair, auto-configured from `config.json`.
pub struct Pipeline {
    inner: Inner,
}

impl Pipeline {
    /// Creates a pipeline from raw bytes.
    ///
    /// # Arguments
    ///
    /// - `model_bytes` — contents of a `model.safetensors` file.
    /// - `tokenizer_bytes` — `vocab.txt` for BERT-family models;
    ///   `vocab.json` for GPT-2 / T5 models.
    /// - `config_json` — contents of `config.json` (determines model type).
    /// - `merges_bytes` — `merges.txt` for GPT-2 / T5 models; `None` for BERT.
    pub fn from_bytes(
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        config_json: &str,
        merges_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let model_type = detect_model_type(config_json);
        let file = ModelFile::parse(model_bytes)?;

        let inner = match model_type.as_str() {
            "bert" | "roberta" | "distilbert" | "electra" | "albert" => {
                let config = BertConfig::from_config_json(config_json)?;
                let model = BertModel::from_safetensors_auto(&file, config)?;
                let lowercase = !config_json.contains("\"uncased\"")
                    && config_json
                        .find("\"do_lower_case\"")
                        .and_then(|p| {
                            config_json[p..].find("true").map(|q| q < 20)
                        })
                        .unwrap_or(true);
                let tokenizer = WordPieceTokenizer::from_vocab_bytes_with_options(
                    tokenizer_bytes,
                    WordPieceOptions {
                        lowercase,
                        ..WordPieceOptions::default()
                    },
                )?;
                Inner::Bert { model, tokenizer }
            }
            "gpt2" | "gpt_neo" | "gpt_neox" | "gptj" => {
                let merges = merges_bytes.ok_or(Error::InvalidInput(
                    "GPT-2 models require merges_bytes (merges.txt)",
                ))?;
                let config = Gpt2Config::from_config_json(config_json)?;
                let model = Gpt2Model::from_safetensors(&file, config)?;
                let tokenizer = BpeTokenizer::from_bytes(tokenizer_bytes, merges)?;
                Inner::Gpt2 { model, tokenizer }
            }
            "t5" | "mt5" | "longt5" | "umt5" => {
                let merges = merges_bytes.ok_or(Error::InvalidInput(
                    "T5 models require merges_bytes (merges.txt)",
                ))?;
                let config = T5Config::from_config_json(config_json)?;
                let model = T5Model::from_safetensors(&file, config)?;
                let tokenizer = BpeTokenizer::from_bytes(tokenizer_bytes, merges)?;
                Inner::T5 { model, tokenizer }
            }
            _ => {
                return Err(Error::InvalidInput(
                    "unsupported model_type in config.json (supported: bert/roberta/distilbert/gpt2/t5)",
                ));
            }
        };

        Ok(Self { inner })
    }

    /// Returns the `model_type` string detected from `config.json`.
    pub fn detected_model_type(config_json: &str) -> String {
        detect_model_type(config_json)
    }

    // ── BERT-family ──────────────────────────────────────────────────────────

    /// Tokenizes `text` and returns a single pooled embedding `[hidden_size]`.
    ///
    /// Only valid for BERT-family models. Returns an error for GPT-2 / T5.
    pub fn embed(&self, text: &str, max_len: usize) -> Result<Tensor> {
        match &self.inner {
            Inner::Bert { model, tokenizer } => model.embed_text(tokenizer, text, max_len),
            _ => Err(Error::InvalidInput("embed() is only supported for BERT-family models")),
        }
    }

    /// Embeds a batch of texts, returning one vector per text.
    ///
    /// Each text is encoded independently (no shared padding). Only valid for
    /// BERT-family models.
    pub fn embed_batch(&self, texts: &[&str], max_len: usize) -> Result<Vec<Tensor>> {
        match &self.inner {
            Inner::Bert { model, tokenizer } => model.embed_batch(tokenizer, texts, max_len),
            _ => Err(Error::InvalidInput("embed_batch() is only supported for BERT-family models")),
        }
    }

    // ── GPT-2 / T5 ───────────────────────────────────────────────────────────

    /// Generates text greedily. `max_new_tokens` controls how many tokens to produce.
    ///
    /// For GPT-2: continues the prompt.
    /// For T5: treats `prompt` as the encoder input (task prefix included).
    ///
    /// Returns the generated string, not including the prompt for GPT-2 (includes
    /// the prompt for T5's encoder → decoder flow).
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        match &self.inner {
            Inner::Gpt2 { model, tokenizer } => {
                let enc = tokenizer.encode(prompt, model.config.max_position_embeddings)?;
                let all_ids = model.generate_greedy(&enc.input_ids, max_new_tokens);
                let new_ids = &all_ids[enc.input_ids.len()..];
                Ok(tokenizer.decode(new_ids))
            }
            Inner::T5 { model, tokenizer } => {
                let enc = tokenizer.encode(prompt, 512)?;
                let out_ids = model.generate_greedy(&enc.input_ids, max_new_tokens);
                Ok(tokenizer.decode(&out_ids))
            }
            Inner::Bert { .. } => {
                Err(Error::InvalidInput("generate() is not supported for BERT-family models"))
            }
        }
    }

    // ── T5 encode only ───────────────────────────────────────────────────────

    /// Runs only the T5 encoder, returning hidden states `[seq_len, d_model]`.
    ///
    /// Useful for embedding sentences with T5 without decoder overhead.
    /// Returns an error for non-T5 models.
    pub fn encode_t5(&self, text: &str, max_len: usize) -> Result<Tensor> {
        match &self.inner {
            Inner::T5 { model, tokenizer } => {
                let enc = tokenizer.encode(text, max_len)?;
                Ok(model.encode(&enc.input_ids))
            }
            _ => Err(Error::InvalidInput("encode_t5() is only valid for T5-family models")),
        }
    }
}

/// Extracts the `"model_type"` field from a config.json string.
/// Returns an empty string if not found.
fn detect_model_type(config_json: &str) -> String {
    let key = "\"model_type\":";
    let start = match config_json.find(key) {
        Some(p) => p + key.len(),
        None => return String::new(),
    };
    let rest = config_json[start..].trim_start();
    if !rest.starts_with('"') {
        return String::new();
    }
    let inner = &rest[1..];
    let end = inner.find('"').unwrap_or(inner.len());
    inner[..end].to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_model_type_bert() {
        let json = r#"{"model_type": "bert", "hidden_size": 768}"#;
        assert_eq!(detect_model_type(json), "bert");
    }

    #[test]
    fn detect_model_type_gpt2() {
        let json = r#"{"model_type":"gpt2","n_embd":768}"#;
        assert_eq!(detect_model_type(json), "gpt2");
    }

    #[test]
    fn detect_model_type_t5() {
        let json = r#"{"model_type": "t5", "d_model": 512}"#;
        assert_eq!(detect_model_type(json), "t5");
    }

    #[test]
    fn detect_model_type_missing() {
        assert_eq!(detect_model_type("{}"), "");
    }
}
