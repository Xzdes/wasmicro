//! WASM bindings — the full JavaScript-facing API.
//!
//! Compiled only when the `wasm` feature is enabled. The boundary is
//! intentionally narrow: plain Rust types live in the rest of the crate;
//! this module only wraps them for `wasm_bindgen`.
//!
//! ## Quick start (JavaScript)
//!
//! ```js
//! import init, { WasmPipeline } from "wasmicro";
//! await init();
//!
//! // BERT embedding — one call, config.json auto-detects everything
//! const pipeline = await WasmPipeline.fromBytes(
//!   modelBytes,          // Uint8Array — model.safetensors
//!   vocabBytes,          // Uint8Array — vocab.txt
//!   configJsonString,    // string     — config.json text
//!   null,                // Uint8Array | null — merges.txt (null for BERT)
//! );
//! const embedding = pipeline.embed("Hello world", 128);    // Float32Array
//! const batch     = pipeline.embedBatch(["a", "b"], 128);  // Float32Array[]
//!
//! // GPT-2 generation
//! const gpt2 = await WasmPipeline.fromBytes(modelBytes, vocabJsonBytes, config, mergesBytes);
//! const text = gpt2.generate("Once upon a time", 50);      // string
//! ```

use wasm_bindgen::prelude::*;

use crate::loader::ModelFile;
use crate::models::bert::{BertConfig, BertModel};
use crate::models::gpt2::{Gpt2Config, Gpt2Model};
use crate::models::t5::{T5Config, T5Model};
use crate::ops::matmul::matmul;
use crate::pipeline::Pipeline;
use crate::tensor::Tensor;
use crate::tokenizer::{EncodedInput, WordPieceOptions, WordPieceTokenizer};
use crate::tokenizer::bpe::BpeTokenizer;

// ─── utilities ───────────────────────────────────────────────────────────────

/// Initialise the WASM panic hook so Rust panics appear in the browser console.
/// Call once after `await init()`. No-op if the `wasm-debug` feature is off.
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "wasm-debug")]
    console_error_panic_hook::set_once();
}

/// Library version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Runs an `n × n` matrix multiply and returns the first output cell.
/// Used by the demo page to benchmark WASM performance from JavaScript.
#[wasm_bindgen]
pub fn matmul_bench(n: usize) -> Result<f32, JsError> {
    if n == 0 || n > 2048 {
        return Err(JsError::new("n must be in the range 1..=2048"));
    }
    let len = n.checked_mul(n).ok_or_else(|| JsError::new("matrix size overflow"))?;
    let a = Tensor::from_vec(vec![1.0; len], &[n, n]);
    let b = Tensor::from_vec(vec![1.0; len], &[n, n]);
    let c = matmul(&a, &b);
    Ok(c.data()[0])
}

// ─── WordPiece tokenizer ──────────────────────────────────────────────────────

/// WordPiece tokenizer (BERT-family models).
#[wasm_bindgen]
pub struct WasmWordPieceTokenizer {
    inner: WordPieceTokenizer,
}

#[wasm_bindgen]
impl WasmWordPieceTokenizer {
    /// Parses UTF-8 `vocab.txt` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &[u8], lowercase: bool) -> Result<WasmWordPieceTokenizer, JsError> {
        let opts = WordPieceOptions { lowercase, ..Default::default() };
        let inner = WordPieceTokenizer::from_vocab_bytes_with_options(vocab, opts)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encodes text without padding.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<WasmEncodedInput, JsError> {
        self.inner.encode(text, max_len)
            .map(WasmEncodedInput::new)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Encodes text and pads to exactly `max_len`.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<WasmEncodedInput, JsError> {
        self.inner.encode_padded(text, max_len)
            .map(WasmEncodedInput::new)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Returns the id for a token, or `undefined` if not in vocab.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_id(token)
    }
}

/// Encoded BERT input.
#[wasm_bindgen]
pub struct WasmEncodedInput {
    inner: EncodedInput,
}
impl WasmEncodedInput {
    fn new(inner: EncodedInput) -> Self { Self { inner } }
}
#[wasm_bindgen]
impl WasmEncodedInput {
    /// Token ids including `[CLS]` and `[SEP]`.
    pub fn input_ids(&self) -> Box<[u32]> { self.inner.input_ids.clone().into_boxed_slice() }
    /// BERT segment ids.
    pub fn token_type_ids(&self) -> Box<[u32]> { self.inner.token_type_ids.clone().into_boxed_slice() }
    /// Attention mask: `1` for real, `0` for padding.
    pub fn attention_mask(&self) -> Box<[u32]> { self.inner.attention_mask.clone().into_boxed_slice() }
}

// ─── BPE tokenizer ────────────────────────────────────────────────────────────

/// Byte-level BPE tokenizer (GPT-2 / RoBERTa / T5).
#[wasm_bindgen]
pub struct WasmBpeTokenizer {
    inner: BpeTokenizer,
}

#[wasm_bindgen]
impl WasmBpeTokenizer {
    /// Constructs a tokenizer from `vocab.json` and `merges.txt` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(vocab_json: &[u8], merges_txt: &[u8]) -> Result<WasmBpeTokenizer, JsError> {
        let inner = BpeTokenizer::from_bytes(vocab_json, merges_txt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encodes text into token ids (including EOS if present in vocab).
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Box<[u32]>, JsError> {
        self.inner.encode(text, max_len)
            .map(|e| e.input_ids.into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Decodes token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids)
    }

    /// Returns the id for a token string, or `undefined` if absent.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_id(token)
    }

    /// EOS token id (`<|endoftext|>` for GPT-2), or `undefined`.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id
    }
}

// ─── BERT model ───────────────────────────────────────────────────────────────

/// BERT encoder (backward-compatible with v0.2 API).
#[wasm_bindgen]
pub struct WasmBertModel {
    inner: BertModel,
}

#[wasm_bindgen]
impl WasmBertModel {
    /// Loads a BERT model from safetensors bytes using explicit config fields.
    ///
    /// For new code, prefer [`WasmPipeline`] which parses `config.json`
    /// automatically and detects the tensor prefix.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bytes: &[u8],
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_position_embeddings: usize,
        type_vocab_size: usize,
        prefix: &str,
    ) -> Result<WasmBertModel, JsError> {
        let config = BertConfig {
            hidden_size, num_hidden_layers, num_attention_heads,
            intermediate_size, vocab_size, max_position_embeddings,
            type_vocab_size, layer_norm_eps: 1e-12,
        };
        let file = ModelFile::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        let inner = BertModel::from_safetensors(&file, config, prefix)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Loads a BERT model from safetensors bytes + `config.json` text.
    /// The tensor prefix is auto-detected.
    pub fn from_config(bytes: &[u8], config_json: &str) -> Result<WasmBertModel, JsError> {
        let config = BertConfig::from_config_json(config_json)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let file = ModelFile::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        let inner = BertModel::from_safetensors_auto(&file, config)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs the encoder, returns flat `[seq_len × hidden_size]` hidden states.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Box<[f32]>, JsError> {
        self.inner.try_forward(input_ids, None)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Encoder + mean-pool → `[hidden_size]` embedding.
    pub fn embed(&self, input_ids: &[u32]) -> Result<Box<[f32]>, JsError> {
        self.inner.try_embed_sentence(input_ids, None, None)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Encoder + mean-pool with an explicit attention mask.
    pub fn embed_with_mask(
        &self, input_ids: &[u32], attention_mask: &[u32],
    ) -> Result<Box<[f32]>, JsError> {
        self.inner.try_embed_sentence(input_ids, None, Some(attention_mask))
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Encoder + mean-pool from a tokenizer output.
    pub fn embed_encoded(&self, encoded: &WasmEncodedInput) -> Result<Box<[f32]>, JsError> {
        self.inner.try_embed_sentence(
            &encoded.inner.input_ids,
            Some(&encoded.inner.token_type_ids),
            Some(&encoded.inner.attention_mask),
        )
        .map(|t| t.data().to_vec().into_boxed_slice())
        .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Tokenizes text and returns one pooled embedding.
    pub fn embed_text(
        &self, tokenizer: &WasmWordPieceTokenizer, text: &str, max_len: usize,
    ) -> Result<Box<[f32]>, JsError> {
        self.inner.embed_text(&tokenizer.inner, text, max_len)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

// ─── GPT-2 model ─────────────────────────────────────────────────────────────

/// GPT-2 decoder-only language model.
#[wasm_bindgen]
pub struct WasmGpt2Model {
    inner: Gpt2Model,
}

#[wasm_bindgen]
impl WasmGpt2Model {
    /// Loads a GPT-2 model from safetensors bytes + `config.json` text.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], config_json: &str) -> Result<WasmGpt2Model, JsError> {
        let config = Gpt2Config::from_config_json(config_json)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let file = ModelFile::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        let inner = Gpt2Model::from_safetensors(&file, config)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns logits `[seq_len × vocab_size]` for all input positions.
    pub fn logits(&self, input_ids: &[u32]) -> Result<Box<[f32]>, JsError> {
        let t = self.inner.logits(input_ids);
        Ok(t.data().to_vec().into_boxed_slice())
    }

    /// Greedy generation. Returns the complete sequence (prompt + new tokens).
    pub fn generate(
        &self,
        tokenizer: &WasmBpeTokenizer,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String, JsError> {
        let enc = tokenizer.inner.encode(prompt, self.inner.config.max_position_embeddings)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let all_ids = self.inner.generate_greedy(&enc.input_ids, max_new_tokens);
        let new_ids = &all_ids[enc.input_ids.len()..];
        Ok(tokenizer.inner.decode(new_ids))
    }
}

// ─── T5 model ────────────────────────────────────────────────────────────────

/// T5 encoder-decoder model.
#[wasm_bindgen]
pub struct WasmT5Model {
    inner: T5Model,
}

#[wasm_bindgen]
impl WasmT5Model {
    /// Loads a T5 model from safetensors bytes + `config.json` text.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], config_json: &str) -> Result<WasmT5Model, JsError> {
        let config = T5Config::from_config_json(config_json)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let file = ModelFile::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        let inner = T5Model::from_safetensors(&file, config)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs the encoder only. Returns hidden states `[seq_len × d_model]`.
    pub fn encode(
        &self,
        tokenizer: &WasmBpeTokenizer,
        text: &str,
        max_len: usize,
    ) -> Result<Box<[f32]>, JsError> {
        let enc = tokenizer.inner.encode(text, max_len)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let out = self.inner.encode(&enc.input_ids);
        Ok(out.data().to_vec().into_boxed_slice())
    }

    /// Seq2seq greedy generation. Returns the generated string.
    pub fn generate(
        &self,
        tokenizer: &WasmBpeTokenizer,
        input_text: &str,
        max_new_tokens: usize,
    ) -> Result<String, JsError> {
        let enc = tokenizer.inner.encode(input_text, 512)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let out_ids = self.inner.generate_greedy(&enc.input_ids, max_new_tokens);
        Ok(tokenizer.inner.decode(&out_ids))
    }
}

// ─── Unified Pipeline ─────────────────────────────────────────────────────────

/// Unified pipeline — one constructor for any supported model type.
///
/// The `model_type` field in `config_json` determines routing:
/// `bert` / `roberta` / `distilbert` → `embed` / `embedBatch`;
/// `gpt2` → `generate`;
/// `t5` → `generate` / `encodeT5`.
#[wasm_bindgen]
pub struct WasmPipeline {
    inner: Pipeline,
}

#[wasm_bindgen]
impl WasmPipeline {
    /// Creates a pipeline from raw bytes.
    ///
    /// - `model_bytes`: `model.safetensors`
    /// - `tokenizer_bytes`: `vocab.txt` for BERT; `vocab.json` for GPT-2/T5
    /// - `config_json`: text of `config.json`
    /// - `merges_bytes`: `merges.txt` for GPT-2/T5; pass `null` for BERT
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        config_json: &str,
        merges_bytes: Option<Vec<u8>>,
    ) -> Result<WasmPipeline, JsError> {
        let inner = Pipeline::from_bytes(
            model_bytes,
            tokenizer_bytes,
            config_json,
            merges_bytes.as_deref(),
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the `model_type` string detected from `config_json`.
    #[wasm_bindgen(js_name = detectedModelType)]
    pub fn detected_model_type(config_json: &str) -> String {
        Pipeline::detected_model_type(config_json)
    }

    // ── BERT-family ──

    /// Tokenises `text` and returns one mean-pooled embedding `[hidden_size]`.
    pub fn embed(&self, text: &str, max_len: usize) -> Result<Box<[f32]>, JsError> {
        self.inner.embed(text, max_len)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Embeds an array of texts. Returns a flat `Float32Array` of
    /// `texts.length × hidden_size` elements (row-major).
    #[wasm_bindgen(js_name = embedBatch)]
    pub fn embed_batch(&self, texts: Vec<String>, max_len: usize) -> Result<Box<[f32]>, JsError> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = self.inner.embed_batch(&refs, max_len)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let flat: Vec<f32> = embeddings.into_iter().flat_map(|t| t.data().to_vec()).collect();
        Ok(flat.into_boxed_slice())
    }

    // ── GPT-2 / T5 ──

    /// Greedy text generation. For GPT-2: continues the prompt.
    /// For T5: prompt is the encoder input (include task prefix, e.g. `"translate English to French: Hello"`).
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String, JsError> {
        self.inner.generate(prompt, max_new_tokens)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// T5-only: runs the encoder and returns hidden states `[seq_len × d_model]`.
    #[wasm_bindgen(js_name = encodeT5)]
    pub fn encode_t5(&self, text: &str, max_len: usize) -> Result<Box<[f32]>, JsError> {
        self.inner.encode_t5(text, max_len)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
