//! WASM bindings.
//!
//! Compiled only when the `wasm` feature is enabled. Exposes a small,
//! stable JavaScript-facing API. The boundary is intentionally narrow —
//! everything inside the library is plain Rust.

use wasm_bindgen::prelude::*;

use crate::loader::ModelFile;
use crate::models::bert::{BertConfig, BertModel};
use crate::ops::matmul::matmul;
use crate::tensor::Tensor;
use crate::tokenizer::{EncodedInput, WordPieceOptions, WordPieceTokenizer};

/// Initialize the WASM panic hook so Rust panics surface in the browser console.
/// Call once from JavaScript on module load. No-op if the `wasm-debug` feature
/// is disabled.
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "wasm-debug")]
    console_error_panic_hook::set_once();
}

/// Library version string, exposed to JavaScript.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Runs an `n x n` matrix multiplication and returns the first output cell.
///
/// Used by the demo page to time WASM performance from the JS side. Returning
/// the first cell prevents the optimizer from dropping the computation.
#[wasm_bindgen]
pub fn matmul_bench(n: usize) -> Result<f32, JsError> {
    if n == 0 || n > 2048 {
        return Err(JsError::new("n must be in the range 1..=2048"));
    }
    let len = n
        .checked_mul(n)
        .ok_or_else(|| JsError::new("matrix size overflow"))?;
    let a = Tensor::from_vec(vec![1.0; len], &[n, n]);
    let b = Tensor::from_vec(vec![1.0; len], &[n, n]);
    let c = matmul(&a, &b);
    Ok(c.data()[0])
}

/// WordPiece tokenizer exposed to JavaScript.
///
/// Construct from `vocab.txt` bytes fetched by the host. Keeping the vocabulary
/// outside the WASM binary preserves the small runtime size.
#[wasm_bindgen]
pub struct WasmWordPieceTokenizer {
    inner: WordPieceTokenizer,
}

#[wasm_bindgen]
impl WasmWordPieceTokenizer {
    /// Parses UTF-8 `vocab.txt` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &[u8], lowercase: bool) -> Result<WasmWordPieceTokenizer, JsError> {
        let options = WordPieceOptions {
            lowercase,
            ..WordPieceOptions::default()
        };
        let inner = WordPieceTokenizer::from_vocab_bytes_with_options(vocab, options)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encodes text without padding.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<WasmEncodedInput, JsError> {
        self.inner
            .encode(text, max_len)
            .map(WasmEncodedInput::new)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Encodes text and pads arrays to exactly `max_len`.
    pub fn encode_padded(&self, text: &str, max_len: usize) -> Result<WasmEncodedInput, JsError> {
        self.inner
            .encode_padded(text, max_len)
            .map(WasmEncodedInput::new)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Returns the id for a token string, or `undefined` if it is not in vocab.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_id(token)
    }
}

/// Encoded BERT input exposed to JavaScript.
#[wasm_bindgen]
pub struct WasmEncodedInput {
    inner: EncodedInput,
}

impl WasmEncodedInput {
    fn new(inner: EncodedInput) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl WasmEncodedInput {
    /// Token ids, including `[CLS]` and `[SEP]`.
    pub fn input_ids(&self) -> Box<[u32]> {
        self.inner.input_ids.clone().into_boxed_slice()
    }

    /// BERT token type ids.
    pub fn token_type_ids(&self) -> Box<[u32]> {
        self.inner.token_type_ids.clone().into_boxed_slice()
    }

    /// Attention mask: `1` for real tokens, `0` for padding.
    pub fn attention_mask(&self) -> Box<[u32]> {
        self.inner.attention_mask.clone().into_boxed_slice()
    }
}

/// BERT encoder exposed to JavaScript.
///
/// Construct with the raw bytes of a `model.safetensors` file plus the model
/// configuration (each field is passed individually to keep the binding
/// dependency-free — no JSON parsing in JS-facing glue code).
#[wasm_bindgen]
pub struct WasmBertModel {
    inner: BertModel,
}

#[wasm_bindgen]
impl WasmBertModel {
    /// Parses a BERT model from safetensors bytes.
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
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps: 1e-12,
        };
        let file = ModelFile::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?;
        let inner = BertModel::from_safetensors(&file, config, prefix)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs the encoder and returns flat `[seq_len, hidden_size]` hidden states.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Box<[f32]>, JsError> {
        self.inner
            .try_forward(input_ids, None)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Runs the encoder + mean-pool, returning a single `[hidden_size]` vector.
    pub fn embed(&self, input_ids: &[u32]) -> Result<Box<[f32]>, JsError> {
        self.inner
            .try_embed_sentence(input_ids, None, None)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Runs the encoder + mean-pool with an explicit attention mask.
    pub fn embed_with_mask(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Box<[f32]>, JsError> {
        self.inner
            .try_embed_sentence(input_ids, None, Some(attention_mask))
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Runs the encoder + mean-pool from a tokenizer output.
    pub fn embed_encoded(&self, encoded: &WasmEncodedInput) -> Result<Box<[f32]>, JsError> {
        self.inner
            .try_embed_sentence(
                &encoded.inner.input_ids,
                Some(&encoded.inner.token_type_ids),
                Some(&encoded.inner.attention_mask),
            )
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Tokenizes text, runs the encoder, and returns one pooled embedding.
    pub fn embed_text(
        &self,
        tokenizer: &WasmWordPieceTokenizer,
        text: &str,
        max_len: usize,
    ) -> Result<Box<[f32]>, JsError> {
        self.inner
            .embed_text(&tokenizer.inner, text, max_len)
            .map(|t| t.data().to_vec().into_boxed_slice())
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
