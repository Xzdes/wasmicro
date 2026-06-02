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
pub fn matmul_bench(n: usize) -> f32 {
    let a = Tensor::from_vec(vec![1.0; n * n], &[n, n]);
    let b = Tensor::from_vec(vec![1.0; n * n], &[n, n]);
    let c = matmul(&a, &b);
    c.data()[0]
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
    pub fn forward(&self, input_ids: &[u32]) -> Box<[f32]> {
        self.inner
            .forward(input_ids, None)
            .data()
            .to_vec()
            .into_boxed_slice()
    }

    /// Runs the encoder + mean-pool, returning a single `[hidden_size]` vector.
    pub fn embed(&self, input_ids: &[u32]) -> Box<[f32]> {
        self.inner
            .embed_sentence(input_ids, None, None)
            .data()
            .to_vec()
            .into_boxed_slice()
    }
}
