//! # wasmicro
//!
//! Tiny transformer inference for the web. One file. No build step.
//!
//! ## Design rules
//!
//! 1. Tiny WASM bundle (target: < 250 KB after `wasm-opt -Oz`).
//! 2. Fast cold start (target: < 500 ms model load + first inference).
//! 3. Forward inference only — no autograd, no optimizers, no training.
//! 4. Owned tensors, no `Rc<RefCell>` indirection.
//! 5. Minimal dependencies. The full default build pulls in only `bytemuck`.
//! 6. Same code runs natively and in WASM. The library never opens files —
//!    callers pass bytes in via [`ModelFile::parse`].
//!
//! ## Quick start
//!
//! ```no_run
//! use std::fs;
//! use wasmicro::{
//!     models::bert::{BertConfig, BertModel},
//!     ModelFile, WordPieceTokenizer,
//! };
//!
//! let model_bytes = fs::read("model.safetensors").unwrap();
//! let vocab_bytes = fs::read("vocab.txt").unwrap();
//! let file = ModelFile::parse(&model_bytes).unwrap();
//! let tokenizer = WordPieceTokenizer::from_vocab_bytes(&vocab_bytes).unwrap();
//!
//! let config = BertConfig::mini_lm_l6_v2();
//! let model = BertModel::from_safetensors(&file, config, "").unwrap();
//!
//! let embedding = model.embed_text(&tokenizer, "hello world", 128).unwrap();
//! println!("embedding dim: {:?}", embedding.shape().as_slice());
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod loader;
pub mod models;
pub mod ops;
pub mod pipeline;
pub mod quant;
pub mod tensor;
pub mod tokenizer;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for the most-used types.
pub use error::{Error, Result};
pub use loader::{Dtype, ModelFile, TensorView};
pub use pipeline::Pipeline;
pub use quant::{QuantizedTensorI8, QuantizedTensorQ4, QuantizedTensorU8};
pub use tensor::{Shape, Tensor};
pub use tokenizer::bpe::BpeTokenizer;
pub use tokenizer::{EncodedInput, WordPieceOptions, WordPieceTokenizer};
