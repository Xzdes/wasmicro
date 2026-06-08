//! Pre-built transformer architectures.
//!
//! Each model is a plain struct of weight tensors with a `forward` method.
//! There is no `trait Model` — every model has its own input/output shape
//! and the caller knows which one they're using.

pub mod bert;
pub mod gpt2;
pub mod t5;
