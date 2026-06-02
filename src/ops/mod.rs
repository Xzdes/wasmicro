//! Forward-only tensor operations.
//!
//! Every op is a free function. Inputs are borrowed (`&Tensor`), outputs are
//! either returned by value or written into a caller-provided `&mut Tensor`.
//! This gives callers full control over allocations — critical for
//! low-latency inference.
//!
//! Optimized paths (SIMD, blocked matmul, int8) will live alongside the
//! reference implementations and be selected at compile time via features.

pub mod activations;
pub mod attention;
pub mod elementwise;
pub mod embedding;
pub mod layernorm;
pub mod linear;
pub mod matmul;
pub mod softmax;
