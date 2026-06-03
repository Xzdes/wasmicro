//! Multi-head self-attention and pooling helpers.
//!
//! Implementation choice: heads are processed in a loop on the Rust side
//! rather than via batched matmul. This keeps the code small and obvious; the
//! per-head work is the standard `softmax(Q K^T / sqrt(d)) V` pipeline.
//! When SIMD / batched matmul lands, the optimized path will replace this
//! reference without changing the public signature.

use crate::ops::elementwise::scale;
use crate::ops::linear::linear;
use crate::ops::matmul::{matmul, matmul_t_b};
use crate::ops::softmax::softmax_last_dim;
use crate::tensor::Tensor;

/// Extracts a single head's columns from a `[seq_len, hidden]` tensor.
/// Returns a `[seq_len, head_dim]` tensor.
fn extract_head(x: &Tensor, head_idx: usize, num_heads: usize) -> Tensor {
    let shape = x.shape().as_slice();
    assert_eq!(shape.len(), 2, "extract_head: x must be 2D");
    let seq_len = shape[0];
    let hidden = shape[1];
    assert!(
        hidden % num_heads == 0,
        "extract_head: hidden ({}) must be divisible by num_heads ({})",
        hidden,
        num_heads
    );
    let head_dim = hidden / num_heads;
    let src = x.data();
    let mut out = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let src_off = t * hidden + head_idx * head_dim;
        let dst_off = t * head_dim;
        out[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
    Tensor::from_vec(out, &[seq_len, head_dim])
}

/// Writes a per-head result `[seq_len, head_dim]` back into the correct
/// columns of a `[seq_len, hidden]` flat buffer.
fn write_head_back(dst: &mut [f32], head_result: &Tensor, head_idx: usize, num_heads: usize) {
    let shape = head_result.shape().as_slice();
    assert_eq!(shape.len(), 2, "write_head_back: head must be 2D");
    let seq_len = shape[0];
    let head_dim = shape[1];
    let hidden = num_heads * head_dim;
    let src = head_result.data();
    for t in 0..seq_len {
        let src_off = t * head_dim;
        let dst_off = t * hidden + head_idx * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
}

/// Multi-head self-attention with output projection.
///
/// All linear projections follow the PyTorch convention: weights are stored
/// as `[hidden, hidden]` with rows-as-outputs. The `linear` helper handles
/// the implicit transpose.
///
/// - `x`: input `[seq_len, hidden]`
/// - `wq`, `wk`, `wv`, `wo`: query / key / value / output projections
/// - `bq`, `bk`, `bv`, `bo`: optional biases (BERT uses all four)
/// - `num_heads`: must divide `hidden` evenly
///
/// Returns a `[seq_len, hidden]` tensor.
///
/// The flat parameter list mirrors the natural shape of self-attention
/// weights (Q/K/V/O × weight/bias). Wrapping them in a struct would force
/// every caller to construct one purely for the call — see `BertModel` for
/// an example of how higher-level structs feed into this op.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    x: &Tensor,
    wq: &Tensor,
    bq: Option<&Tensor>,
    wk: &Tensor,
    bk: Option<&Tensor>,
    wv: &Tensor,
    bv: Option<&Tensor>,
    wo: &Tensor,
    bo: Option<&Tensor>,
    num_heads: usize,
) -> Tensor {
    let shape = x.shape().as_slice();
    assert_eq!(shape.len(), 2, "multi_head_attention: x must be 2D");
    let hidden = shape[1];
    assert!(
        hidden % num_heads == 0,
        "multi_head_attention: hidden ({}) must be divisible by num_heads ({})",
        hidden,
        num_heads
    );

    let q = linear(x, wq, bq);
    let k = linear(x, wk, bk);
    let v = linear(x, wv, bv);

    let concat = multi_head_attention_from_qkv(&q, &k, &v, num_heads);
    linear(&concat, wo, bo)
}

/// Computes scaled dot-product multi-head attention from precomputed Q/K/V.
///
/// - `q`, `k`, `v`: shape `[seq_len, hidden]`
/// - `num_heads`: must divide `hidden` evenly
/// - returned tensor shape: `[seq_len, hidden]`
pub fn multi_head_attention_from_qkv(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
) -> Tensor {
    let q_shape = q.shape().as_slice();
    let k_shape = k.shape().as_slice();
    let v_shape = v.shape().as_slice();
    assert_eq!(
        q_shape.len(),
        2,
        "multi_head_attention_from_qkv: q must be 2D"
    );
    assert_eq!(
        k_shape, q_shape,
        "multi_head_attention_from_qkv: k shape mismatch"
    );
    assert_eq!(
        v_shape, q_shape,
        "multi_head_attention_from_qkv: v shape mismatch"
    );

    let seq_len = q_shape[0];
    let hidden = q_shape[1];
    assert!(
        hidden % num_heads == 0,
        "multi_head_attention_from_qkv: hidden ({}) must be divisible by num_heads ({})",
        hidden,
        num_heads
    );
    let head_dim = hidden / num_heads;
    let scale_factor = 1.0 / (head_dim as f32).sqrt();
    let mut concat = vec![0.0f32; seq_len * hidden];

    for h in 0..num_heads {
        let q_h = extract_head(q, h, num_heads);
        let k_h = extract_head(k, h, num_heads);
        let v_h = extract_head(v, h, num_heads);

        // scores = Q_h @ K_h^T / sqrt(head_dim)
        let scores = matmul_t_b(&q_h, &k_h);
        let scores = scale(&scores, scale_factor);
        let attn = softmax_last_dim(&scores);
        let head_out = matmul(&attn, &v_h);
        write_head_back(&mut concat, &head_out, h, num_heads);
    }

    Tensor::from_vec(concat, &[seq_len, hidden])
}

/// Mean-pools a `[seq_len, hidden]` tensor along the sequence dimension.
///
/// If `attention_mask` is provided, masked positions (mask value `0`) are
/// excluded from both the sum and the divisor. This matches the
/// sentence-transformers mean-pooling convention.
///
/// Returns a `[hidden]` tensor.
pub fn mean_pool(x: &Tensor, attention_mask: Option<&[u32]>) -> Tensor {
    let shape = x.shape().as_slice();
    assert_eq!(shape.len(), 2, "mean_pool: x must be 2D");
    let seq_len = shape[0];
    let hidden = shape[1];
    if let Some(m) = attention_mask {
        assert_eq!(
            m.len(),
            seq_len,
            "mean_pool: mask length must equal sequence length"
        );
    }

    let data = x.data();
    let mut out = vec![0.0f32; hidden];
    let mut count = 0.0f32;
    for t in 0..seq_len {
        let valid = match attention_mask {
            Some(m) => m[t] != 0,
            None => true,
        };
        if valid {
            let row = &data[t * hidden..(t + 1) * hidden];
            for (o, &v) in out.iter_mut().zip(row) {
                *o += v;
            }
            count += 1.0;
        }
    }
    if count > 0.0 {
        let inv = 1.0 / count;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
    Tensor::from_vec(out, &[hidden])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_then_write_back_is_identity() {
        // hidden = 4, num_heads = 2 -> head_dim = 2
        // Build a [3, 4] tensor with distinct values per cell.
        let x = Tensor::from_vec((0..12).map(|v| v as f32).collect(), &[3, 4]);
        let mut reconstructed = vec![0.0f32; 12];
        for h in 0..2 {
            let head = extract_head(&x, h, 2);
            write_head_back(&mut reconstructed, &head, h, 2);
        }
        assert_eq!(reconstructed, x.data());
    }

    #[test]
    fn attention_output_shape_matches_input() {
        // hidden = 8, num_heads = 2, seq_len = 3
        let seq_len = 3;
        let hidden = 8;
        let x = Tensor::from_vec(vec![0.1f32; seq_len * hidden], &[seq_len, hidden]);
        // Identity-ish projections.
        let identity = identity_matrix(hidden);
        let zero_bias = Tensor::from_vec(vec![0.0; hidden], &[hidden]);
        let y = multi_head_attention(
            &x,
            &identity,
            Some(&zero_bias),
            &identity,
            Some(&zero_bias),
            &identity,
            Some(&zero_bias),
            &identity,
            Some(&zero_bias),
            2,
        );
        assert_eq!(y.shape().as_slice(), &[seq_len, hidden]);
    }

    #[test]
    fn mean_pool_basic() {
        let x = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]);
        let y = mean_pool(&x, None);
        // Mean per column: (1+3+5)/3 = 3, (2+4+6)/3 = 4
        assert_eq!(y.shape().as_slice(), &[2]);
        assert_eq!(y.data(), &[3.0, 4.0]);
    }

    #[test]
    fn mean_pool_respects_mask() {
        let x = Tensor::from_vec(vec![1., 2., 99., 99., 5., 6.], &[3, 2]);
        let mask = [1u32, 0, 1];
        let y = mean_pool(&x, Some(&mask));
        // Only rows 0 and 2 contribute: (1+5)/2 = 3, (2+6)/2 = 4
        assert_eq!(y.data(), &[3.0, 4.0]);
    }

    fn identity_matrix(n: usize) -> Tensor {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, &[n, n])
    }
}
