//! Root Mean Square Layer Normalization (RMSNorm).
//!
//! Used by T5, LLaMA, and Mistral as a lighter alternative to LayerNorm.
//! Unlike LayerNorm, RMSNorm skips mean subtraction — it normalizes by the
//! RMS of the input alone, which is cheaper and often equally effective.

use crate::tensor::Tensor;

/// Applies RMSNorm over the last dimension of `x`.
///
/// Formula: `output = x / rms(x) * weight`  where
/// `rms(x) = sqrt(mean(x²) + eps)`.
///
/// - `x`: any shape; normalization is over the last dimension.
/// - `weight` (gamma): scale, shape `[last_dim]`.
/// - `eps`: stability epsilon (T5 uses `1e-6`).
///
/// Returns a tensor of the same shape as `x`.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let shape = x.shape().as_slice();
    let last_dim = shape[shape.len() - 1];
    debug_assert_eq!(
        weight.shape().as_slice(),
        &[last_dim],
        "rms_norm: weight must be [last_dim]"
    );
    let batch = x.numel() / last_dim;
    let data = x.data();
    let w = weight.data();
    let mut out = vec![0.0f32; x.numel()];

    for b in 0..batch {
        let off = b * last_dim;
        let row = &data[off..off + last_dim];
        let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
        let rms_inv = 1.0 / (sum_sq / last_dim as f32 + eps).sqrt();
        for i in 0..last_dim {
            out[off + i] = row[i] * rms_inv * w[i];
        }
    }

    Tensor::from_vec(out, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight() {
        // x = [3, 4] → rms = sqrt((9+16)/2) = sqrt(12.5)
        let x = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 2]);
        let w = Tensor::from_vec(vec![1.0f32, 1.0], &[2]);
        let y = rms_norm(&x, &w, 0.0);
        let rms = (12.5f32).sqrt();
        assert!((y.data()[0] - 3.0 / rms).abs() < 1e-5);
        assert!((y.data()[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_scaled_weight() {
        let x = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 2]);
        let w = Tensor::from_vec(vec![2.0f32, 0.5], &[2]);
        let y = rms_norm(&x, &w, 0.0);
        let rms = (12.5f32).sqrt();
        assert!((y.data()[0] - 3.0 / rms * 2.0).abs() < 1e-5);
        assert!((y.data()[1] - 4.0 / rms * 0.5).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_batch() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], &[3]);
        let y = rms_norm(&x, &w, 1e-6);
        assert_eq!(y.shape().as_slice(), &[2, 3]);
        for &v in y.data() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn rms_norm_eps_prevents_div_by_zero() {
        let x = Tensor::from_vec(vec![0.0f32, 0.0], &[1, 2]);
        let w = Tensor::from_vec(vec![1.0f32, 1.0], &[2]);
        let y = rms_norm(&x, &w, 1e-6);
        for &v in y.data() {
            assert!(v.is_finite());
        }
    }
}
