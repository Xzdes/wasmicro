//! Layer normalization along the last dimension.
//!
//! Uses Welford's online algorithm to compute mean and variance in a single
//! pass over the input, which halves the memory bandwidth versus the naive
//! two-pass formulation.

use crate::tensor::Tensor;

/// Applies layer normalization along the last dimension.
///
/// - `x`: input, any shape `[.., n]`
/// - `gamma`: scale, shape `[n]` (set to ones to disable)
/// - `beta`: offset, shape `[n]` (set to zeros to disable)
/// - `eps`: numerical stability constant (typical: `1e-5`)
///
/// Returns a tensor of the same shape as `x`.
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    let shape = x.shape().as_slice();
    let n = *shape.last().expect("layer_norm: x must be non-scalar");
    assert_eq!(
        gamma.shape().as_slice(),
        &[n],
        "layer_norm: gamma must have shape [n]"
    );
    assert_eq!(
        beta.shape().as_slice(),
        &[n],
        "layer_norm: beta must have shape [n]"
    );

    let data = x.data();
    let g = gamma.data();
    let b = beta.data();
    let mut out = vec![0.0f32; data.len()];

    for (in_row, out_row) in data.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        // Welford's algorithm for mean and variance.
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        for (i, &v) in in_row.iter().enumerate() {
            let delta = v - mean;
            mean += delta / (i + 1) as f32;
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let var = m2 / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for (i, (o, &v)) in out_row.iter_mut().zip(in_row).enumerate() {
            *o = (v - mean) * inv_std * g[i] + b[i];
        }
    }

    Tensor::from_vec(out, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = Tensor::from_vec(vec![1.0; 4], &[4]);
        let beta = Tensor::from_vec(vec![0.0; 4], &[4]);
        let y = layer_norm(&x, &gamma, &beta, 1e-5);

        let mean: f32 = y.data().iter().sum::<f32>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-5));

        let var: f32 = y.data().iter().map(|&v| v * v).sum::<f32>() / 4.0;
        assert!(approx_eq(var, 1.0, 1e-3));
    }

    #[test]
    fn layer_norm_scale_and_shift() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = Tensor::from_vec(vec![2.0; 4], &[4]);
        let beta = Tensor::from_vec(vec![10.0; 4], &[4]);
        let y = layer_norm(&x, &gamma, &beta, 1e-5);

        // Mean should be 10 (the beta), since gamma * normalized has mean 0.
        let mean: f32 = y.data().iter().sum::<f32>() / 4.0;
        assert!(approx_eq(mean, 10.0, 1e-4));
    }
}
