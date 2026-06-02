//! Elementwise activation functions used by transformer architectures.

use crate::tensor::Tensor;
use core::f32::consts::PI;

/// ReLU: `max(0, x)`.
pub fn relu(x: &Tensor) -> Tensor {
    let mut out = vec![0.0f32; x.numel()];
    for (o, &v) in out.iter_mut().zip(x.data()) {
        *o = v.max(0.0);
    }
    Tensor::from_vec(out, x.shape().as_slice())
}

/// SiLU (a.k.a. Swish): `x * sigmoid(x)`.
///
/// Used by LLaMA, Mistral, and many modern decoder-only LLMs.
pub fn silu(x: &Tensor) -> Tensor {
    let mut out = vec![0.0f32; x.numel()];
    for (o, &v) in out.iter_mut().zip(x.data()) {
        *o = v / (1.0 + (-v).exp());
    }
    Tensor::from_vec(out, x.shape().as_slice())
}

/// GELU using the tanh approximation.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// Matches HuggingFace's `gelu_new` / `gelu_pytorch_tanh`. Used by GPT-2 and
/// many BERT variants. Use [`gelu_erf`] for the exact erf-based form when the
/// model's config calls for `"gelu"` rather than `"gelu_new"`.
pub fn gelu_tanh(x: &Tensor) -> Tensor {
    let c = (2.0 / PI).sqrt();
    let mut out = vec![0.0f32; x.numel()];
    for (o, &v) in out.iter_mut().zip(x.data()) {
        let inner = c * (v + 0.044715 * v * v * v);
        *o = 0.5 * v * (1.0 + inner.tanh());
    }
    Tensor::from_vec(out, x.shape().as_slice())
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// Uses a polynomial approximation of `erf` (Abramowitz & Stegun 7.1.26) with
/// a maximum error of ~1.5e-7, which is well below `f32` precision for the
/// usual inference range. This matches HuggingFace's default `"gelu"` for
/// most BERT models.
pub fn gelu_erf(x: &Tensor) -> Tensor {
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    let mut out = vec![0.0f32; x.numel()];
    for (o, &v) in out.iter_mut().zip(x.data()) {
        *o = 0.5 * v * (1.0 + erf_approx(v * inv_sqrt2));
    }
    Tensor::from_vec(out, x.shape().as_slice())
}

/// Polynomial approximation of `erf`. Maximum absolute error ~1.5e-7.
fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26. Coefficients are truncated to the f32
    // mantissa limit; the original 9-digit constants round to the same f32
    // values, so numerical results are bit-identical.
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn relu_clamps_negatives() {
        let x = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
        let y = relu(&x);
        assert_eq!(y.data(), &[0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn silu_at_zero_is_zero() {
        let x = Tensor::from_vec(vec![0.0], &[1]);
        let y = silu(&x);
        assert!(approx_eq(y.data()[0], 0.0, 1e-7));
    }

    #[test]
    fn gelu_tanh_at_zero_is_zero() {
        let x = Tensor::from_vec(vec![0.0], &[1]);
        let y = gelu_tanh(&x);
        assert!(approx_eq(y.data()[0], 0.0, 1e-7));
    }

    #[test]
    fn gelu_erf_matches_known_values() {
        // GELU(1.0) ~ 0.84134
        // GELU(-1.0) ~ -0.15866
        let x = Tensor::from_vec(vec![1.0, -1.0], &[2]);
        let y = gelu_erf(&x);
        assert!(approx_eq(y.data()[0], 0.8413_447, 1e-4));
        assert!(approx_eq(y.data()[1], -0.1586_553, 1e-4));
    }

    #[test]
    fn erf_known_values() {
        // erf(0) = 0
        assert!(approx_eq(erf_approx(0.0), 0.0, 1e-6));
        // erf(1) ~ 0.8427
        assert!(approx_eq(erf_approx(1.0), 0.8427_008, 1e-5));
        // erf(-1) ~ -0.8427
        assert!(approx_eq(erf_approx(-1.0), -0.8427_008, 1e-5));
    }
}
