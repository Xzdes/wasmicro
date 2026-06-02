//! Weight-only quantized linear operations.

use crate::quant::{QuantizedTensorI8, QuantizedTensorQ4, QuantizedTensorU8};
use crate::tensor::Tensor;

/// Computes `C = A @ B^T` where `B` is signed 8-bit quantized.
///
/// - `a` shape: `[m, k]`
/// - `b` shape: `[n, k]`
/// - returned tensor shape: `[m, n]`
pub fn matmul_t_b_i8(a: &Tensor, b: &QuantizedTensorI8) -> Tensor {
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();
    validate_matmul_shapes(a_shape, b_shape, "matmul_t_b_i8");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let a_data = a.data();
    let b_data = b.data();
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let a_row = &a_data[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b_data[j * k..(j + 1) * k];
            let scale = b.scale_for_row(j);
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_row[kk] * b_row[kk] as f32 * scale;
            }
            out_row[j] = acc;
        }
    }

    Tensor::from_vec(out, &[m, n])
}

/// Applies `y = x @ W^T + b` where `W` is signed 8-bit quantized.
pub fn linear_i8(x: &Tensor, weight: &QuantizedTensorI8, bias: Option<&Tensor>) -> Tensor {
    let mut y = matmul_t_b_i8(x, weight);
    add_optional_bias(&mut y, bias, "linear_i8");
    y
}

/// Computes `C = A @ B^T` where `B` is unsigned affine 8-bit quantized.
///
/// - `a` shape: `[m, k]`
/// - `b` shape: `[n, k]`
/// - returned tensor shape: `[m, n]`
pub fn matmul_t_b_u8(a: &Tensor, b: &QuantizedTensorU8) -> Tensor {
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();
    validate_matmul_shapes(a_shape, b_shape, "matmul_t_b_u8");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let a_data = a.data();
    let b_data = b.data();
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let a_row = &a_data[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b_data[j * k..(j + 1) * k];
            let scale = b.scale_for_row(j);
            let zero_point = b.zero_point_for_row(j) as i32;
            let mut acc = 0.0f32;
            for kk in 0..k {
                let deq = (b_row[kk] as i32 - zero_point) as f32 * scale;
                acc += a_row[kk] * deq;
            }
            out_row[j] = acc;
        }
    }

    Tensor::from_vec(out, &[m, n])
}

/// Applies `y = x @ W^T + b` where `W` is unsigned affine 8-bit quantized.
pub fn linear_u8(x: &Tensor, weight: &QuantizedTensorU8, bias: Option<&Tensor>) -> Tensor {
    let mut y = matmul_t_b_u8(x, weight);
    add_optional_bias(&mut y, bias, "linear_u8");
    y
}

/// Computes `C = A @ B^T` where `B` is packed signed 4-bit quantized.
///
/// - `a` shape: `[m, k]`
/// - `b` shape: `[n, k]`
/// - returned tensor shape: `[m, n]`
pub fn matmul_t_b_q4(a: &Tensor, b: &QuantizedTensorQ4) -> Tensor {
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();
    validate_matmul_shapes(a_shape, b_shape, "matmul_t_b_q4");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let a_data = a.data();
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let a_row = &a_data[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let scale = b.scale_for_row(j);
            let mut acc = 0.0f32;
            for kk in 0..k {
                let q = b.get(j * k + kk) as f32;
                acc += a_row[kk] * q * scale;
            }
            out_row[j] = acc;
        }
    }

    Tensor::from_vec(out, &[m, n])
}

/// Applies `y = x @ W^T + b` where `W` is packed signed 4-bit quantized.
pub fn linear_q4(x: &Tensor, weight: &QuantizedTensorQ4, bias: Option<&Tensor>) -> Tensor {
    let mut y = matmul_t_b_q4(x, weight);
    add_optional_bias(&mut y, bias, "linear_q4");
    y
}

fn validate_matmul_shapes(a_shape: &[usize], b_shape: &[usize], op: &str) {
    assert_eq!(a_shape.len(), 2, "{}: `a` must be 2D", op);
    assert_eq!(b_shape.len(), 2, "{}: `b` must be 2D", op);
    assert_eq!(
        a_shape[1], b_shape[1],
        "{}: inner dimensions must match: {:?} @ {:?}^T",
        op, a_shape, b_shape
    );
}

fn add_optional_bias(y: &mut Tensor, bias: Option<&Tensor>, op: &str) {
    let Some(bias) = bias else {
        return;
    };

    let y_shape = y.shape().as_slice();
    let bias_shape = bias.shape().as_slice();
    assert_eq!(bias_shape.len(), 1, "{}: bias must be 1D", op);
    assert_eq!(
        bias_shape[0], y_shape[1],
        "{}: bias length must match output columns",
        op
    );

    let n = y_shape[1];
    for row in y.data_mut().chunks_mut(n) {
        for j in 0..n {
            row[j] += bias.data()[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "actual {a}, expected {e}");
        }
    }

    #[test]
    fn linear_i8_matches_known_values() {
        let x = Tensor::from_vec(vec![3.0, 5.0], &[1, 2]);
        let w = QuantizedTensorI8::from_vec(vec![2, -4, 1, 2], &[2, 2], vec![0.5, 2.0]);
        let b = Tensor::from_vec(vec![1.0, -6.0], &[2]);
        let y = linear_i8(&x, &w, Some(&b));
        assert_eq!(y.shape().as_slice(), &[1, 2]);
        assert_close(y.data(), &[-6.0, 20.0]);
    }

    #[test]
    fn linear_u8_matches_known_values() {
        let x = Tensor::from_vec(vec![3.0, 5.0], &[1, 2]);
        let w =
            QuantizedTensorU8::from_vec(vec![12, 6, 11, 12], &[2, 2], vec![0.5, 2.0], vec![10, 10]);
        let b = Tensor::from_vec(vec![1.0, -6.0], &[2]);
        let y = linear_u8(&x, &w, Some(&b));
        assert_eq!(y.shape().as_slice(), &[1, 2]);
        assert_close(y.data(), &[-6.0, 20.0]);
    }

    #[test]
    fn linear_q4_matches_known_values() {
        let x = Tensor::from_vec(vec![3.0, 5.0], &[1, 2]);
        let w = QuantizedTensorQ4::from_i4_values(&[2, -4, 1, 2], &[2, 2], vec![0.5, 2.0]);
        let b = Tensor::from_vec(vec![1.0, -6.0], &[2]);
        let y = linear_q4(&x, &w, Some(&b));
        assert_eq!(y.shape().as_slice(), &[1, 2]);
        assert_close(y.data(), &[-6.0, 20.0]);
    }
}
