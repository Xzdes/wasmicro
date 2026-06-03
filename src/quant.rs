//! Small weight-only quantized tensor types.
//!
//! These types do not replace [`Tensor`](crate::Tensor). They are narrow
//! storage formats for transformer weights, paired with explicit quantized
//! linear ops. Activations stay `f32`, which keeps the runtime simple and
//! avoids a general dtype system.

use crate::error::Result;
use crate::loader::ModelFile;
use crate::tensor::{Shape, Tensor};

/// Signed 8-bit quantized matrix with per-tensor or per-row scales.
#[derive(Clone, Debug)]
pub struct QuantizedTensorI8 {
    data: Vec<i8>,
    shape: Shape,
    scales: Vec<f32>,
}

impl QuantizedTensorI8 {
    /// Creates a signed 8-bit quantized tensor from owned data.
    ///
    /// `shape` must be 2D. `scales` must contain either one global scale or
    /// one scale per output row.
    pub fn from_vec(data: Vec<i8>, shape: &[usize], scales: Vec<f32>) -> Self {
        let shape = Shape::new(shape);
        validate_quantized_matrix(data.len(), &shape, scales.len(), "QuantizedTensorI8");
        Self {
            data,
            shape,
            scales,
        }
    }

    /// Quantizes an `f32` matrix row-by-row into signed 8-bit weights.
    pub fn quantize_per_row(weight: &Tensor) -> Self {
        let shape = weight.shape();
        validate_2d_shape(shape, "QuantizedTensorI8::quantize_per_row");
        let dims = shape.as_slice();
        let rows = dims[0];
        let cols = dims[1];
        let mut data = Vec::with_capacity(weight.numel());
        let mut scales = Vec::with_capacity(rows);

        for row in 0..rows {
            let row_data = &weight.data()[row * cols..(row + 1) * cols];
            let max_abs = row_data
                .iter()
                .fold(0.0f32, |acc, v| if v.abs() > acc { v.abs() } else { acc });
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);
            for v in row_data {
                data.push((v / scale).round().clamp(-127.0, 127.0) as i8);
            }
        }

        Self {
            data,
            shape: *shape,
            scales,
        }
    }

    /// Loads signed 8-bit values plus `f32` scales from a safetensors file.
    pub fn from_safetensors(
        file: &ModelFile<'_>,
        values_name: &str,
        scales_name: &str,
    ) -> Result<Self> {
        let values = file.get(values_name)?;
        let scales = file.get(scales_name)?.to_tensor()?;
        Ok(Self::from_vec(
            values.as_i8()?.to_vec(),
            values.shape,
            scales.data().to_vec(),
        ))
    }

    /// Raw signed 8-bit quantized values.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Matrix shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Per-tensor or per-row scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Dequantizes into an owned `f32` tensor.
    pub fn dequantize(&self) -> Tensor {
        let dims = self.shape.as_slice();
        let cols = dims[1];
        let mut out = Vec::with_capacity(self.data.len());
        for (idx, q) in self.data.iter().enumerate() {
            let row = idx / cols;
            out.push(*q as f32 * self.scale_for_row(row));
        }
        Tensor::from_vec(out, dims)
    }

    pub(crate) fn scale_for_row(&self, row: usize) -> f32 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[row]
        }
    }
}

/// Unsigned affine 8-bit quantized matrix with per-tensor or per-row params.
#[derive(Clone, Debug)]
pub struct QuantizedTensorU8 {
    data: Vec<u8>,
    shape: Shape,
    scales: Vec<f32>,
    zero_points: Vec<u8>,
}

impl QuantizedTensorU8 {
    /// Creates an unsigned affine 8-bit quantized tensor from owned data.
    ///
    /// `shape` must be 2D. `scales` and `zero_points` must contain either one
    /// global parameter or one parameter per output row.
    pub fn from_vec(
        data: Vec<u8>,
        shape: &[usize],
        scales: Vec<f32>,
        zero_points: Vec<u8>,
    ) -> Self {
        let shape = Shape::new(shape);
        validate_quantized_matrix(data.len(), &shape, scales.len(), "QuantizedTensorU8");
        validate_param_len(
            zero_points.len(),
            shape.as_slice()[0],
            "QuantizedTensorU8 zero_points",
        );
        Self {
            data,
            shape,
            scales,
            zero_points,
        }
    }

    /// Quantizes an `f32` matrix row-by-row into unsigned affine 8-bit weights.
    pub fn quantize_per_row(weight: &Tensor) -> Self {
        let shape = weight.shape();
        validate_2d_shape(shape, "QuantizedTensorU8::quantize_per_row");
        let dims = shape.as_slice();
        let rows = dims[0];
        let cols = dims[1];
        let mut data = Vec::with_capacity(weight.numel());
        let mut scales = Vec::with_capacity(rows);
        let mut zero_points = Vec::with_capacity(rows);

        for row in 0..rows {
            let row_data = &weight.data()[row * cols..(row + 1) * cols];
            let (min, max) = row_data
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, v| {
                    (acc.0.min(*v), acc.1.max(*v))
                });
            let scale = if max == min { 1.0 } else { (max - min) / 255.0 };
            let zero_point = (-min / scale).round().clamp(0.0, 255.0) as u8;
            scales.push(scale);
            zero_points.push(zero_point);
            for v in row_data {
                let q = (*v / scale).round() + zero_point as f32;
                data.push(q.clamp(0.0, 255.0) as u8);
            }
        }

        Self {
            data,
            shape: *shape,
            scales,
            zero_points,
        }
    }

    /// Loads unsigned affine 8-bit values, `f32` scales, and zero points from
    /// a safetensors file.
    pub fn from_safetensors(
        file: &ModelFile<'_>,
        values_name: &str,
        scales_name: &str,
        zero_points_name: &str,
    ) -> Result<Self> {
        let values = file.get(values_name)?;
        let scales = file.get(scales_name)?.to_tensor()?;
        let zero_points = file.get(zero_points_name)?;
        Ok(Self::from_vec(
            values.as_u8()?.to_vec(),
            values.shape,
            scales.data().to_vec(),
            zero_points.as_u8()?.to_vec(),
        ))
    }

    /// Raw unsigned 8-bit quantized values.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Matrix shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Per-tensor or per-row scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Per-tensor or per-row zero points.
    pub fn zero_points(&self) -> &[u8] {
        &self.zero_points
    }

    /// Dequantizes into an owned `f32` tensor.
    pub fn dequantize(&self) -> Tensor {
        let dims = self.shape.as_slice();
        let cols = dims[1];
        let mut out = Vec::with_capacity(self.data.len());
        for (idx, q) in self.data.iter().enumerate() {
            let row = idx / cols;
            out.push(
                (*q as i32 - self.zero_point_for_row(row) as i32) as f32 * self.scale_for_row(row),
            );
        }
        Tensor::from_vec(out, dims)
    }

    pub(crate) fn scale_for_row(&self, row: usize) -> f32 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[row]
        }
    }

    pub(crate) fn zero_point_for_row(&self, row: usize) -> u8 {
        if self.zero_points.len() == 1 {
            self.zero_points[0]
        } else {
            self.zero_points[row]
        }
    }
}

/// Packed signed 4-bit quantized matrix with per-tensor or per-row scales.
#[derive(Clone, Debug)]
pub struct QuantizedTensorQ4 {
    data: Vec<u8>,
    len: usize,
    shape: Shape,
    scales: Vec<f32>,
}

impl QuantizedTensorQ4 {
    /// Creates a packed signed 4-bit tensor from owned packed bytes.
    ///
    /// `len` is the number of unpacked elements. Low nibble stores the even
    /// element, high nibble stores the odd element.
    pub fn from_packed(data: Vec<u8>, len: usize, shape: &[usize], scales: Vec<f32>) -> Self {
        let shape = Shape::new(shape);
        assert_eq!(
            len,
            shape.numel(),
            "QuantizedTensorQ4: len must match shape"
        );
        assert_eq!(
            data.len(),
            len.div_ceil(2),
            "QuantizedTensorQ4: packed byte length must be ceil(len / 2)"
        );
        validate_quantized_matrix(len, &shape, scales.len(), "QuantizedTensorQ4");
        Self {
            data,
            len,
            shape,
            scales,
        }
    }

    /// Creates a packed signed 4-bit tensor from unpacked values in `[-8, 7]`.
    pub fn from_i4_values(values: &[i8], shape: &[usize], scales: Vec<f32>) -> Self {
        let mut data = vec![0u8; values.len().div_ceil(2)];
        for (idx, value) in values.iter().enumerate() {
            let packed = pack_i4(*value);
            let slot = &mut data[idx / 2];
            if idx % 2 == 0 {
                *slot = (*slot & 0xf0) | packed;
            } else {
                *slot = (*slot & 0x0f) | (packed << 4);
            }
        }
        Self::from_packed(data, values.len(), shape, scales)
    }

    /// Quantizes an `f32` matrix row-by-row into packed signed 4-bit weights.
    pub fn quantize_per_row(weight: &Tensor) -> Self {
        let shape = weight.shape();
        validate_2d_shape(shape, "QuantizedTensorQ4::quantize_per_row");
        let dims = shape.as_slice();
        let rows = dims[0];
        let cols = dims[1];
        let mut values = Vec::with_capacity(weight.numel());
        let mut scales = Vec::with_capacity(rows);

        for row in 0..rows {
            let row_data = &weight.data()[row * cols..(row + 1) * cols];
            let max_abs = row_data
                .iter()
                .fold(0.0f32, |acc, v| if v.abs() > acc { v.abs() } else { acc });
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            scales.push(scale);
            for v in row_data {
                values.push((v / scale).round().clamp(-8.0, 7.0) as i8);
            }
        }

        Self::from_i4_values(&values, dims, scales)
    }

    /// Packed signed 4-bit data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Number of unpacked elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no unpacked elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Matrix shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Per-tensor or per-row scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns one unpacked signed 4-bit value.
    pub fn get(&self, idx: usize) -> i8 {
        assert!(idx < self.len, "QuantizedTensorQ4: index out of bounds");
        let byte = self.data[idx / 2];
        let nibble = if idx.is_multiple_of(2) { byte & 0x0f } else { byte >> 4 };
        unpack_i4(nibble)
    }

    /// Dequantizes into an owned `f32` tensor.
    pub fn dequantize(&self) -> Tensor {
        let dims = self.shape.as_slice();
        let cols = dims[1];
        let mut out = Vec::with_capacity(self.len);
        for idx in 0..self.len {
            let row = idx / cols;
            out.push(self.get(idx) as f32 * self.scale_for_row(row));
        }
        Tensor::from_vec(out, dims)
    }

    pub(crate) fn scale_for_row(&self, row: usize) -> f32 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[row]
        }
    }
}

fn validate_quantized_matrix(data_len: usize, shape: &Shape, param_len: usize, name: &str) {
    validate_2d_shape(shape, name);
    assert_eq!(
        data_len,
        shape.numel(),
        "{}: data length must match shape",
        name
    );
    validate_param_len(param_len, shape.as_slice()[0], name);
}

fn validate_2d_shape(shape: &Shape, name: &str) {
    assert_eq!(
        shape.as_slice().len(),
        2,
        "{}: quantized weights must be 2D",
        name
    );
}

fn validate_param_len(param_len: usize, rows: usize, name: &str) {
    assert!(
        param_len == 1 || param_len == rows,
        "{}: quantization parameters must be per-tensor or per-row",
        name
    );
}

fn pack_i4(value: i8) -> u8 {
    assert!(
        (-8..=7).contains(&value),
        "QuantizedTensorQ4: i4 value out of range"
    );
    (value & 0x0f) as u8
}

fn unpack_i4(nibble: u8) -> i8 {
    let value = (nibble & 0x0f) as i8;
    if value >= 8 {
        value - 16
    } else {
        value
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
    fn i8_dequantizes_per_row() {
        let q = QuantizedTensorI8::from_vec(vec![2, -4, 1, 2], &[2, 2], vec![0.5, 2.0]);
        let deq = q.dequantize();
        assert_close(deq.data(), &[1.0, -2.0, 2.0, 4.0]);
    }

    #[test]
    fn u8_dequantizes_affine_per_row() {
        let q =
            QuantizedTensorU8::from_vec(vec![12, 6, 11, 12], &[2, 2], vec![0.5, 2.0], vec![10, 10]);
        let deq = q.dequantize();
        assert_close(deq.data(), &[1.0, -2.0, 2.0, 4.0]);
    }

    #[test]
    fn q4_packs_and_dequantizes_per_row() {
        let q = QuantizedTensorQ4::from_i4_values(&[2, -4, 1, 2], &[2, 2], vec![0.5, 2.0]);
        assert_eq!(q.data(), &[0xc2, 0x21]);
        assert_eq!(q.get(0), 2);
        assert_eq!(q.get(1), -4);
        let deq = q.dequantize();
        assert_close(deq.data(), &[1.0, -2.0, 2.0, 4.0]);
    }

    #[test]
    fn quantize_i8_round_trips_small_values() {
        let weight = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], &[2, 2]);
        let q = QuantizedTensorI8::quantize_per_row(&weight);
        assert_eq!(q.shape().as_slice(), &[2, 2]);
        assert_eq!(q.scales().len(), 2);
        let deq = q.dequantize();
        assert!((deq.data()[1] - 1.0).abs() < 0.01);
        assert!((deq.data()[2] + 1.0).abs() < 0.02);
    }
}
