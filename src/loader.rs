//! Safetensors model file loader.
//!
//! The host (the application embedding `wasmicro`) is responsible for
//! providing model bytes. In native code this means `std::fs::read`; in the
//! browser it means reading an `ArrayBuffer` via `fetch`. The library itself
//! never opens files.
//!
//! ## Format
//!
//! Safetensors files are laid out as:
//!
//! ```text
//! [ 8 bytes ][ header_len bytes ][ tensor data ... ]
//!   |             |                 |
//!   |             |                 +-- raw little-endian tensor bytes
//!   |             +-- UTF-8 JSON describing each tensor
//!   +-- little-endian u64 header length
//! ```
//!
//! The JSON header maps tensor names to `{ "dtype", "shape", "data_offsets" }`
//! objects, plus an optional `"__metadata__"` entry.
//!
//! ## Why a custom parser?
//!
//! The `safetensors` crate pulls in `serde` and `serde_json`, which together
//! add ~150 KB to WASM bundles. This module implements a focused parser for
//! the safetensors header shape — about 200 lines of code, no dependencies.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Supported tensor element types. Loading non-`F32` tensors as `f32` works
/// only via explicit conversion (not implemented in this initial version).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 16-bit bfloat.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Boolean (1 byte per element).
    Bool,
}

impl Dtype {
    /// Returns the size in bytes of a single element of this dtype.
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::I64 => 8,
        }
    }

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "BOOL" => Self::Bool,
            _ => return Err(Error::UnsupportedDtype),
        })
    }
}

/// Metadata for one tensor inside a safetensors file.
#[derive(Debug, Clone)]
struct TensorEntry {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    /// Absolute offsets in the original byte buffer.
    data_start: usize,
    data_end: usize,
}

/// A parsed safetensors file. Does not own the bytes — borrows from the
/// caller so the data can stay in an `mmap`, an `ArrayBuffer`, or any other
/// non-`Vec` storage.
#[derive(Debug)]
pub struct ModelFile<'a> {
    bytes: &'a [u8],
    tensors: Vec<TensorEntry>,
}

impl<'a> ModelFile<'a> {
    /// Parses a safetensors file from a byte slice.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::HeaderTooShort);
        }
        let header_len =
            u64::from_le_bytes(bytes[0..8].try_into().expect("checked len above")) as usize;
        if 8usize
            .checked_add(header_len)
            .is_none_or(|end| end > bytes.len())
        {
            return Err(Error::HeaderLengthOutOfBounds);
        }
        let header = &bytes[8..8 + header_len];
        let payload_start = 8 + header_len;
        let payload_len = bytes.len() - payload_start;

        // The header MUST be valid UTF-8 — safetensors mandates it.
        let header_str = core::str::from_utf8(header).map_err(|_| Error::HeaderNotUtf8)?;

        let mut cursor = Cursor::new(header_str.as_bytes());
        let tensors = parse_header_object(&mut cursor, payload_start, payload_len)?;

        Ok(Self { bytes, tensors })
    }

    /// Number of tensors in the file (excluding the `__metadata__` block).
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns `true` if the file contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterator over tensor names in declaration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    /// Returns a borrowed view of the named tensor.
    pub fn get(&self, name: &str) -> Result<TensorView<'_>> {
        let entry = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or(Error::TensorNotFound)?;
        Ok(TensorView {
            dtype: entry.dtype,
            shape: &entry.shape,
            raw: &self.bytes[entry.data_start..entry.data_end],
        })
    }
}

/// A view into a tensor inside a [`ModelFile`]. Borrows the underlying bytes.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// On-disk element type.
    pub dtype: Dtype,
    /// Shape, in declared order.
    pub shape: &'a [usize],
    /// Raw little-endian tensor bytes.
    pub raw: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Returns the tensor data as `&[f32]` without copying.
    ///
    /// Fails if the dtype is not `F32` or if the byte buffer is not aligned
    /// to 4 bytes. Use [`to_tensor`](Self::to_tensor) for an owned copy that
    /// works regardless of alignment.
    pub fn as_f32(&self) -> Result<&'a [f32]> {
        if self.dtype != Dtype::F32 {
            return Err(Error::DtypeMismatch);
        }
        bytemuck::try_cast_slice::<u8, f32>(self.raw).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => Error::Alignment,
            bytemuck::PodCastError::OutputSliceWouldHaveSlop
            | bytemuck::PodCastError::SizeMismatch => Error::UnevenLength,
            _ => Error::Alignment,
        })
    }

    /// Allocates an owned [`Tensor`] from this view (F32 only).
    ///
    /// Falls back to a manual little-endian copy if the underlying bytes
    /// are not aligned for direct `&[f32]` access.
    pub fn to_tensor(&self) -> Result<Tensor> {
        if self.dtype != Dtype::F32 {
            return Err(Error::DtypeMismatch);
        }
        let elem_size = self.dtype.size();
        if self.raw.len() % elem_size != 0 {
            return Err(Error::UnevenLength);
        }
        let expected: usize = self.shape.iter().product::<usize>() * elem_size;
        if expected != self.raw.len() {
            return Err(Error::ShapeDataMismatch);
        }

        // Prefer zero-copy when aligned; otherwise decode bytes by hand.
        let data: Vec<f32> = match bytemuck::try_cast_slice::<u8, f32>(self.raw) {
            Ok(slice) => slice.to_vec(),
            Err(_) => {
                let mut out = Vec::with_capacity(self.raw.len() / 4);
                for chunk in self.raw.chunks_exact(4) {
                    let arr: [u8; 4] = chunk.try_into().expect("chunks_exact(4)");
                    out.push(f32::from_le_bytes(arr));
                }
                out
            }
        };
        Ok(Tensor::from_vec(data, self.shape))
    }
}

// =============================================================================
// Minimal JSON parser tailored to safetensors headers.
//
// Not a general-purpose parser. It handles exactly the structure produced by
// safetensors: a top-level object whose keys are either tensor names mapping
// to `{dtype, shape, data_offsets}` objects, or the special `__metadata__`
// key (whose value we skip).
// =============================================================================

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        self.skip_ws();
        if self.advance() == Some(b) {
            Ok(())
        } else {
            Err(Error::InvalidHeader("unexpected byte"))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.skip_ws();
        if self.advance() != Some(b'"') {
            return Err(Error::InvalidHeader("expected string"));
        }
        let mut s = String::new();
        loop {
            let b = self
                .advance()
                .ok_or(Error::InvalidHeader("unterminated string"))?;
            match b {
                b'"' => return Ok(s),
                b'\\' => {
                    let esc = self
                        .advance()
                        .ok_or(Error::InvalidHeader("truncated escape"))?;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{0008}'),
                        b'f' => s.push('\u{000C}'),
                        // \uXXXX is not expected in tensor names; reject loudly.
                        _ => return Err(Error::InvalidHeader("unsupported escape")),
                    }
                }
                _ => s.push(b as char),
            }
        }
    }

    fn parse_u64(&mut self) -> Result<u64> {
        self.skip_ws();
        let mut n: u64 = 0;
        let mut any = false;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                n = n
                    .checked_mul(10)
                    .and_then(|n| n.checked_add((b - b'0') as u64))
                    .ok_or(Error::InvalidHeader("integer overflow"))?;
                self.pos += 1;
                any = true;
            } else {
                break;
            }
        }
        if !any {
            return Err(Error::InvalidHeader("expected integer"));
        }
        Ok(n)
    }

    fn parse_usize_array(&mut self) -> Result<Vec<usize>> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            out.push(self.parse_u64()? as usize);
            self.skip_ws();
            match self.advance() {
                Some(b',') => continue,
                Some(b']') => return Ok(out),
                _ => return Err(Error::InvalidHeader("expected ',' or ']'")),
            }
        }
    }

    fn skip_value(&mut self) -> Result<()> {
        self.skip_ws();
        let b = self.peek().ok_or(Error::InvalidHeader("unexpected end"))?;
        match b {
            b'{' => self.skip_object(),
            b'[' => self.skip_array(),
            b'"' => {
                let _ = self.parse_string()?;
                Ok(())
            }
            b't' => self.expect_lit(b"true"),
            b'f' => self.expect_lit(b"false"),
            b'n' => self.expect_lit(b"null"),
            b'-' | b'0'..=b'9' => self.skip_number(),
            _ => Err(Error::InvalidHeader("unexpected token")),
        }
    }

    fn skip_object(&mut self) -> Result<()> {
        self.expect(b'{')?;
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            let _ = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_value()?;
            self.skip_ws();
            match self.advance() {
                Some(b',') => {
                    self.skip_ws();
                    continue;
                }
                Some(b'}') => return Ok(()),
                _ => return Err(Error::InvalidHeader("expected ',' or '}'")),
            }
        }
    }

    fn skip_array(&mut self) -> Result<()> {
        self.expect(b'[')?;
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_value()?;
            self.skip_ws();
            match self.advance() {
                Some(b',') => continue,
                Some(b']') => return Ok(()),
                _ => return Err(Error::InvalidHeader("expected ',' or ']'")),
            }
        }
    }

    fn skip_number(&mut self) -> Result<()> {
        while let Some(b) = self.peek() {
            if matches!(
                b,
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            ) {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn expect_lit(&mut self, lit: &[u8]) -> Result<()> {
        for &b in lit {
            if self.advance() != Some(b) {
                return Err(Error::InvalidHeader("bad literal"));
            }
        }
        Ok(())
    }
}

fn parse_header_object(
    cursor: &mut Cursor<'_>,
    payload_start: usize,
    payload_len: usize,
) -> Result<Vec<TensorEntry>> {
    cursor.expect(b'{')?;
    let mut tensors = Vec::new();
    cursor.skip_ws();
    if cursor.peek() == Some(b'}') {
        cursor.pos += 1;
        return Ok(tensors);
    }
    loop {
        cursor.skip_ws();
        let name = cursor.parse_string()?;
        cursor.skip_ws();
        cursor.expect(b':')?;
        cursor.skip_ws();

        if name == "__metadata__" {
            cursor.skip_value()?;
        } else {
            let entry = parse_tensor_entry(cursor, name, payload_start, payload_len)?;
            tensors.push(entry);
        }

        cursor.skip_ws();
        match cursor.advance() {
            Some(b',') => continue,
            Some(b'}') => return Ok(tensors),
            _ => return Err(Error::InvalidHeader("expected ',' or '}' after entry")),
        }
    }
}

fn parse_tensor_entry(
    cursor: &mut Cursor<'_>,
    name: String,
    payload_start: usize,
    payload_len: usize,
) -> Result<TensorEntry> {
    cursor.expect(b'{')?;
    let mut dtype: Option<Dtype> = None;
    let mut shape: Option<Vec<usize>> = None;
    let mut offsets: Option<(usize, usize)> = None;

    cursor.skip_ws();
    if cursor.peek() == Some(b'}') {
        cursor.pos += 1;
        return Err(Error::InvalidHeader("empty tensor object"));
    }

    loop {
        cursor.skip_ws();
        let key = cursor.parse_string()?;
        cursor.skip_ws();
        cursor.expect(b':')?;
        cursor.skip_ws();
        match key.as_str() {
            "dtype" => {
                let s = cursor.parse_string()?;
                dtype = Some(Dtype::from_str(&s)?);
            }
            "shape" => {
                shape = Some(cursor.parse_usize_array()?);
            }
            "data_offsets" => {
                let arr = cursor.parse_usize_array()?;
                if arr.len() != 2 {
                    return Err(Error::InvalidHeader("data_offsets must have 2 entries"));
                }
                offsets = Some((arr[0], arr[1]));
            }
            _ => cursor.skip_value()?, // forward-compatible: ignore unknown fields
        }
        cursor.skip_ws();
        match cursor.advance() {
            Some(b',') => continue,
            Some(b'}') => break,
            _ => return Err(Error::InvalidHeader("expected ',' or '}' in tensor object")),
        }
    }

    let dtype = dtype.ok_or(Error::InvalidHeader("missing dtype"))?;
    let shape = shape.ok_or(Error::InvalidHeader("missing shape"))?;
    let (start, end) = offsets.ok_or(Error::InvalidHeader("missing data_offsets"))?;

    if start > end || end > payload_len {
        return Err(Error::DataOffsetsOutOfBounds);
    }

    // Shape-vs-bytes sanity check.
    let expected_bytes = shape.iter().product::<usize>() * dtype.size();
    if expected_bytes != end - start {
        return Err(Error::ShapeDataMismatch);
    }

    Ok(TensorEntry {
        name,
        dtype,
        shape,
        data_start: payload_start + start,
        data_end: payload_start + end,
    })
}

// =============================================================================
// Tests — build a small safetensors file by hand and round-trip it.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a synthetic safetensors file containing the given tensors.
    fn build_test_file(tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let dtype_str = match dtype {
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
                Dtype::I8 => "I8",
                Dtype::U8 => "U8",
                Dtype::I32 => "I32",
                Dtype::I64 => "I64",
                Dtype::Bool => "BOOL",
            };
            let shape_str = shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
                name,
                dtype_str,
                shape_str,
                offset,
                offset + data.len()
            ));
            offset += data.len();
        }
        // Add a metadata block to ensure the parser ignores it.
        header.push_str(r#","__metadata__":{"format":"pt"}"#);
        header.push('}');

        let header_bytes = header.into_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for (_, _, _, data) in tensors {
            out.extend_from_slice(data);
        }
        out
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parse_single_tensor() {
        let data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let file = build_test_file(&[("weight", Dtype::F32, vec![2, 2], data)]);
        let m = ModelFile::parse(&file).expect("parse");
        assert_eq!(m.len(), 1);
        let v = m.get("weight").expect("get");
        assert_eq!(v.dtype, Dtype::F32);
        assert_eq!(v.shape, &[2, 2]);
        let t = v.to_tensor().expect("to_tensor");
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn parse_multiple_tensors_preserves_order() {
        let a = f32_bytes(&[1.0, 2.0]);
        let b = f32_bytes(&[10.0, 20.0, 30.0]);
        let file = build_test_file(&[
            ("layer.weight", Dtype::F32, vec![2], a),
            ("layer.bias", Dtype::F32, vec![3], b),
        ]);
        let m = ModelFile::parse(&file).expect("parse");
        let names: Vec<&str> = m.names().collect();
        assert_eq!(names, vec!["layer.weight", "layer.bias"]);

        let bias = m.get("layer.bias").unwrap().to_tensor().unwrap();
        assert_eq!(bias.data(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn missing_tensor_returns_error() {
        let data = f32_bytes(&[1.0]);
        let file = build_test_file(&[("w", Dtype::F32, vec![1], data)]);
        let m = ModelFile::parse(&file).unwrap();
        assert!(matches!(m.get("missing"), Err(Error::TensorNotFound)));
    }

    #[test]
    fn truncated_file_is_rejected() {
        let short = vec![0u8; 4];
        assert!(matches!(ModelFile::parse(&short), Err(Error::HeaderTooShort)));
    }

    #[test]
    fn header_length_past_buffer_is_rejected() {
        let mut bad = (1024u64).to_le_bytes().to_vec();
        bad.extend_from_slice(b"{}");
        assert!(matches!(
            ModelFile::parse(&bad),
            Err(Error::HeaderLengthOutOfBounds)
        ));
    }

    #[test]
    fn shape_data_mismatch_is_rejected() {
        // Header claims a [2, 2] F32 tensor (= 16 bytes) but we provide only 8.
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#;
        let header_bytes = header.as_bytes();
        let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header_bytes);
        file.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            ModelFile::parse(&file),
            Err(Error::ShapeDataMismatch)
        ));
    }
}
