//! Error types used across the crate.
//!
//! Kept deliberately small: an enum with no payload allocations in the common
//! path, plus a `Result<T>` alias. Implements `std::error::Error` for
//! interoperability with `?` and `Box<dyn Error>` in user code.

use core::fmt;

/// Errors produced by `wasmicro` operations and the model loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The byte buffer is too short to contain a safetensors header.
    HeaderTooShort,
    /// The header length prefix points past the end of the buffer.
    HeaderLengthOutOfBounds,
    /// The header bytes are not valid UTF-8.
    HeaderNotUtf8,
    /// The header JSON failed to parse. The static string gives a brief reason.
    InvalidHeader(&'static str),
    /// The dtype string in the header is not recognized.
    UnsupportedDtype,
    /// A tensor's data offsets point past the end of the payload.
    DataOffsetsOutOfBounds,
    /// No tensor with the requested name exists in the file.
    TensorNotFound,
    /// The requested dtype does not match the dtype stored on disk.
    DtypeMismatch,
    /// The byte slice is not properly aligned for the requested element type.
    Alignment,
    /// The byte length is not a multiple of the element size.
    UnevenLength,
    /// The declared tensor shape does not match the data byte length.
    ShapeDataMismatch,
    /// The tokenizer vocabulary or tokenizer request is invalid.
    InvalidTokenizer(&'static str),
    /// The model configuration or inference input is invalid.
    InvalidInput(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTooShort => write!(f, "safetensors header is too short"),
            Self::HeaderLengthOutOfBounds => write!(f, "header length exceeds buffer size"),
            Self::HeaderNotUtf8 => write!(f, "header is not valid UTF-8"),
            Self::InvalidHeader(s) => write!(f, "invalid safetensors header: {}", s),
            Self::UnsupportedDtype => write!(f, "unsupported tensor dtype"),
            Self::DataOffsetsOutOfBounds => write!(f, "tensor data offsets are out of bounds"),
            Self::TensorNotFound => write!(f, "tensor not found in model file"),
            Self::DtypeMismatch => write!(f, "tensor dtype does not match the requested type"),
            Self::Alignment => write!(f, "tensor data is not aligned for the requested type"),
            Self::UnevenLength => write!(f, "tensor byte length is not a multiple of element size"),
            Self::ShapeDataMismatch => write!(f, "declared shape does not match data length"),
            Self::InvalidTokenizer(s) => write!(f, "invalid tokenizer: {}", s),
            Self::InvalidInput(s) => write!(f, "invalid input: {}", s),
        }
    }
}

impl std::error::Error for Error {}

/// Crate-wide result alias.
pub type Result<T> = core::result::Result<T, Error>;
