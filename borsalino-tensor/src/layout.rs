// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Shapes, strides, and the row-major layout contract.
//!
//! T0 binds **row-major, contiguous, unpadded** buffers (handover §4.4).
//! That is not a limitation of convenience: it is what makes the Katakuri
//! manifest zero-copy. A manifest entry carries a shape and a byte span; the
//! stride is *implied* by the shape, so nothing has to be materialized to
//! reinterpret a weight buffer. Padded and transposed layouts are a later
//! slice, and they will arrive as an explicit layout rather than as an
//! accident of a stride vector.
//!
//! Element counts and byte lengths are checked once, here, at the contract
//! boundary: a shape that cannot be counted cannot be dispatched.

use crate::dtype::Dtype;
use crate::error::TensorError;

/// A tensor's dimensions, in row-major order.
///
/// Construction is the only place a shape can be rejected: the element
/// count is computed once and stored, so downstream arithmetic cannot
/// overflow. A zero dimension is legal (empty batch/head dims are an
/// adversarial *shape class*, not an invalidity — handover §5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<u32>,
    element_count: u64,
}

impl Shape {
    /// Build a shape from row-major dimensions.
    ///
    /// # Errors
    ///
    /// [`TensorError::ShapeOverflow`] when the dimensions' product exceeds
    /// `u64::MAX`.
    pub fn new(dims: &[u32]) -> Result<Self, TensorError> {
        let element_count = dims
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(u64::from(dim)));
        let element_count = element_count.ok_or_else(|| TensorError::ShapeOverflow {
            shape: dims.to_vec(),
        })?;
        Ok(Self {
            dims: dims.to_vec(),
            element_count,
        })
    }

    /// The dimensions, in row-major order.
    #[must_use]
    pub fn dims(&self) -> &[u32] {
        &self.dims
    }

    /// The number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The number of elements, `0` when any dimension is zero.
    #[must_use]
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    /// Whether this is the rank-0 scalar shape.
    ///
    /// A scalar holds exactly one element and is the contracted output of
    /// full reductions; `[0]` is *not* a scalar, it is empty.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// The canonical row-major strides, in element units.
    ///
    /// Degenerate zero-length shapes with very large trailing dimensions
    /// saturate rather than wrap: their strides are never used, because the
    /// buffer holds no elements.
    #[must_use]
    pub fn row_major_strides(&self) -> Strides {
        let mut strides = vec![0u64; self.dims.len()];
        let mut running = 1u64;
        for axis in (0..self.dims.len()).rev() {
            strides[axis] = running;
            running = running.saturating_mul(u64::from(self.dims[axis]));
        }
        Strides::new(&strides)
    }
}

/// Strides in element units, row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides {
    strides: Vec<u64>,
}

impl Strides {
    /// Wrap a stride vector. Container only — the layout checks the values.
    #[must_use]
    pub fn new(strides: &[u64]) -> Self {
        Self {
            strides: strides.to_vec(),
        }
    }

    /// The strides, one per dimension.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.strides
    }

    /// The number of dimensions described.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.strides.len()
    }
}

/// A dtype, a shape, and the strides they are addressed with.
///
/// The layout is the only thing a kernel needs besides the bytes: shape,
/// strides, and element type together describe what to read and where. It
/// carries no pointer and no length — binding a buffer is the consumer's
/// concern (handover §4.4), so this type is data a manifest can hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout {
    dtype: Dtype,
    shape: Shape,
    strides: Strides,
}

impl TensorLayout {
    /// The canonical row-major layout for a shape.
    #[must_use]
    pub fn row_major(dtype: Dtype, shape: Shape) -> Self {
        let strides = shape.row_major_strides();
        Self {
            dtype,
            shape,
            strides,
        }
    }

    /// An explicit layout, checked to be row-major contiguous.
    ///
    /// # Errors
    ///
    /// [`TensorError::RankMismatch`] when the stride count disagrees with
    /// the rank, and [`TensorError::NonContiguous`] when the strides are not
    /// the canonical ones for the shape.
    pub fn new(dtype: Dtype, shape: Shape, strides: Strides) -> Result<Self, TensorError> {
        if strides.rank() != shape.rank() {
            return Err(TensorError::RankMismatch {
                op: "layout",
                port: 0,
                expected: "one stride per dimension",
                got: strides.rank(),
            });
        }
        if strides != shape.row_major_strides() {
            return Err(TensorError::NonContiguous {
                shape: shape.dims().to_vec(),
                strides: strides.as_slice().to_vec(),
            });
        }
        Ok(Self {
            dtype,
            shape,
            strides,
        })
    }

    /// The element type.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The strides, in element units.
    #[must_use]
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Whether the strides are the canonical row-major ones.
    #[must_use]
    pub fn is_row_major_contiguous(&self) -> bool {
        self.strides == self.shape.row_major_strides()
    }

    /// The number of elements.
    #[must_use]
    pub const fn element_count(&self) -> u64 {
        self.shape.element_count()
    }

    /// The layout's byte footprint, `element_count * dtype size`.
    ///
    /// Saturating for the same reason as
    /// [`Shape::row_major_strides`]: a shape whose byte footprint exceeds
    /// `u64::MAX` cannot be bound, and
    /// [`check_byte_len`](Self::check_byte_len) rejects every real length
    /// against it.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.element_count().saturating_mul(self.dtype.size_bytes())
    }

    /// A human-readable label, used in error payloads.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{} {:?}", self.dtype, self.shape.dims())
    }

    /// Check a declared byte length against this layout's footprint.
    ///
    /// # Errors
    ///
    /// [`TensorError::ByteLenMismatch`] with the layout's label as the name.
    pub fn check_byte_len(&self, byte_len: u64) -> Result<(), TensorError> {
        self.check_byte_len_named(&self.label(), byte_len)
    }

    /// Check a declared byte length against this layout's footprint, naming
    /// the offender in the error.
    ///
    /// # Errors
    ///
    /// [`TensorError::ByteLenMismatch`] when `byte_len` is not exactly
    /// `element_count * dtype size`.
    pub fn check_byte_len_named(&self, name: &str, byte_len: u64) -> Result<(), TensorError> {
        let expected = self.byte_len();
        if expected == byte_len {
            Ok(())
        } else {
            Err(TensorError::ByteLenMismatch {
                name: name.to_string(),
                shape: self.shape.dims().to_vec(),
                dtype: self.dtype,
                expected,
                got: byte_len,
            })
        }
    }
}
