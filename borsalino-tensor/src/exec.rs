// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The execution surface: what it means to *run* an op.
//!
//! T0 declares ops as types with no compute. T1 adds execution, and this
//! module is the one trait an implementation satisfies: [`Executable`].
//! [`cpu`](crate::cpu) implements it for every op in the inventory; T2's
//! WGSL kernels will implement the same trait behind their own wrapper
//! types, which is what makes [`oracle`](crate::oracle) — one harness, any
//! two implementations, bit-for-bit comparison — possible without either
//! implementation knowing about the other.
//!
//! ## The call shape
//!
//! An implementation borrows its inputs (a [`View`]: the T0 layout plus the
//! row-major elements it describes) and writes into a caller-provided output
//! buffer. No buffer is allocated for the result — element-sized work
//! happens in place, which is what lets the CPU path run on Ijima's
//! GPU-less hardware and what lets a test harness hand the same case to two
//! implementations cheaply.
//!
//! ## The gate order
//!
//! Every implementation runs the same checks, in this order, before it reads
//! a single element:
//!
//! 1. [`validate`] — the T0 gate: arity, per-port dtype, and the op's
//!    [`ShapeRule`](crate::op::ShapeRule). A contract violation is a typed
//!    error, never a panic and never compute-and-hope.
//! 2. [`check_span`] — each operand slice is exactly its layout's element
//!    count, so a short slice cannot be read past its end.
//! 3. [`check_output`] — the output buffer is exactly the inferred output
//!    shape's element count.
//! 4. [`require_finite`] — no NaN or infinity reached the compute.
//!
//! A call that fails any of these returns the typed error with the output
//! buffer **untouched**, so a caller can retry after fixing the operand.

use crate::dtype::DtypeMarker;
use crate::error::TensorError;
use crate::layout::{Shape, TensorLayout};
use crate::op::TensorOp;

/// One operation's executable form.
///
/// Implemented by the CPU reference ([`cpu`](crate::cpu)) for every op in
/// the T0 inventory, and by every later kernel. The trait deliberately says
/// nothing about *where* the compute happens: an implementation that
/// dispatches to a GPU and one that runs a scalar loop are
/// interchangeable, which is exactly the equivalence-test premise of
/// handover §5.
pub trait Executable {
    /// The borrowed operand views, in the op's port order.
    ///
    /// A single-operand op borrows a [`View`]; two-operand ops borrow a
    /// tuple of two. The types are the port dtypes' element types, so an
    /// `f16` operand is `u16` bits rather than a type std does not have.
    type Operands<'a>;

    /// The output buffer's element type — f32 except for [`Cast`], whose
    /// output type is its target dtype's.
    ///
    /// [`Cast`]: crate::ops::l0::Cast
    type Output: Copy;

    /// Validate the call, then execute it into the caller's `output` slice.
    ///
    /// # Errors
    ///
    /// Any gate failure (see the module docs), the cast-pair policy for
    /// [`Cast`](crate::ops::l0::Cast), a gather index past the table, or an
    /// empty operand for an op with no identity element.
    fn execute(
        operands: Self::Operands<'_>,
        output: &mut [Self::Output],
    ) -> Result<(), TensorError>;
}

/// A borrowed operand: the T0 layout plus the row-major elements it
/// describes.
///
/// The layout is by reference because it is the caller's: building one
/// validates contiguity and the byte footprint once
/// ([`TensorLayout::row_major`]), and every op in a call can borrow the same
/// layout without cloning. A `View` may describe a window inside a larger
/// buffer — the loader's zero-copy case — so an implementation must address
/// elements relative to `data`, never to a buffer origin.
#[derive(Clone, Copy, Debug)]
pub struct View<'a, T> {
    /// The layout the elements are interpreted with.
    pub layout: &'a TensorLayout,
    /// The row-major elements, exactly `layout.element_count()` of them.
    pub data: &'a [T],
}

impl<'a, T> View<'a, T> {
    /// Borrow a layout and its elements together.
    #[must_use]
    pub const fn new(layout: &'a TensorLayout, data: &'a [T]) -> Self {
        Self { layout, data }
    }
}

/// A dtype marker that also names its storage type and its conversion to
/// f32.
///
/// Cast is the only op that names two element types at once, so it needs
/// this bridge: `Cast<F32, F16>` reads `f32`s and writes `u16` bit
/// patterns. Implementing it is how a marker becomes a cast endpoint, which
/// keeps the defined cast pairs ([`CastContract`](crate::dtype::CastContract))
/// and the type-level marker set in agreement.
pub trait Element: DtypeMarker {
    /// The Rust type a buffer of this dtype is sliced as.
    ///
    /// f16 and bf16 have no std type, so their element type is `u16` holding
    /// the raw IEEE-754 binary16 / bfloat16 bit pattern. This crate does not
    /// add a dependency to change that (see the cast module).
    type Repr: Copy + core::fmt::Debug;

    /// Decode one element as f32. Exact for every widening decode.
    fn to_f32(repr: Self::Repr) -> f32;

    /// Encode an f32 as one element, per the dtype's cast contract.
    fn from_f32(value: f32) -> Self::Repr;
}

/// Run the T0 operand gate and return the output shape it implies.
///
/// This is [`TensorOp::check_inputs`] plus the inference a caller needs
/// right afterwards, so an implementation has one validation step instead of
/// two. The layouts are cloned into the owned slice the T0 gate takes; that
/// is `O(rank)` allocation per call, not `O(elements)`, and it is the price
/// of having exactly one gate implementation in the crate rather than a
/// borrowed copy of it that could drift.
///
/// # Errors
///
/// Whatever [`TensorOp::check_inputs`] rejects — including a [`Cast`] pair
/// with no defined conversion, because `Cast` overrides the gate.
///
/// [`Cast`]: crate::ops::l0::Cast
pub fn validate<O: TensorOp>(views: &[&TensorLayout]) -> Result<Shape, TensorError> {
    let owned: Vec<TensorLayout> = views.iter().map(|&view| view.clone()).collect();
    O::check_inputs(&owned)?;
    let shapes: Vec<&Shape> = views.iter().map(|view| view.shape()).collect();
    O::SHAPE_RULE.infer(O::NAME, &shapes)
}

/// Check that an operand slice is exactly its layout's element count.
///
/// # Errors
///
/// [`TensorError::ByteLenMismatch`] naming the op and port when the slice is
/// shorter or longer than the layout describes.
pub fn check_span<O: TensorOp + ?Sized>(
    port: usize,
    layout: &TensorLayout,
    len: usize,
) -> Result<(), TensorError> {
    let got = bytes_of(len as u64, layout.dtype().size_bytes());
    let expected = layout.byte_len();
    if got == expected {
        Ok(())
    } else {
        Err(TensorError::ByteLenMismatch {
            name: format!("op `{}` port {port}", O::NAME),
            shape: layout.shape().dims().to_vec(),
            dtype: layout.dtype(),
            expected,
            got,
        })
    }
}

/// Check that the caller's output buffer is exactly the inferred output
/// shape's element count.
///
/// The output buffer is a bare slice rather than a [`View`] because the op
/// owns the shape relation: the caller learns what to allocate from
/// [`TensorOp::infer_shape`], and this check makes a disagreement a typed
/// error instead of a panic at the first out-of-range write.
///
/// # Errors
///
/// [`TensorError::ByteLenMismatch`] naming the op's output when the buffer
/// is the wrong length.
pub fn check_output<O: TensorOp + ?Sized>(shape: &Shape, len: usize) -> Result<(), TensorError> {
    let size = O::OUTPUT.dtype.size_bytes();
    let got = bytes_of(len as u64, size);
    let expected = bytes_of(shape.element_count(), size);
    if got == expected {
        Ok(())
    } else {
        Err(TensorError::ByteLenMismatch {
            name: format!("op `{}` output", O::NAME),
            shape: shape.dims().to_vec(),
            dtype: O::OUTPUT.dtype,
            expected,
            got,
        })
    }
}

/// Whether an inferred output holds no elements, so there is nothing to
/// compute and nothing to write.
///
/// This is an early exit taken **after** the gate, not a shortcut around it:
/// arity, dtypes, the shape rule, every operand span, the output length, and
/// finiteness have all been checked by the time an implementation consults
/// it. Without it a legal shape whose output is empty still walks the
/// declared dimensions — `[u32::MAX, u32::MAX, 0, 0]` is a legal
/// batched-GEMM shape, and its batch loop would iterate 2^64 times over
/// nothing.
#[must_use]
pub fn writes_nothing(shape: &Shape) -> bool {
    shape.element_count() == 0
}

/// Reject a float operand that contains a NaN or an infinity.
///
/// The nonfinite policy (handover §5, Amari's W3 wish): non-finite data is
/// refused before compute, so a NaN can never decide an ordering or hide
/// inside a reduction. The cost is one pass over the operand, which is
/// documented rather than avoided: it is a linear scan with no allocation
/// and no dependence on shape.
///
/// The first offending element in row-major order is reported, so the same
/// operand always yields the same error.
///
/// # Errors
///
/// [`TensorError::NonFiniteInput`] with the port and the element index.
pub fn require_finite<O: TensorOp + ?Sized>(port: usize, data: &[f32]) -> Result<(), TensorError> {
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteInput {
                op: O::NAME,
                port,
                index: index as u64,
            });
        }
    }
    Ok(())
}

/// `count × size` in bytes, saturating rather than wrapping.
///
/// A count only ever comes from a slice length or a [`Shape`], both of which
/// are already bounded well below `u64::MAX / 8`; saturating keeps the
/// comparison honest instead of wrapping a huge count into a small one.
fn bytes_of(count: u64, size: u64) -> u64 {
    count.saturating_mul(size)
}
