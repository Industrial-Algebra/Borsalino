// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The equivalence harness: one entry point, any two implementations.
//!
//! Handover §5 gates every op as a triple — **CPU reference + WGSL kernel +
//! equivalence test**. This module is the third leg's machinery, and it is
//! written before the kernels so that a kernel is judged by the same code
//! the CPU reference is judged by: a case is a length plus a closure that
//! writes into a caller-provided buffer, so
//!
//! ```
//! use borsalino_tensor::oracle::check_bit_identical;
//!
//! let reference = |out: &mut [f32]| {
//!     out.copy_from_slice(&[1.0, 2.0]);
//!     Ok(())
//! };
//! // In T2 the second closure dispatches the WGSL kernel instead.
//! let candidate = |out: &mut [f32]| {
//!     out.copy_from_slice(&[1.0, 2.0]);
//!     Ok(())
//! };
//! check_bit_identical(2, reference, candidate).unwrap();
//! ```
//!
//! ## Why bit patterns and not tolerances
//!
//! L0–L2 are exact classes: an elementwise op has no freedom, and a
//! reduction's freedom is the summation order, which the determinism
//! contract fixes. So the comparison is on **bit patterns**, not `==` and
//! not an epsilon: `-0.0` and `0.0` are different results, and two NaNs
//! compare equal only if their bits match. Transcendental ops (L3's
//! activations, softmax) will need a tolerance class of their own; they are
//! not in this inventory.
//!
//! ## Why these panic
//!
//! This is an *assertion* harness: a mismatch is a failed test, and the
//! panic message carries the index, both bit patterns, and the values. An
//! op's own typed error is returned to the caller unchanged
//! ([`TensorError`]), because a test that expects a rejection needs to see
//! which one.

use crate::error::TensorError;

/// An output element comparable by bit pattern.
pub trait Bits: Copy {
    /// The element as a `u64` bit pattern, for exact comparison.
    fn bits(self) -> u64;

    /// The zero value the harness allocates its buffers with.
    ///
    /// Zero is only ever a placeholder: every element of a correct run is
    /// overwritten, and an implementation that leaves one behind fails the
    /// comparison against the reference — which is exactly the bug this
    /// catches.
    fn zero() -> Self;
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    fn zero() -> Self {
        0.0
    }
}

impl Bits for u16 {
    fn bits(self) -> u64 {
        u64::from(self)
    }

    fn zero() -> Self {
        0
    }
}

impl Bits for u32 {
    fn bits(self) -> u64 {
        u64::from(self)
    }

    fn zero() -> Self {
        0
    }
}

/// Run two implementations of the same op on the same case and require
/// bit-identical outputs.
///
/// `len` is the output element count, which both implementations' own gates
/// check against the inferred output shape — the harness does not know the
/// op, so a wrong `len` surfaces as each implementation's typed
/// [`TensorError::ByteLenMismatch`].
///
/// # Errors
///
/// Whatever the reference implementation returns. A candidate error is
/// reported as a panic: the reference defines the contract, so a candidate
/// that rejects a case the reference accepts has diverged.
///
/// # Panics
///
/// When the outputs differ, or when the candidate errors where the
/// reference succeeded. This is an assertion harness, not an operational
/// API.
pub fn check_bit_identical<T: Bits>(
    len: usize,
    reference: impl FnOnce(&mut [T]) -> Result<(), TensorError>,
    candidate: impl FnOnce(&mut [T]) -> Result<(), TensorError>,
) -> Result<Vec<T>, TensorError> {
    let mut expected = vec![T::zero(); len];
    let mut actual = vec![T::zero(); len];
    reference(&mut expected)?;
    if let Err(error) = candidate(&mut actual) {
        panic!("candidate rejected a case the reference accepts: {error}");
    }
    for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(
            left.bits(),
            right.bits(),
            "element {index} differs: reference {:#x}, candidate {:#x}",
            left.bits(),
            right.bits()
        );
    }
    Ok(expected)
}

/// Run one implementation and require its output to match a golden vector
/// bit for bit.
///
/// This is the direction the composed-path golden tests need (handover §5):
/// published vectors in, reference out, compared exactly.
///
/// # Errors
///
/// Whatever the implementation returns.
///
/// # Panics
///
/// When the output differs from `expected`, or when `len` disagrees with
/// `expected.len()`. This is an assertion harness.
pub fn check_expected<T: Bits>(
    len: usize,
    run: impl FnOnce(&mut [T]) -> Result<(), TensorError>,
    expected: &[T],
) -> Result<Vec<T>, TensorError> {
    assert_eq!(
        len,
        expected.len(),
        "the golden vector is {len} elements, the harness was given {}",
        expected.len()
    );
    let mut actual = vec![T::zero(); len];
    run(&mut actual)?;
    for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(
            left.bits(),
            right.bits(),
            "element {index} differs: golden {:#x}, actual {:#x}",
            left.bits(),
            right.bits()
        );
    }
    Ok(actual)
}
