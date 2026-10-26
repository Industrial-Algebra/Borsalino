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
//! ## Verdict first, poison as a diagnostic
//!
//! The verdict is **the comparison**. Both buffers are seeded with a poison
//! pattern so that an implementation which writes nothing cannot be mistaken
//! for one that wrote zeros, but the poison never decides the outcome: the
//! harness compares first, and consults the poison only to *explain* a
//! difference it has already found. A correct output whose bits happen to
//! equal a sentinel is therefore a pass, which is the rule this ordering
//! exists to guarantee.
//!
//! ## Why the sentinels are non-canonical NaN payloads
//!
//! A finite sentinel (the obvious `0xdead_beef`) collides with legal
//! results: a correct output equal to those bits was falsely reported as
//! unwritten. Moving the verdict to the comparison removes the false
//! *failure*; choosing the sentinel well narrows the opposite error, a false
//! *pass* (candidate writes nothing, reference legitimately produced exactly
//! the sentinel bits).
//!
//! The candidate sentinel is a **non-canonical quiet-NaN payload**
//! (`0x7fc0_0001` for f32/u32; `0x7f81`, a NaN in both the binary16 and
//! bfloat16 encodings, for u16). No IEEE-754 arithmetic result can carry a
//! non-canonical payload — conforming implementations canonicalise NaN
//! results (`0x7fc0_0000` for f32) — and no op on *finite* inputs produces a
//! NaN except through a `0/0`-class division. So a legitimate reference
//! output equal to the sentinel would require hardware that emits
//! non-canonical NaNs, which the contract does not admit.
//!
//! **Residual, documented rather than claimed impossible**: should some
//! implementation emit a non-canonical NaN that happens to equal the
//! sentinel bit for bit, a no-write candidate would pass. That is the only
//! false-pass class left, and it is a class the platform's NaN
//! canonicalisation rules out.
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

    /// The poison value a harness buffer starts with, for buffer `slot`.
    ///
    /// Slot `0` is the reference's buffer and slot `1` the candidate's, and
    /// the two patterns must differ so that "both wrote nothing" cannot
    /// masquerade as agreement. The value is a **diagnostic**: it is only
    /// consulted after the comparison has already failed, so a correct
    /// output that happens to equal it is still a pass.
    ///
    /// Implementations must return a **non-canonical quiet-NaN payload** for
    /// slot `1` (see the module docs): no conforming IEEE-754 result carries
    /// one, which is what keeps "wrote nothing" from being mistaken for a
    /// legitimate NaN result.
    fn poison(slot: u8) -> Self;
}

/// The poison pattern for `slot`, as a 32-bit word.
///
/// Both are quiet NaNs with **non-canonical payloads** (`0x7fc0_0000` is
/// f32's canonical quiet NaN): distinct from each other, and unreachable as
/// an arithmetic result on any conforming implementation. That is what keeps
/// the diagnostic from colliding with a legitimate value — the failure mode
/// of the earlier finite sentinels.
const fn poison_word(slot: u8) -> u32 {
    if slot == 0 { 0x7fc0_0003 } else { 0x7fc0_0001 }
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    fn poison(slot: u8) -> Self {
        f32::from_bits(poison_word(slot))
    }
}

impl Bits for u16 {
    fn bits(self) -> u64 {
        u64::from(self)
    }

    fn poison(slot: u8) -> Self {
        // `0x7f81` / `0x7f83` are NaNs with a non-canonical payload in *both*
        // narrow encodings this element type carries: binary16 (canonical
        // `0x7e00`) and bfloat16 (canonical `0x7fc0`). A cast from finite
        // data cannot produce either.
        if slot == 0 { 0x7f83 } else { 0x7f81 }
    }
}

impl Bits for u32 {
    fn bits(self) -> u64 {
        u64::from(self)
    }

    fn poison(slot: u8) -> Self {
        poison_word(slot)
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
    let mut expected = vec![T::poison(0); len];
    let mut actual = vec![T::poison(1); len];
    reference(&mut expected)?;
    if let Err(error) = candidate(&mut actual) {
        panic!("candidate rejected a case the reference accepts: {error}");
    }
    if let Some(index) = first_difference(&expected, &actual) {
        report_candidate_difference(index, &expected, &actual);
    }
    Ok(expected)
}

/// The first index at which two buffers disagree, or `None` when they agree.
///
/// Agreement is the verdict, so this runs before any poison is consulted: a
/// correct output whose bits equal a sentinel still passes.
fn first_difference<T: Bits>(expected: &[T], actual: &[T]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left.bits() != right.bits())
}

/// Report the first place a candidate diverged from the reference.
///
/// Only reached once the comparison has already failed, so the poison is
/// purely explanatory here: a candidate slot still holding its own sentinel
/// means the candidate never wrote that element.
///
/// # Panics
///
/// Always — this is the harness's verdict, not a fallible step.
fn report_candidate_difference<T: Bits>(index: usize, expected: &[T], actual: &[T]) -> ! {
    let differing = expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left.bits() != right.bits())
        .count();
    let len = expected.len();
    let got = actual[index].bits();
    if got == T::poison(1).bits() {
        panic!(
            "element {index} of {len} was never written: the candidate left \
             the poison pattern {got:#x} in its output buffer \
             ({differing} element(s) differ)"
        );
    }
    let want = expected[index].bits();
    panic!(
        "element {index} of {len} differs: reference {want:#x}, candidate \
         {got:#x} ({differing} element(s) differ)"
    );
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
    let mut actual = vec![T::poison(1); len];
    run(&mut actual)?;
    if let Some(index) = first_difference(expected, &actual) {
        let got = actual[index].bits();
        let want = expected[index].bits();
        if got == T::poison(1).bits() {
            panic!(
                "element {index} of {len} was never written: the run left \
                 the poison pattern {got:#x} in its output buffer"
            );
        }
        panic!("element {index} of {len} differs: golden {want:#x}, actual {got:#x}");
    }
    Ok(actual)
}
