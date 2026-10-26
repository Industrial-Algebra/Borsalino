// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 — determinism, and the oracle harness T2's kernels are judged by.
//!
//! Two claims are checked here:
//!
//! 1. **The CPU reference is replayable.** Every op, run twice on the same
//!    inputs, produces the same bit patterns — the property the whole
//!    determinism class rests on.
//! 2. **The harness can tell.** [`check_bit_identical`] is fed deliberately
//!    wrong candidates (a reordered reduction, a one-ulp error, a flipped
//!    signed zero, an error where the reference succeeds) and must reject
//!    each one. A harness that cannot fail is a harness that proves nothing.

mod common;

use borsalino_tensor::dtype::{F16, F32};
use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::{Executable, View};
use borsalino_tensor::ops::l0::{Cast, Copy};
use borsalino_tensor::ops::l1::{MaskedMean, Max, Sum};
use borsalino_tensor::ops::l2::MatMul;
use borsalino_tensor::oracle::{Bits, check_bit_identical, check_expected};
use common::{bits, f32_layout, view};

/// A fixture that exercises every op below: a 3x5 operand, a 5x2 one.
fn matmul_case() -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..15).map(|value| (value as f32) * 0.25 - 1.0).collect();
    let rhs = (0..10).map(|value| (value as f32) * 0.5 - 2.0).collect();
    (lhs, rhs)
}

#[test]
fn every_op_is_replayable_bit_for_bit() {
    // Matmul, twice, on the same operands.
    let (lhs, rhs) = matmul_case();
    let lhs_layout = f32_layout(&[3, 5]);
    let rhs_layout = f32_layout(&[5, 2]);
    let mut first = [0.0f32; 6];
    let mut second = [0.0f32; 6];
    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut first,
    )
    .unwrap();
    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut second,
    )
    .unwrap();
    assert_eq!(bits(&first), bits(&second));

    // A reduction over 1,000 elements, twice: the order is fixed, so the
    // bits cannot move between runs.
    let wide_layout = f32_layout(&[1000]);
    let wide: Vec<f32> = (0..1000)
        .map(|value| (value as f32) * 1.000_001e-3)
        .collect();
    let mut first = [0.0f32; 1];
    let mut second = [0.0f32; 1];
    Sum::execute(view(&wide_layout, &wide), &mut first).unwrap();
    Sum::execute(view(&wide_layout, &wide), &mut second).unwrap();
    assert_eq!(bits(&first), bits(&second));

    // A masked pool, twice, on subnormal-magnitude data.
    let smallest = f32::from_bits(1);
    let values_layout = f32_layout(&[1, 4, 2]);
    let mask_layout = f32_layout(&[1, 4]);
    let values = [
        smallest,
        smallest,
        smallest * 3.0,
        smallest,
        0.0,
        smallest,
        smallest,
        smallest,
    ];
    let mask = [1.0f32, 1.0, 0.0, 1.0];
    let mut first = [0.0f32; 2];
    let mut second = [0.0f32; 2];
    MaskedMean::execute(
        (view(&values_layout, &values), view(&mask_layout, &mask)),
        &mut first,
    )
    .unwrap();
    MaskedMean::execute(
        (view(&values_layout, &values), view(&mask_layout, &mask)),
        &mut second,
    )
    .unwrap();
    assert_eq!(bits(&first), bits(&second));
}

#[test]
fn a_cast_is_replayable_bit_for_bit() {
    let source = f32_layout(&[3]);
    let values = [65504.0f32, 1.0 / 3.0, f32::from_bits(1)];

    let mut first = [0u16; 3];
    let mut second = [0u16; 3];
    <Cast<F32, F16> as Executable>::execute(View::new(&source, &values), &mut first).unwrap();
    <Cast<F32, F16> as Executable>::execute(View::new(&source, &values), &mut second).unwrap();
    assert_eq!(first, second);
}

#[test]
fn the_harness_accepts_the_reference_against_itself() {
    // This is the shape T2 uses: the CPU reference in one closure, the
    // kernel in the other.
    let (lhs, rhs) = matmul_case();
    let lhs_layout = f32_layout(&[3, 5]);
    let rhs_layout = f32_layout(&[5, 2]);

    let output = check_bit_identical(
        6,
        |out: &mut [f32]| MatMul::execute((view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)), out),
        |out: &mut [f32]| MatMul::execute((view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)), out),
    )
    .expect("the reference agrees with itself");

    assert_eq!(output.len(), 6);
}

#[test]
fn the_harness_handles_a_narrow_output_type() {
    // A cast kernel's output is u16 bit patterns, so the harness is generic
    // over the element type — not f32-only.
    let source = f32_layout(&[2]);
    let values = [1.0f32, -0.0];

    let output = check_bit_identical(
        2,
        |out: &mut [u16]| <Cast<F32, F16> as Executable>::execute(View::new(&source, &values), out),
        |out: &mut [u16]| <Cast<F32, F16> as Executable>::execute(View::new(&source, &values), out),
    )
    .expect("the cast is replayable");

    assert_eq!(output, vec![0x3c00, 0x8000]);
}

#[test]
fn the_harness_propagates_the_reference_error() {
    // A rejected case is returned, not swallowed: a test that expects a
    // rejection needs to see which one.
    let empty = f32_layout(&[0]);

    let error = check_bit_identical(
        1,
        |out: &mut [f32]| Max::execute(view(&empty, &[]), out),
        |out: &mut [f32]| {
            out[0] = 0.0;
            Ok(())
        },
    )
    .err();

    assert!(
        matches!(error, Some(TensorError::EmptyReduction { op: "max" })),
        "the reference's typed error is returned unchanged: {error:?}"
    );
}

#[test]
#[should_panic(expected = "differs")]
fn the_harness_rejects_a_reordered_reduction() {
    // The candidate sums right-to-left. On these values that changes the
    // bits, which is precisely what the determinism gate exists to catch.
    let value_layout = f32_layout(&[3]);
    let values = [1.0e8f32, -1.0e8, 1.0];

    let _ = check_bit_identical(
        1,
        |out: &mut [f32]| Sum::execute(view(&value_layout, &values), out),
        |out: &mut [f32]| {
            let mut total = 0.0f32;
            for &value in values.iter().rev() {
                total += value;
            }
            out[0] = total;
            Ok(())
        },
    );
}

#[test]
#[should_panic(expected = "differs")]
fn the_harness_rejects_a_one_ulp_difference() {
    let values = [1.5f32, 2.5];

    let _ = check_bit_identical(
        2,
        |out: &mut [f32]| {
            out.copy_from_slice(&values);
            Ok(())
        },
        |out: &mut [f32]| {
            out[0] = f32::from_bits(values[0].to_bits() + 1);
            out[1] = values[1];
            Ok(())
        },
    );
}

#[test]
#[should_panic(expected = "differs")]
fn the_harness_distinguishes_signed_zero() {
    let _ = check_bit_identical(
        1,
        |out: &mut [f32]| {
            out[0] = -0.0;
            Ok(())
        },
        |out: &mut [f32]| {
            out[0] = 0.0;
            Ok(())
        },
    );
}

#[test]
#[should_panic(expected = "candidate rejected")]
fn the_harness_rejects_a_candidate_that_errors_where_the_reference_succeeds() {
    let _ = check_bit_identical(
        1,
        |out: &mut [f32]| {
            out[0] = 1.0;
            Ok(())
        },
        |_out: &mut [f32]| Err(TensorError::EmptyReduction { op: "max" }),
    );
}

#[test]
fn check_expected_matches_a_golden_vector() {
    let golden = [2.0f32, 3.0, 5.0];

    let actual = check_expected(
        3,
        |out: &mut [f32]| {
            out.copy_from_slice(&[2.0, 3.0, 5.0]);
            Ok(())
        },
        &golden,
    )
    .expect("the golden vector matches");

    assert_eq!(bits(&actual), bits(&golden));
}

#[test]
#[should_panic(expected = "differs")]
fn check_expected_rejects_a_mismatched_element() {
    let _ = check_expected(
        2,
        |out: &mut [f32]| {
            out.copy_from_slice(&[1.0, 2.0]);
            Ok(())
        },
        &[1.0, 2.000_001],
    );
}

#[test]
#[should_panic(expected = "golden vector")]
fn check_expected_rejects_a_length_mismatch() {
    let _ = check_expected(
        3,
        |out: &mut [f32]| {
            out.copy_from_slice(&[1.0, 2.0, 3.0]);
            Ok(())
        },
        &[1.0, 2.0],
    );
}

#[test]
#[should_panic(expected = "was never written")]
fn the_harness_rejects_a_candidate_that_writes_nothing() {
    // The reference legitimately writes zeros. If both buffers started
    // zeroed, "wrote nothing" would be indistinguishable from "wrote
    // zeros" — so each side is poisoned with its own pattern instead.
    let _ = check_bit_identical(
        3,
        |out: &mut [f32]| {
            out.fill(0.0);
            Ok(())
        },
        |_out: &mut [f32]| Ok(()),
    );
}

#[test]
#[should_panic(expected = "was never written")]
fn the_harness_rejects_two_implementations_that_write_nothing() {
    // Even when both sides leave the buffer untouched the harness must not
    // agree: an op that writes nothing has not executed. The two poison
    // patterns are distinct precisely so this case fails loudly.
    let _ = check_bit_identical(2, |_out: &mut [f32]| Ok(()), |_out: &mut [f32]| Ok(()));
}

#[test]
#[should_panic(expected = "was never written")]
fn the_harness_rejects_a_candidate_that_writes_only_some_elements() {
    let _ = check_bit_identical(
        3,
        |out: &mut [f32]| {
            out.fill(2.5);
            Ok(())
        },
        |out: &mut [f32]| {
            out[0] = 2.5;
            out[1] = 2.5;
            Ok(())
        },
    );
}

#[test]
fn the_harness_accepts_a_candidate_that_writes_every_element() {
    // The poisoning must not make a correct run look wrong — including a run
    // whose legitimate output *is* zero.
    let output = check_bit_identical(
        2,
        |out: &mut [f32]| {
            out.fill(0.0);
            Ok(())
        },
        |out: &mut [f32]| {
            out.fill(0.0);
            Ok(())
        },
    )
    .expect("both sides wrote every element");

    assert_eq!(bits(&output), bits(&[0.0, 0.0]));
}

#[test]
#[should_panic(expected = "was never written")]
fn check_expected_rejects_a_run_that_writes_nothing() {
    let _ = check_expected(2, |_out: &mut [f32]| Ok(()), &[1.0, 2.0]);
}

#[test]
fn a_correct_output_that_collides_with_the_candidate_sentinel_still_passes() {
    // Reviewer repro (round 2): the harness's candidate poison is a finite
    // f32 bit pattern, so a *correct* output whose bits happen to equal it
    // was reported as "never written" — a false verdict on a real result.
    // The verdict must be the comparison, not the sentinel.
    let source = f32_layout(&[2]);
    let values = [f32::from_bits(0xdead_bee0), f32::from_bits(0xdead_beef)];

    let output = check_bit_identical(
        2,
        |out: &mut [f32]| Copy::execute(view(&source, &values), out),
        |out: &mut [f32]| Copy::execute(view(&source, &values), out),
    )
    .expect("agreement is the verdict, whatever the bytes are");

    assert_eq!(bits(&output), bits(&values));
}

#[test]
fn a_correct_output_that_collides_with_the_current_sentinel_still_passes() {
    // The same class at the harness level: both sides agree on the exact bit
    // pattern the candidate buffer was poisoned with.
    let poison = f32::from_bits(0xdead_bee0);
    let _ = check_bit_identical(
        3,
        |out: &mut [f32]| {
            out.fill(poison);
            Ok(())
        },
        |out: &mut [f32]| {
            out.fill(poison);
            Ok(())
        },
    )
    .expect("an output equal to the sentinel is a pass when both sides agree");
}

#[test]
fn a_golden_vector_that_collides_with_the_sentinel_still_passes() {
    let golden = [f32::from_bits(0xdead_bee0), f32::from_bits(0xdead_beef)];

    let actual = check_expected(
        2,
        |out: &mut [f32]| {
            out.copy_from_slice(&golden);
            Ok(())
        },
        &golden,
    )
    .expect("a golden value equal to the sentinel is a pass");

    assert_eq!(bits(&actual), bits(&golden));
}

#[test]
#[should_panic(expected = "was never written")]
fn a_no_write_candidate_behind_a_nan_reference_is_still_diagnosed() {
    // The realistic NaN case: the reference produced a canonical quiet NaN
    // (a 0/0-class result), the candidate wrote nothing. The sentinel must
    // not collide with the canonical payload, so the diagnosis survives.
    let canonical_nan = f32::from_bits(0x7fc0_0000);
    let _ = check_bit_identical(
        2,
        |out: &mut [f32]| {
            out.fill(canonical_nan);
            Ok(())
        },
        |_out: &mut [f32]| Ok(()),
    );
}

#[test]
fn the_candidate_sentinel_is_a_non_canonical_nan_payload() {
    // The whole residual-false-pass argument rests on this property: the
    // sentinel must be unreachable as an IEEE-754 *result*, which means a
    // NaN payload no conforming implementation canonicalises to. Pinned
    // here so a future "simplify the constant" edit cannot quietly restore
    // a producible sentinel.
    let candidate = <f32 as Bits>::poison(1);
    let reference = <f32 as Bits>::poison(0);
    assert!(
        candidate.is_nan(),
        "the sentinel must not be a finite value"
    );
    assert!(reference.is_nan());
    assert_ne!(
        candidate.to_bits(),
        0x7fc0_0000,
        "the canonical quiet NaN is a 0/0-class result, so it cannot be the sentinel"
    );
    assert_ne!(candidate.bits(), reference.bits(), "the slots must differ");

    // u16 carries binary16 and bfloat16 patterns: the sentinel must be a NaN
    // with a non-canonical payload in *both* encodings.
    let candidate = <u16 as Bits>::poison(1);
    assert_eq!((candidate >> 10) & 0x1f, 0x1f, "binary16 exponent all ones");
    assert_ne!(candidate & 0x03ff, 0, "binary16 mantissa nonzero (a NaN)");
    assert_eq!((candidate >> 7) & 0xff, 0xff, "bfloat16 exponent all ones");
    assert_ne!(candidate & 0x007f, 0, "bfloat16 mantissa nonzero (a NaN)");
    assert_ne!(candidate, 0x7e00, "not the binary16 canonical NaN");
    assert_ne!(candidate, 0x7fc0, "not the bfloat16 canonical NaN");
    assert_ne!(candidate, <u16 as Bits>::poison(0));
}
