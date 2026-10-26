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
use borsalino_tensor::ops::l0::Cast;
use borsalino_tensor::ops::l1::{MaskedMean, Max, Sum};
use borsalino_tensor::ops::l2::MatMul;
use borsalino_tensor::oracle::{check_bit_identical, check_expected};
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
