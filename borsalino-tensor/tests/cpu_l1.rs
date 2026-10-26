// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 CPU reference — L1 reductions, with the summation order pinned.
//!
//! Every order assertion here is written so it would **fail** under a
//! different order: the fixture values are chosen so left-to-right and
//! right-to-left produce different bit patterns, and the test computes both
//! folds to prove the difference is real rather than assumed.

mod common;

use std::time::Duration;

use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::Executable;
use borsalino_tensor::ops::l1::{MaskedMean, Max, RowwiseL2Norm, Sum};
use common::{SENTINEL, bits, f32_layout, view, within};

/// An independent left-to-right fold — the order the contract fixes.
fn fold_left(values: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &value in values {
        total += value;
    }
    total
}

/// An independent right-to-left fold — the order the contract rejects.
fn fold_right(values: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for &value in values.iter().rev() {
        total += value;
    }
    total
}

#[test]
fn sum_pins_left_to_right_order() {
    // 1e8 swallows the 1.0 that follows it, so the two orders disagree:
    // left-to-right cancels first and keeps the 1.0, right-to-left loses it.
    let data = [1.0e8, -1.0e8, 1.0];
    assert_ne!(
        fold_left(&data).to_bits(),
        fold_right(&data).to_bits(),
        "the fixture must be order-sensitive for this test to mean anything"
    );

    let operand = f32_layout(&[3]);
    let mut output = [SENTINEL; 1];
    Sum::execute(view(&operand, &data), &mut output).expect("a rank-1 sum");

    assert_eq!(
        output[0].to_bits(),
        fold_left(&data).to_bits(),
        "the reference sums left to right"
    );
    assert_eq!(output[0], 1.0);
}

#[test]
fn sum_accumulates_in_f32_not_f64() {
    // 1e8 + 1 + 1 - 1e8: f32 accumulation loses both additions, f64 keeps
    // them. The contract is f32 (the kernels' arithmetic domain), so the
    // answer is 0, not 2.
    let data = [1.0e8, 1.0, 1.0, -1.0e8];
    let f64_total: f64 = data.iter().map(|&value| f64::from(value)).sum();
    assert_eq!(f64_total, 2.0);

    let operand = f32_layout(&[2, 2]);
    let mut output = [SENTINEL; 1];
    Sum::execute(view(&operand, &data), &mut output).expect("a 2x2 sum");

    assert_eq!(
        output[0], 0.0,
        "f32 accumulation is contract; f64 would give {f64_total}"
    );
}

#[test]
fn sum_flattens_in_row_major_order() {
    // The order is over the logical row-major element order, so a 2x3 sum is
    // bit-identical to summing the same six values as a rank-1 operand.
    let data = [1.0e8, 1.0, -1.0e8, 1.0, 1.0, 1.0];
    let matrix = f32_layout(&[2, 3]);
    let flat = f32_layout(&[6]);

    let mut from_matrix = [SENTINEL; 1];
    let mut from_flat = [SENTINEL; 1];
    Sum::execute(view(&matrix, &data), &mut from_matrix).expect("2x3 sum");
    Sum::execute(view(&flat, &data), &mut from_flat).expect("rank-1 sum");

    assert_eq!(bits(&from_matrix), bits(&from_flat));
    assert_eq!(from_matrix[0].to_bits(), fold_left(&data).to_bits());
}

#[test]
fn sum_of_an_empty_tensor_is_the_addition_identity() {
    let operand = f32_layout(&[0, 4]);
    let mut output = [SENTINEL; 1];

    Sum::execute(view(&operand, &[]), &mut output).expect("an empty sum is legal");

    assert_eq!(output[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn sum_rejects_non_finite_data_with_its_position() {
    for offending in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let operand = f32_layout(&[4]);
        let data = [1.0, 2.0, offending, 4.0];
        let mut output = [SENTINEL; 1];

        let error = Sum::execute(view(&operand, &data), &mut output);

        assert!(
            matches!(
                error,
                Err(TensorError::NonFiniteInput {
                    op: "sum",
                    port: 0,
                    index: 2,
                })
            ),
            "a non-finite summand is a typed rejection: {error:?}"
        );
        assert_eq!(output[0], SENTINEL);
    }
}

#[test]
fn sum_rejects_an_operand_span_that_disagrees_with_the_layout() {
    let operand = f32_layout(&[4]);

    let error = Sum::execute(view(&operand, &[1.0, 2.0]), &mut [0.0; 1]);

    assert!(matches!(error, Err(TensorError::ByteLenMismatch { .. })));
}

#[test]
fn max_keeps_the_first_of_equal_elements() {
    // -0.0 and 0.0 compare equal, so the tie-break decides the sign bit. The
    // order makes it replayable: the earliest maximum wins.
    let operand = f32_layout(&[2]);

    let mut output = [SENTINEL; 1];
    Max::execute(view(&operand, &[-0.0, 0.0]), &mut output).expect("a signed-zero max");
    assert_eq!(output[0].to_bits(), (-0.0f32).to_bits());

    let mut output = [SENTINEL; 1];
    Max::execute(view(&operand, &[0.0, -0.0]), &mut output).expect("a signed-zero max");
    assert_eq!(output[0].to_bits(), 0.0f32.to_bits());
}

#[test]
fn max_finds_the_largest_of_a_non_power_of_two_spread() {
    let operand = f32_layout(&[3, 5, 7]);
    let mut data: Vec<f32> = (0..105).map(|value| (value as f32) * 0.25).collect();
    data[73] = 99.5;
    let mut output = [SENTINEL; 1];

    Max::execute(view(&operand, &data), &mut output).expect("a 3x5x7 max");

    assert_eq!(output[0], 99.5);
}

#[test]
fn max_rejects_an_empty_tensor() {
    let operand = f32_layout(&[0]);
    let mut output = [SENTINEL; 1];

    let error = Max::execute(view(&operand, &[]), &mut output);

    assert!(
        matches!(error, Err(TensorError::EmptyReduction { op: "max" })),
        "max has no representable identity, so emptiness is an error: {error:?}"
    );
    assert_eq!(output[0], SENTINEL);
}

#[test]
fn rowwise_l2_norm_reduces_the_trailing_axis() {
    let operand = f32_layout(&[3, 4]);
    let data = [
        3.0, 4.0, 0.0, 0.0, // 5
        0.0, 0.0, 0.0, 0.0, // 0
        1.0, 1.0, 1.0, 1.0, // 2
    ];
    let mut output = [SENTINEL; 3];

    RowwiseL2Norm::execute(view(&operand, &data), &mut output).expect("a 3x4 norm");

    assert_eq!(bits(&output), bits(&[5.0, 0.0, 2.0]));
}

#[test]
fn rowwise_l2_norm_of_a_zero_width_row_is_zero() {
    let operand = f32_layout(&[2, 0]);
    let mut output = [SENTINEL; 2];

    RowwiseL2Norm::execute(view(&operand, &[]), &mut output).expect("an empty row norms to 0");

    assert_eq!(bits(&output), bits(&[0.0, 0.0]));
}

#[test]
fn rowwise_l2_norm_underflows_subnormal_rows_deterministically() {
    // The squares of subnormal-magnitude values are below f32's smallest
    // subnormal, so they flush to zero and the norm is 0 — not a NaN, not an
    // error. Pinned because it is the subnormal-spread class handover §5
    // names, and because a different summation order must not change it.
    let smallest = f32::from_bits(1); // 2^-149, the smallest f32 subnormal
    let operand = f32_layout(&[2, 3]);
    let data = [smallest, smallest * 3.0, smallest, 2.0, 0.0, -2.0];
    let mut output = [SENTINEL; 2];

    RowwiseL2Norm::execute(view(&operand, &data), &mut output).expect("a subnormal-spread norm");

    assert_eq!(output[0], 0.0);
    assert_eq!(output[1], 8.0f32.sqrt());
}

#[test]
fn rowwise_l2_norm_pins_left_to_right_summation() {
    // One large square followed by two hundred unit squares. Ascending, the
    // units are each below half an ulp of the running total and are lost as
    // they arrive; descending, they accumulate to 200 first, which *is* a
    // multiple of that ulp. The two orders therefore differ in the bits.
    let mut data = vec![1.0e4f32];
    data.extend(std::iter::repeat_n(1.0f32, 200));
    let squares: Vec<f32> = data.iter().map(|&value| value * value).collect();
    assert_ne!(
        fold_left(&squares).to_bits(),
        fold_right(&squares).to_bits(),
        "the fixture must be order-sensitive"
    );

    let operand = f32_layout(&[1, data.len() as u32]);
    let mut output = [SENTINEL; 1];
    RowwiseL2Norm::execute(view(&operand, &data), &mut output).expect("a 1x201 norm");

    assert_eq!(output[0].to_bits(), fold_left(&squares).sqrt().to_bits());
    assert_eq!(output[0], 1.0e4);
}

#[test]
fn masked_mean_pools_over_the_token_axis() {
    // [2,3,2] values, [2,3] mask: the padded token must not contribute.
    let values_layout = f32_layout(&[2, 3, 2]);
    let mask_layout = f32_layout(&[2, 3]);
    let values = [
        1.0, 1.0, // b0 t0
        3.0, 3.0, // b0 t1
        100.0, 100.0, // b0 t2 (masked out)
        2.0, 4.0, // b1 t0
        6.0, 8.0, // b1 t1
        10.0, 12.0, // b1 t2
    ];
    let mask = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut output = [SENTINEL; 4];

    MaskedMean::execute(
        (view(&values_layout, &values), view(&mask_layout, &mask)),
        &mut output,
    )
    .expect("a masked pool");

    // b0: (1+3)/2 = 2; b1: token 0 and token 2 only -> (2+10)/2, (4+12)/2.
    assert_eq!(bits(&output), bits(&[2.0, 2.0, 6.0, 8.0]));
}

#[test]
fn masked_mean_of_an_all_zero_mask_row_is_zero_not_nan() {
    let values_layout = f32_layout(&[2, 2, 1]);
    let mask_layout = f32_layout(&[2, 2]);
    let mut output = [SENTINEL; 2];

    MaskedMean::execute(
        (
            view(&values_layout, &[1.0, 2.0, 3.0, 4.0]),
            view(&mask_layout, &[0.0, 0.0, 1.0, 1.0]),
        ),
        &mut output,
    )
    .expect("an all-zero mask row is a legal shape");

    assert_eq!(bits(&output), bits(&[0.0, 3.5]));
}

#[test]
fn masked_mean_pins_its_left_to_right_order() {
    // Order-sensitive along the token axis: the mask sum and the weighted
    // sum both run ascending.
    let values_layout = f32_layout(&[1, 3, 1]);
    let mask_layout = f32_layout(&[1, 3]);
    let values = [1.0e8f32, -1.0e8, 1.0];
    let mask = [1.0f32, 1.0, 1.0];

    let weighted: Vec<f32> = values
        .iter()
        .zip(mask)
        .map(|(&value, weight)| weight * value)
        .collect();
    assert_ne!(
        fold_left(&weighted).to_bits(),
        fold_right(&weighted).to_bits()
    );

    let mut output = [SENTINEL; 1];
    MaskedMean::execute(
        (view(&values_layout, &values), view(&mask_layout, &mask)),
        &mut output,
    )
    .expect("a 1x3x1 masked pool");

    let expected = fold_left(&weighted) / fold_left(&mask);
    assert_eq!(output[0].to_bits(), expected.to_bits());
}

#[test]
fn masked_mean_rejects_a_rank_mismatch() {
    let values_layout = f32_layout(&[2, 3]);
    let mask_layout = f32_layout(&[2, 3]);

    let error = MaskedMean::execute(
        (
            view(&values_layout, &[0.0; 6]),
            view(&mask_layout, &[0.0; 6]),
        ),
        &mut [0.0; 2],
    );

    assert!(matches!(
        error,
        Err(TensorError::RankMismatch { port: 0, .. })
    ));
}

#[test]
fn masked_mean_rejects_a_batch_mismatch() {
    let values_layout = f32_layout(&[2, 3, 1]);
    let mask_layout = f32_layout(&[3, 3]);

    let error = MaskedMean::execute(
        (
            view(&values_layout, &[0.0; 6]),
            view(&mask_layout, &[0.0; 9]),
        ),
        &mut [0.0; 2],
    );

    assert!(matches!(
        error,
        Err(TensorError::BatchDimensionMismatch { .. })
    ));
}

#[test]
fn masked_mean_rejects_non_finite_mask_data() {
    let values_layout = f32_layout(&[1, 2, 1]);
    let mask_layout = f32_layout(&[1, 2]);
    let mut output = [SENTINEL; 1];

    let error = MaskedMean::execute(
        (
            view(&values_layout, &[1.0, 2.0]),
            view(&mask_layout, &[1.0, f32::NAN]),
        ),
        &mut output,
    );

    assert!(
        matches!(
            error,
            Err(TensorError::NonFiniteInput {
                op: "masked_mean",
                port: 1,
                index: 1,
            })
        ),
        "the mask is data, so it is gated like data: {error:?}"
    );
    assert_eq!(output[0], SENTINEL);
}

#[test]
fn masked_mean_rejects_a_zero_token_axis_output_buffer_mismatch() {
    // [2,0,3] pools to [2,3]; an empty output buffer must be rejected.
    let values_layout = f32_layout(&[2, 0, 3]);
    let mask_layout = f32_layout(&[2, 0]);

    let error = MaskedMean::execute(
        (view(&values_layout, &[]), view(&mask_layout, &[])),
        &mut [],
    );

    assert!(
        matches!(error, Err(TensorError::ByteLenMismatch { .. })),
        "a zero-token pool still has an output shape: {error:?}"
    );

    // ... and the empty pool of a nonzero output shape is all zeros.
    let mut output = [SENTINEL; 6];
    MaskedMean::execute(
        (view(&values_layout, &[]), view(&mask_layout, &[])),
        &mut output,
    )
    .expect("[2,0,3] pools to [2,3] of zeros");
    assert_eq!(bits(&output), bits(&[0.0; 6]));
}

#[test]
fn an_empty_masked_pool_returns_without_iterating() {
    // [MAX, 0, 0] values and [MAX, 0] mask pool to [MAX, 0]: an empty output
    // reached through a huge batch, so the batch loop must be skipped rather
    // than walked 2^32 times with no token and no column to visit.
    within(Duration::from_secs(2), || {
        let values_layout = f32_layout(&[u32::MAX, 0, 0]);
        let mask_layout = f32_layout(&[u32::MAX, 0]);
        let mut output: [f32; 0] = [];
        MaskedMean::execute(
            (view(&values_layout, &[]), view(&mask_layout, &[])),
            &mut output,
        )
        .expect("an empty pool over an empty token axis");
    });
}

#[test]
fn an_empty_masked_pool_still_runs_the_gate() {
    // The zero-token case already has its own output shape; a wrong buffer
    // length is still reported.
    let values_layout = f32_layout(&[2, 0, 3]);
    let mask_layout = f32_layout(&[2, 0]);
    let error = MaskedMean::execute(
        (view(&values_layout, &[]), view(&mask_layout, &[])),
        &mut [0.0; 5],
    );
    assert!(matches!(error, Err(TensorError::ByteLenMismatch { .. })));
}
