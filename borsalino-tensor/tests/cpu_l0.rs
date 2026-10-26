// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 CPU reference — L0 elementwise and buffer ops, adversarially.
//!
//! The shape classes here are the ones handover §5 names: boundary buffer
//! offsets, non-power-of-2 dimensions, zero-sized and single-element
//! operands, and huge-but-valid declared shapes whose buffers the test never
//! allocates.

mod common;

use borsalino_tensor::dtype::Dtype;
use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::{Executable, View};
use borsalino_tensor::ops::l0::{Add, BroadcastDiv, Copy, Gather, Mul};
use common::{SENTINEL, bits, f32_layout, layout, shape, u32_layout, view};

#[test]
fn copy_is_bit_exact_through_a_boundary_offset_window() {
    // The operand is a window inside a larger buffer: the loader's zero-copy
    // case. Sentinel values on both sides must never be read.
    let buffer = [
        SENTINEL,
        SENTINEL,
        SENTINEL,
        1.0,
        -0.0,
        f32::MIN,
        3.5,
        SENTINEL,
    ];
    let data = &buffer[3..7];
    let operand = f32_layout(&[2, 2]);
    let mut output = [SENTINEL; 4];

    Copy::execute(view(&operand, data), &mut output).expect("copy accepts a 2x2 window");

    assert_eq!(bits(&output), bits(data));
    // -0.0 survives as -0.0: a copy is bit-exact, not value-equal.
    assert_eq!(output[1].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn copy_zero_sized_operands_are_legal_and_write_nothing() {
    let operand = f32_layout(&[0, 4]);
    let mut output: [f32; 0] = [];

    Copy::execute(view(&operand, &[]), &mut output).expect("an empty operand copies nothing");
    assert!(output.is_empty());
}

#[test]
fn copy_single_element_scalars_round_trip() {
    let operand = f32_layout(&[]);
    let mut output = [SENTINEL; 1];

    Copy::execute(view(&operand, &[42.25]), &mut output).expect("a scalar copy");

    assert_eq!(bits(&output), bits(&[42.25]));
}

#[test]
fn copy_rejects_a_span_that_disagrees_with_its_layout() {
    let operand = f32_layout(&[4]);

    // Short operand: reading it would run past the end.
    let error = Copy::execute(view(&operand, &[1.0, 2.0, 3.0]), &mut [0.0; 4]);
    assert!(
        matches!(
            error,
            Err(TensorError::ByteLenMismatch {
                expected: 16,
                got: 12,
                ..
            })
        ),
        "a short operand is a typed error, not an out-of-bounds read: {error:?}"
    );

    // Long operand: the layout does not describe the extra element, so the
    // caller has mis-built the view.
    let error = Copy::execute(view(&operand, &[1.0, 2.0, 3.0, 4.0, 5.0]), &mut [0.0; 4]);
    assert!(matches!(
        error,
        Err(TensorError::ByteLenMismatch { got: 20, .. })
    ));
}

#[test]
fn copy_rejects_a_short_output_buffer_before_writing_anything() {
    let operand = f32_layout(&[4]);
    let mut output = [SENTINEL; 3];

    let error = Copy::execute(view(&operand, &[1.0, 2.0, 3.0, 4.0]), &mut output);

    assert!(matches!(error, Err(TensorError::ByteLenMismatch { .. })));
    assert_eq!(
        bits(&output),
        bits(&[SENTINEL; 3]),
        "a rejected call must leave the output buffer untouched"
    );
}

#[test]
fn copy_rejects_a_non_finite_operand_before_compute() {
    for offending in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let operand = f32_layout(&[4]);
        let mut output = [SENTINEL; 4];
        let data = [1.0, offending, 3.0, 4.0];

        let error = Copy::execute(view(&operand, &data), &mut output);

        assert!(
            matches!(
                error,
                Err(TensorError::NonFiniteInput {
                    op: "copy",
                    port: 0,
                    index: 1,
                })
            ),
            "non-finite data is rejected with its position: {error:?}"
        );
        assert_eq!(bits(&output), bits(&[SENTINEL; 4]));
    }
}

#[test]
fn copy_rejects_a_huge_but_valid_declared_shape() {
    // `[u32::MAX, 2]` f32 is ~34 GB: a valid T0 layout (the byte footprint
    // fits a u64) whose buffer the test never allocates. The op must reject
    // the four-element slice it actually got instead of trusting the shape.
    let operand = f32_layout(&[u32::MAX, 2]);
    assert_eq!(operand.byte_len(), u64::from(u32::MAX) * 2 * 4);

    let error = Copy::execute(view(&operand, &[1.0, 2.0, 3.0, 4.0]), &mut [0.0; 4]);

    assert!(
        matches!(error, Err(TensorError::ByteLenMismatch { .. })),
        "a shape the buffer cannot hold is a typed error: {error:?}"
    );
}

#[test]
fn add_broadcasts_a_trailing_row_against_a_matrix() {
    let lhs_layout = f32_layout(&[2, 3]);
    let rhs_layout = f32_layout(&[3]);
    let mut output = [SENTINEL; 6];

    Add::execute(
        (
            view(&lhs_layout, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            view(&rhs_layout, &[10.0, 20.0, 30.0]),
        ),
        &mut output,
    )
    .expect("a rank-1 operand broadcasts against a rank-2 one");

    assert_eq!(bits(&output), bits(&[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]));
}

#[test]
fn add_broadcasts_a_scalar_and_a_singleton_axis() {
    // Scalar against a matrix: the rank-0 operand has no strides at all.
    let matrix = f32_layout(&[2, 2]);
    let scalar = f32_layout(&[]);
    let mut output = [SENTINEL; 4];
    Add::execute(
        (view(&matrix, &[1.0, 2.0, 3.0, 4.0]), view(&scalar, &[0.5])),
        &mut output,
    )
    .expect("a scalar broadcasts");
    assert_eq!(bits(&output), bits(&[1.5, 2.5, 3.5, 4.5]));

    // A size-1 middle axis against a non-power-of-2 shape.
    let wide = f32_layout(&[3, 1, 5]);
    let narrow = f32_layout(&[1, 7, 5]);
    let left: Vec<f32> = (0..15).map(|value| value as f32).collect();
    let right: Vec<f32> = (0..35).map(|value| (value as f32) * 0.5).collect();
    let mut output = [SENTINEL; 105];
    Mul::execute((view(&wide, &left), view(&narrow, &right)), &mut output)
        .expect("[3,1,5] * [1,7,5] broadcasts to [3,7,5]");

    let expected = shape(&[3, 7, 5]);
    assert_eq!(output.len(), expected.element_count() as usize);
    // Spot-check the broadcast arithmetic against its own index formula.
    for row in 0..3 {
        for middle in 0..7 {
            for column in 0..5 {
                let at = row * 35 + middle * 5 + column;
                let want = left[row * 5 + column] * right[middle * 5 + column];
                assert_eq!(output[at].to_bits(), want.to_bits(), "at {at}");
            }
        }
    }
}

#[test]
fn mul_broadcasts_a_missing_leading_axis() {
    let lhs = f32_layout(&[2, 3]);
    let rhs = f32_layout(&[2, 1, 3, 1]);
    let mut output = [SENTINEL; 0];

    // [2,3] against [2,1,3,1] is incompatible (3 vs 1 and missing axes are
    // fine, but 2 vs 3 in the third-from-last is not) — the rule must say so.
    let error = Mul::execute((view(&lhs, &[1.0; 6]), view(&rhs, &[1.0; 6])), &mut output);
    assert!(
        matches!(error, Err(TensorError::BroadcastIncompatible { .. })),
        "broadcast compatibility is decided by the T0 rule: {error:?}"
    );

    // A genuinely missing leading axis does broadcast.
    let flat = f32_layout(&[3]);
    let deep = f32_layout(&[2, 3]);
    let mut output = [SENTINEL; 6];
    Mul::execute(
        (
            view(&deep, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            view(&flat, &[2.0, 2.0, 2.0]),
        ),
        &mut output,
    )
    .expect("[2,3] * [3] broadcasts");
    assert_eq!(bits(&output), bits(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]));
}

#[test]
fn add_rejects_a_non_f32_operand() {
    // The `Executable` surface makes a narrowed operand unrepresentable —
    // `Add::Operands` is two f32 views — so the dtype gate is exercised the
    // way a manifest or a dynamic consumer reaches it: by layout alone.
    let error = <Add as borsalino_tensor::op::TensorOp>::check_inputs(&[
        f32_layout(&[2]),
        layout(Dtype::F16, &[2]),
    ]);

    // The dtype gate fires even though the shapes agree: nothing narrows
    // silently into f32 arithmetic.
    assert!(
        matches!(
            error,
            Err(TensorError::DtypeMismatch {
                op: "add",
                port: 1,
                expected: Dtype::F32,
                got: Dtype::F16,
            })
        ),
        "a narrowed operand is rejected at the port: {error:?}"
    );
}

#[test]
fn add_rejects_an_output_buffer_of_the_wrong_shape() {
    let lhs = f32_layout(&[2, 3]);
    let rhs = f32_layout(&[3]);
    let mut output = [SENTINEL; 5];

    let error = Add::execute((view(&lhs, &[0.0; 6]), view(&rhs, &[0.0; 3])), &mut output);

    assert!(
        matches!(error, Err(TensorError::ByteLenMismatch { .. })),
        "the caller learns the output shape from infer_shape, and a\n         disagreement is typed: {error:?}"
    );
    assert_eq!(bits(&output), bits(&[SENTINEL; 5]));
}

#[test]
fn add_rejects_incompatible_broadcasts() {
    let lhs = f32_layout(&[2, 3]);
    let rhs = f32_layout(&[2, 4]);

    let error = Add::execute(
        (view(&lhs, &[0.0; 6]), view(&rhs, &[0.0; 8])),
        &mut [0.0; 6],
    );

    assert!(matches!(
        error,
        Err(TensorError::BroadcastIncompatible { .. })
    ));
}

#[test]
fn broadcast_div_divides_by_a_broadcast_divisor() {
    let values = f32_layout(&[2, 3]);
    let scale = f32_layout(&[3]);
    let mut output = [SENTINEL; 6];

    BroadcastDiv::execute(
        (
            view(&values, &[2.0, 4.0, 8.0, 3.0, 6.0, 9.0]),
            view(&scale, &[2.0, 2.0, 3.0]),
        ),
        &mut output,
    )
    .expect("a trailing divisor broadcasts");

    assert_eq!(bits(&output), bits(&[1.0, 2.0, 8.0 / 3.0, 1.5, 3.0, 3.0]));
}

#[test]
fn broadcast_div_by_zero_yields_the_ieee_result_not_an_error() {
    // T0: division by zero produces the IEEE-754 result. The *input* was
    // finite, so the gate passes; the non-finite output is deliberate.
    let values = f32_layout(&[2]);
    let divisors = f32_layout(&[2]);
    let mut output = [SENTINEL; 2];

    BroadcastDiv::execute(
        (view(&values, &[1.0, 0.0]), view(&divisors, &[0.0, 0.0])),
        &mut output,
    )
    .expect("division by zero is not trapped");

    assert_eq!(output[0], f32::INFINITY);
    assert!(output[1].is_nan());
}

#[test]
fn gather_copies_whole_table_rows() {
    // [4,3] table, three indices, one repeated and one at the last row.
    let table = f32_layout(&[4, 3]);
    let indices = u32_layout(&[3]);
    let data = [
        0.0, 0.5, 1.0, // row 0
        1.0, 1.5, 2.0, // row 1
        2.0, 2.5, 3.0, // row 2
        3.0, 3.5, 4.0, // row 3
    ];
    let mut output = [SENTINEL; 9];

    Gather::execute(
        (view(&table, &data), View::new(&indices, &[3, 0, 3])),
        &mut output,
    )
    .expect("in-range indices gather");

    assert_eq!(
        bits(&output),
        bits(&[3.0, 3.5, 4.0, 0.0, 0.5, 1.0, 3.0, 3.5, 4.0])
    );
}

#[test]
fn gather_rejects_an_index_past_the_table() {
    let table = f32_layout(&[4, 3]);
    let indices = u32_layout(&[3]);
    let mut output = [SENTINEL; 9];

    let error = Gather::execute(
        (view(&table, &[0.0; 12]), View::new(&indices, &[0, 4, 1])),
        &mut output,
    );

    assert!(
        matches!(
            error,
            Err(TensorError::IndexOutOfRange {
                op: "gather",
                position: 1,
                index: 4,
                rows: 4,
            })
        ),
        "the offending position names the index, not just the value: {error:?}"
    );
    assert_eq!(bits(&output), bits(&[SENTINEL; 9]));
}

#[test]
fn gather_accepts_a_zero_length_index_list() {
    let table = f32_layout(&[4, 3]);
    let indices = u32_layout(&[0]);
    let mut output: [f32; 0] = [];

    Gather::execute(
        (view(&table, &[0.0; 12]), View::new(&indices, &[])),
        &mut output,
    )
    .expect("an empty index list gathers nothing, and the table is still checked");

    // A zero-width table with indices is the other empty shape class.
    let flat_table = f32_layout(&[2, 0]);
    let one_index = u32_layout(&[1]);
    let mut output: [f32; 0] = [];
    Gather::execute(
        (view(&flat_table, &[]), View::new(&one_index, &[1])),
        &mut output,
    )
    .expect("[2,0] gather yields [1,0]");
}

#[test]
fn gather_rejects_a_rank_mismatch() {
    let table = f32_layout(&[3]);
    let indices = u32_layout(&[2]);

    let error = Gather::execute(
        (view(&table, &[0.0; 3]), View::new(&indices, &[0, 1])),
        &mut [0.0; 2],
    );

    assert!(matches!(
        error,
        Err(TensorError::RankMismatch { port: 0, .. })
    ));
}

#[test]
fn elementwise_ops_reject_a_short_operand_span_on_the_second_port() {
    let lhs = f32_layout(&[2]);
    let rhs = f32_layout(&[2]);

    let error = Add::execute((view(&lhs, &[1.0, 2.0]), view(&rhs, &[1.0])), &mut [0.0; 2]);

    assert!(
        matches!(error, Err(TensorError::ByteLenMismatch { .. })),
        "each operand is length-checked separately: {error:?}"
    );
}

#[test]
fn single_element_shapes_work_for_every_l0_op() {
    let one = f32_layout(&[1]);
    let mut output = [SENTINEL; 1];

    Add::execute((view(&one, &[2.0]), view(&one, &[3.0])), &mut output).expect("1x1 add");
    assert_eq!(bits(&output), bits(&[5.0]));

    Mul::execute((view(&one, &[2.0]), view(&one, &[3.0])), &mut output).expect("1x1 mul");
    assert_eq!(bits(&output), bits(&[6.0]));

    BroadcastDiv::execute((view(&one, &[6.0]), view(&one, &[3.0])), &mut output).expect("1x1 div");
    assert_eq!(bits(&output), bits(&[2.0]));

    let index = u32_layout(&[1]);
    let table = f32_layout(&[1, 1]);
    Gather::execute((view(&table, &[9.0]), View::new(&index, &[0])), &mut output)
        .expect("1x1 gather");
    assert_eq!(bits(&output), bits(&[9.0]));
}

#[test]
fn the_broadcast_odometer_wraps_correctly_on_non_power_of_two_dims() {
    // [3,5] against [5]: nine outputs, so the odometer carries twice. A
    // carry bug shows up as a wrong row offset, not as a crash.
    let lhs_layout = f32_layout(&[3, 5]);
    let rhs_layout = f32_layout(&[5]);
    let lhs: Vec<f32> = (0..15).map(|value| value as f32).collect();
    let rhs: Vec<f32> = (0..5).map(|value| (value as f32) * 10.0).collect();
    let mut output = [SENTINEL; 15];

    Add::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("[3,5] + [5]");

    // `[3,5] + [5]` and `[3,5] + [1,5]` are the same call: assert against the
    // second spelling rather than re-deriving the expected values by hand.
    let mut reference = [SENTINEL; 15];
    let explicit = f32_layout(&[1, 5]);
    Add::execute(
        (view(&lhs_layout, &lhs), view(&explicit, &rhs)),
        &mut reference,
    )
    .expect("[3,5] + [1,5]");

    assert_eq!(bits(&output), bits(&reference));
    assert_eq!(bits(&output[10..15]), bits(&[10.0, 21.0, 32.0, 43.0, 54.0]));
}
