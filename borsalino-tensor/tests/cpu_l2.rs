// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 CPU reference — L2 GEMM, with the accumulation order pinned.
//!
//! The contract here is the `k`-chain (ascending, f32) and the empty-shape
//! classes: zero inner dimension, zero output dimension, empty batch and
//! head. Each expected value is produced by an independent left-to-right
//! fold written in this file, not by the implementation under test.

mod common;

use std::time::Duration;

use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::Executable;
use borsalino_tensor::ops::l2::{BatchedMatMul, MatMul, Transpose2D};
use common::{SENTINEL, bits, f32_layout, view, within};

/// An independent row-major `[m,k] x [k,n]` reference with an ascending,
/// f32 `k`-chain — the order the contract fixes.
fn reference_gemm(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut total = 0.0f32;
            for contracted in 0..k {
                total += lhs[row * k + contracted] * rhs[contracted * n + column];
            }
            out[row * n + column] = total;
        }
    }
    out
}

#[test]
fn matmul_matches_an_independent_reference_on_non_power_of_two_shapes() {
    let m = 3;
    let k = 5;
    let n = 7;
    let lhs: Vec<f32> = (0..m * k)
        .map(|value| (value as f32) * 0.25 - 1.0)
        .collect();
    let rhs: Vec<f32> = (0..k * n).map(|value| (value as f32) * 0.5 - 2.0).collect();

    let lhs_layout = f32_layout(&[m as u32, k as u32]);
    let rhs_layout = f32_layout(&[k as u32, n as u32]);
    let mut output = vec![SENTINEL; m * n];

    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("a 3x5 by 5x7 GEMM");

    assert_eq!(bits(&output), bits(&reference_gemm(&lhs, &rhs, m, k, n)));
}

#[test]
fn matmul_accumulates_ascending_in_k() {
    // lhs row [1e8, -1e8, 1.0] against a unit column: ascending, the two big
    // terms cancel first and the 1.0 survives; descending, it is absorbed.
    let lhs_layout = f32_layout(&[1, 3]);
    let rhs_layout = f32_layout(&[3, 1]);
    let lhs = [1.0e8f32, -1.0e8, 1.0];
    let rhs = [1.0f32, 1.0, 1.0];

    let mut ascending = [SENTINEL; 1];
    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut ascending,
    )
    .expect("a 1x3 by 3x1 GEMM");

    // The rejection case: the reverse k order loses the 1.0.
    let mut descending = 0.0f32;
    for offset in (0..3).rev() {
        descending += lhs[offset] * rhs[offset];
    }
    assert_ne!(ascending[0].to_bits(), descending.to_bits());

    assert_eq!(ascending[0], 1.0);
    assert_eq!(
        ascending[0].to_bits(),
        reference_gemm(&lhs, &rhs, 1, 3, 1)[0].to_bits()
    );
}

#[test]
fn matmul_accumulates_in_f32_not_f64() {
    // 1e8 + 1 + 1 - 1e8: the f32 chain loses both additions, an f64 chain
    // keeps them. The oracle must live in the kernels' arithmetic domain —
    // WGSL has no f64 — so the answer is 0, not 2.
    let lhs_layout = f32_layout(&[1, 4]);
    let rhs_layout = f32_layout(&[4, 1]);
    let lhs = [1.0e8f32, 1.0, 1.0, -1.0e8];
    let rhs = [1.0f32, 1.0, 1.0, 1.0];
    let f64_chain: f64 = lhs
        .iter()
        .map(|&value| f64::from(value))
        .zip(rhs.iter().map(|&value| f64::from(value)))
        .map(|(left, right)| left * right)
        .sum();
    assert_eq!(f64_chain, 2.0);

    let mut output = [SENTINEL; 1];
    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("a 1x4 by 4x1 GEMM");

    assert_eq!(
        output[0], 0.0,
        "f32 accumulation is contract; an f64 chain would give {f64_chain}"
    );
    assert_eq!(
        output[0].to_bits(),
        reference_gemm(&lhs, &rhs, 1, 4, 1)[0].to_bits()
    );
}

#[test]
fn matmul_of_a_zero_inner_dimension_is_the_addition_identity() {
    let lhs_layout = f32_layout(&[2, 0]);
    let rhs_layout = f32_layout(&[0, 3]);
    let mut output = [SENTINEL; 6];

    MatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
        &mut output,
    )
    .expect("[2,0] x [0,3] is a legal empty contraction");

    assert_eq!(bits(&output), bits(&[0.0; 6]));
}

#[test]
fn matmul_with_an_empty_output_shape_writes_nothing() {
    // Zero rows: no output elements at all.
    let lhs_layout = f32_layout(&[0, 3]);
    let rhs_layout = f32_layout(&[3, 2]);
    let mut output: [f32; 0] = [];
    MatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &[1.0; 6])),
        &mut output,
    )
    .expect("[0,3] x [3,2]");

    // Zero columns: same, from the other side.
    let lhs_layout = f32_layout(&[2, 3]);
    let rhs_layout = f32_layout(&[3, 0]);
    let mut output: [f32; 0] = [];
    MatMul::execute(
        (view(&lhs_layout, &[1.0; 6]), view(&rhs_layout, &[])),
        &mut output,
    )
    .expect("[2,3] x [3,0]");
}

#[test]
fn matmul_rejects_disagreeing_inner_dimensions() {
    let lhs_layout = f32_layout(&[2, 3]);
    let rhs_layout = f32_layout(&[4, 2]);

    let error = MatMul::execute(
        (view(&lhs_layout, &[0.0; 6]), view(&rhs_layout, &[0.0; 8])),
        &mut [0.0; 4],
    );

    assert!(matches!(
        error,
        Err(TensorError::InnerDimensionMismatch {
            op: "matmul",
            lhs: 3,
            rhs: 4,
        })
    ));
}

#[test]
fn matmul_rejects_a_short_output_buffer_for_a_huge_declared_shape() {
    // [u32::MAX, 1] x [1, u32::MAX] is a valid T0 shape pair whose output is
    // astronomically large. The op must reject the four-element buffer
    // instead of computing.
    let lhs_layout = f32_layout(&[u32::MAX, 1]);
    let rhs_layout = f32_layout(&[1, u32::MAX]);
    let mut output = [SENTINEL; 4];

    let error = MatMul::execute(
        (view(&lhs_layout, &[1.0]), view(&rhs_layout, &[1.0])),
        &mut output,
    );

    assert!(matches!(error, Err(TensorError::ByteLenMismatch { .. })));
    assert_eq!(bits(&output), bits(&[SENTINEL; 4]));
}

#[test]
fn matmul_survives_a_subnormal_magnitude_spread() {
    // Products of subnormals underflow to zero; products of one normal and
    // one subnormal stay subnormal until they stop being representable. All
    // of it is deterministic, so the reference is the contract.
    let smallest = f32::from_bits(1); // 2^-149
    let lhs = [smallest, smallest * 2.0, 1.0, smallest];
    let rhs = [smallest, 1.0, smallest * 4.0, 1.0];
    let lhs_layout = f32_layout(&[2, 2]);
    let rhs_layout = f32_layout(&[2, 2]);
    let mut output = [SENTINEL; 4];

    MatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("a subnormal-spread 2x2 GEMM");

    assert_eq!(bits(&output), bits(&reference_gemm(&lhs, &rhs, 2, 2, 2)));
    // Row 0 column 0 is the sum of two underflowed products: zero, not NaN.
    assert_eq!(output[0], 0.0);
}

#[test]
fn matmul_rejects_non_finite_operands_on_either_port() {
    let lhs_layout = f32_layout(&[1, 2]);
    let rhs_layout = f32_layout(&[2, 1]);
    let mut output = [SENTINEL; 1];

    let error = MatMul::execute(
        (
            view(&lhs_layout, &[1.0, f32::INFINITY]),
            view(&rhs_layout, &[1.0, 1.0]),
        ),
        &mut output,
    );
    assert!(matches!(
        error,
        Err(TensorError::NonFiniteInput {
            port: 0,
            index: 1,
            ..
        })
    ));

    let error = MatMul::execute(
        (
            view(&lhs_layout, &[1.0, 2.0]),
            view(&rhs_layout, &[f32::NAN, 1.0]),
        ),
        &mut output,
    );
    assert!(matches!(
        error,
        Err(TensorError::NonFiniteInput {
            port: 1,
            index: 0,
            ..
        })
    ));
    assert_eq!(output[0], SENTINEL);
}

#[test]
fn matmul_reads_operands_through_boundary_offset_windows() {
    // Both operands sit at a non-zero offset inside a larger buffer, with
    // sentinels around them: alignment-boundary reads are an adversarial
    // class, and a stray read shows up as a sentinel-sized result.
    let lhs_buffer = [SENTINEL, SENTINEL, 1.0, 2.0, 3.0, 4.0, SENTINEL];
    let rhs_buffer = [SENTINEL, 5.0, 6.0, 7.0, 8.0, SENTINEL, SENTINEL];
    let lhs = &lhs_buffer[2..6];
    let rhs = &rhs_buffer[1..5];
    let lhs_layout = f32_layout(&[2, 2]);
    let rhs_layout = f32_layout(&[2, 2]);
    let mut output = [SENTINEL; 4];

    MatMul::execute(
        (view(&lhs_layout, lhs), view(&rhs_layout, rhs)),
        &mut output,
    )
    .expect("a windowed 2x2 GEMM");

    assert_eq!(bits(&output), bits(&reference_gemm(lhs, rhs, 2, 2, 2)));
    assert_eq!(bits(&output), bits(&[19.0, 22.0, 43.0, 50.0]));
}

#[test]
fn batched_matmul_matches_two_independent_gemms() {
    // [2,2,3,4] x [2,2,4,5] -> [2,2,3,5]: four independent 3x4x5 GEMMs.
    let (batch, heads, rows, inner, columns) = (2usize, 2, 3, 4, 5);
    let lhs: Vec<f32> = (0..batch * heads * rows * inner)
        .map(|value| (value as f32) * 0.125 - 1.0)
        .collect();
    let rhs: Vec<f32> = (0..batch * heads * inner * columns)
        .map(|value| (value as f32) * 0.25 - 2.0)
        .collect();
    let lhs_layout = f32_layout(&[batch as u32, heads as u32, rows as u32, inner as u32]);
    let rhs_layout = f32_layout(&[batch as u32, heads as u32, inner as u32, columns as u32]);
    let mut output = vec![SENTINEL; batch * heads * rows * columns];

    BatchedMatMul::execute(
        (view(&lhs_layout, &lhs), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("a batched attention-projection GEMM");

    let mut expected = Vec::new();
    for slice in 0..batch * heads {
        let left = &lhs[slice * rows * inner..][..rows * inner];
        let right = &rhs[slice * inner * columns..][..inner * columns];
        expected.extend(reference_gemm(left, right, rows, inner, columns));
    }
    assert_eq!(bits(&output), bits(&expected));
}

#[test]
fn batched_matmul_accepts_empty_batch_and_head_dimensions() {
    // An empty batch is legal: the output is empty, not an error.
    let lhs_layout = f32_layout(&[0, 2, 3, 4]);
    let rhs_layout = f32_layout(&[0, 2, 4, 5]);
    let mut output: [f32; 0] = [];
    BatchedMatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
        &mut output,
    )
    .expect("[0,2,3,4] x [0,2,4,5]");

    // ... and so is an empty head dimension.
    let lhs_layout = f32_layout(&[2, 0, 3, 4]);
    let rhs_layout = f32_layout(&[2, 0, 4, 5]);
    let mut output: [f32; 0] = [];
    BatchedMatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
        &mut output,
    )
    .expect("[2,0,3,4] x [2,0,4,5]");

    // A zero sequence length inside a live batch is empty too.
    let lhs_layout = f32_layout(&[2, 2, 0, 4]);
    let rhs_layout = f32_layout(&[2, 2, 4, 5]);
    let rhs = vec![1.0f32; 2 * 2 * 4 * 5];
    let mut output: [f32; 0] = [];
    BatchedMatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &rhs)),
        &mut output,
    )
    .expect("[2,2,0,4] x [2,2,4,5]");
}

#[test]
fn batched_matmul_rejects_head_mismatches() {
    let lhs_layout = f32_layout(&[2, 2, 3, 4]);
    let rhs_layout = f32_layout(&[2, 3, 4, 5]);

    let error = BatchedMatMul::execute(
        (
            view(&lhs_layout, &[0.0; 48]),
            view(&rhs_layout, &[0.0; 120]),
        ),
        &mut [0.0; 60],
    );

    assert!(matches!(
        error,
        Err(TensorError::BatchDimensionMismatch { .. })
    ));
}

#[test]
fn batched_matmul_rejects_a_bad_rank() {
    let lhs_layout = f32_layout(&[2, 3, 4]);
    let rhs_layout = f32_layout(&[2, 4, 5]);

    let error = BatchedMatMul::execute(
        (view(&lhs_layout, &[0.0; 24]), view(&rhs_layout, &[0.0; 40])),
        &mut [0.0; 6],
    );

    assert!(matches!(
        error,
        Err(TensorError::RankMismatch { port: 0, .. })
    ));
}

#[test]
fn batched_matmul_rejects_disagreeing_inner_dimensions() {
    let lhs_layout = f32_layout(&[1, 1, 2, 3]);
    let rhs_layout = f32_layout(&[1, 1, 4, 2]);

    let error = BatchedMatMul::execute(
        (view(&lhs_layout, &[0.0; 6]), view(&rhs_layout, &[0.0; 8])),
        &mut [0.0; 4],
    );

    assert!(matches!(
        error,
        Err(TensorError::InnerDimensionMismatch { lhs: 3, rhs: 4, .. })
    ));
}

#[test]
fn an_empty_output_gemm_returns_without_iterating() {
    // [MAX, MAX, 0, 0] x [MAX, MAX, 0, 0] is a legal shape pair whose output
    // holds no elements. The batch loop must not walk 2^64 empty slices to
    // discover that.
    within(Duration::from_secs(2), || {
        let lhs_layout = f32_layout(&[u32::MAX, u32::MAX, 0, 0]);
        let rhs_layout = f32_layout(&[u32::MAX, u32::MAX, 0, 0]);
        let mut output: [f32; 0] = [];
        BatchedMatMul::execute(
            (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
            &mut output,
        )
        .expect("an empty batch of empty GEMMs");
    });

    // The 2-D shape of the same class: a huge row count with no columns, so
    // every row iteration would be empty.
    within(Duration::from_secs(2), || {
        let lhs_layout = f32_layout(&[u32::MAX, 0]);
        let rhs_layout = f32_layout(&[0, 0]);
        let mut output: [f32; 0] = [];
        MatMul::execute(
            (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
            &mut output,
        )
        .expect("an empty output GEMM");
    });
}

#[test]
fn an_empty_output_gemm_still_runs_the_full_gate() {
    // Skipping the compute loop must not skip validation: the checks are the
    // contract, the loop is the work.

    // Inner dimensions still have to agree.
    let lhs_layout = f32_layout(&[2, 2, 0, 4]);
    let rhs_layout = f32_layout(&[2, 2, 5, 5]);
    let error = BatchedMatMul::execute((view(&lhs_layout, &[]), view(&rhs_layout, &[])), &mut []);
    assert!(matches!(
        error,
        Err(TensorError::InnerDimensionMismatch { lhs: 4, rhs: 5, .. })
    ));

    // A non-finite operand is still rejected — even though the result would
    // have been empty, the input gate runs first.
    let lhs_layout = f32_layout(&[1, 1, 0, 2]);
    let rhs_layout = f32_layout(&[1, 1, 2, 3]);
    let rhs = [1.0f32, f32::NAN, 0.0, 0.0, 0.0, 0.0];
    let error = BatchedMatMul::execute((view(&lhs_layout, &[]), view(&rhs_layout, &rhs)), &mut []);
    assert!(matches!(
        error,
        Err(TensorError::NonFiniteInput {
            port: 1,
            index: 1,
            ..
        })
    ));

    // A non-empty output shape with a short buffer is still an error.
    let lhs_layout = f32_layout(&[2, 0]);
    let rhs_layout = f32_layout(&[0, 0]);
    let error = MatMul::execute(
        (view(&lhs_layout, &[]), view(&rhs_layout, &[])),
        &mut [0.0; 4],
    );
    assert!(matches!(error, Err(TensorError::ByteLenMismatch { .. })));
}

// ── transpose_2d (issue #71) ──────────────────────────────────────────

/// An independent `[r, c] -> [c, r]` permutation.
fn reference_transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = input[r * cols + c];
        }
    }
    out
}

#[test]
fn transpose_2d_permutes_a_non_square_matrix() {
    let rows = 3usize;
    let cols = 5usize;
    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.5 - 3.0).collect();
    let layout = f32_layout(&[rows as u32, cols as u32]);
    let mut output = vec![SENTINEL; rows * cols];
    Transpose2D::execute(view(&layout, &input), &mut output).unwrap();
    assert_eq!(
        bits(&output),
        bits(&reference_transpose(&input, rows, cols))
    );
    // Spot-check the corner mappings: [0,0] stays; in[0, cols-1] lands
    // at out[cols-1, 0]; in[rows-1, 0] lands at out[0, rows-1].
    assert_eq!(output[0], input[0]);
    assert_eq!(output[(cols - 1) * rows], input[cols - 1]);
    assert_eq!(output[rows - 1], input[(rows - 1) * cols]);
}

#[test]
fn transpose_2d_round_trips() {
    let input: Vec<f32> = (0..12).map(|i| i as f32 * 0.25).collect();
    let layout = f32_layout(&[3, 4]);
    let mut once = vec![SENTINEL; 12];
    Transpose2D::execute(view(&layout, &input), &mut once).unwrap();
    let layout_t = f32_layout(&[4, 3]);
    let mut twice = vec![SENTINEL; 12];
    Transpose2D::execute(view(&layout_t, &once), &mut twice).unwrap();
    assert_eq!(bits(&twice), bits(&input));
}

#[test]
fn transpose_2d_on_empty_dimensions_writes_nothing() {
    for dims in [[0u32, 4], [4, 0]] {
        let layout = f32_layout(&dims);
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        Transpose2D::execute(view(&layout, &input), &mut output).unwrap();
        assert!(output.is_empty());
    }
}

#[test]
fn transpose_2d_rejects_non_rank_2_shapes() {
    let layout = f32_layout(&[2, 3, 4]);
    let input = vec![0.0f32; 24];
    let mut output = vec![SENTINEL; 24];
    match Transpose2D::execute(view(&layout, &input), &mut output) {
        Err(TensorError::RankMismatch {
            op, expected, got, ..
        }) => {
            assert_eq!(op, "transpose_2d");
            assert_eq!(expected, "rank 2");
            assert_eq!(got, 3);
        }
        other => panic!("rank-3 input must be rejected, got {other:?}"),
    }
}

/// Review r1 P2-3: an empty shape must do zero work regardless of how
/// large the OTHER dimension is — [100000000, 0] took 589 ms with the
/// row loop iterating empty inner loops. The `within` timeout is the
/// assertion.
#[test]
fn transpose_2d_on_empty_rows_is_constant_time() {
    within(Duration::from_millis(100), || {
        let layout = f32_layout(&[1_000_000_000, 0]);
        let input: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        Transpose2D::execute(view(&layout, &input), &mut output).unwrap();
    });
}

#[test]
fn transpose_2d_composes_with_matmul_the_way_consumers_need_it() {
    // The issue #71 story end to end: Katakuri weights are `[out, in]`;
    // matmul contracts `[M, K] x [K, N]`. A transposed `[out, in]` weight
    // feeds the right operand directly.
    let m = 2usize;
    let k = 3usize;
    let n = 2usize;
    let activations = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [m, k]
    let weights_out_in = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [n, k] = [out, in]

    let w_layout = f32_layout(&[n as u32, k as u32]);
    let mut weights_kn = vec![SENTINEL; k * n];
    Transpose2D::execute(view(&w_layout, &weights_out_in), &mut weights_kn).unwrap();

    let a_layout = f32_layout(&[m as u32, k as u32]);
    let b_layout = f32_layout(&[k as u32, n as u32]);
    let mut out = vec![SENTINEL; m * n];
    MatMul::execute(
        (view(&a_layout, &activations), view(&b_layout, &weights_kn)),
        &mut out,
    )
    .unwrap();
    // Independent expectation: activations x weights^T, by hand.
    let expected = [
        1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0,
        1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0,
        4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0,
        4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0,
    ];
    assert_eq!(bits(&out), bits(&expected));
}
