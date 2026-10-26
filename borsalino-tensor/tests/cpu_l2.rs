// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 CPU reference — L2 GEMM, with the accumulation order pinned.
//!
//! The contract here is the `k`-chain (ascending, f32) and the empty-shape
//! classes: zero inner dimension, zero output dimension, empty batch and
//! head. Each expected value is produced by an independent left-to-right
//! fold written in this file, not by the implementation under test.

mod common;

use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::Executable;
use borsalino_tensor::ops::l2::{BatchedMatMul, MatMul};
use common::{SENTINEL, bits, f32_layout, view};

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
