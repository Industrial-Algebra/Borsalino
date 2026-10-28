// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! CPU reference for L2 — GEMM.
//!
//! **Accumulation order is part of these ops' contract** (T0's
//! [`FixedOrderGemm`](crate::op::DeterminismClass::FixedOrderGemm)): for
//! each output element, the contracted `k` dimension is summed ascending,
//! left to right, in f32. Outputs are written in row-major order. Nothing is
//! tiled, threaded, or vectorized in a way that could reorder the chain —
//! this is the fixed point every T2 kernel is measured against.
//!
//! The accumulation is deliberately f32 and not f64: WGSL has no f64, so an
//! f64 oracle would be un-matchable by the kernels it is meant to judge (see
//! the [module policy](crate::cpu)). `k == 0` therefore yields the addition
//! identity, `0.0`, for every output element, and empty output shapes
//! (`m == 0`, `n == 0`, an empty batch or head) write nothing at all.

use crate::error::TensorError;
use crate::exec::{
    Executable, View, check_output, check_span, require_finite, validate, writes_nothing,
};
use crate::op::TensorOp;
use crate::ops::l2::{BatchedMatMul, MatMul, Transpose2D};

impl Executable for MatMul {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (lhs, rhs): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[lhs.layout, rhs.layout])?;
        check_span::<Self>(0, lhs.layout, lhs.data.len())?;
        check_span::<Self>(1, rhs.layout, rhs.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, lhs.data)?;
        require_finite::<Self>(1, rhs.data)?;
        if writes_nothing(&out_shape) {
            return Ok(());
        }

        let m = lhs.layout.shape().dims()[0] as usize;
        let k = lhs.layout.shape().dims()[1] as usize;
        let n = rhs.layout.shape().dims()[1] as usize;
        gemm(lhs.data, rhs.data, output, m, k, n);
        Ok(())
    }
}

impl Executable for BatchedMatMul {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (lhs, rhs): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[lhs.layout, rhs.layout])?;
        check_span::<Self>(0, lhs.layout, lhs.data.len())?;
        check_span::<Self>(1, rhs.layout, rhs.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, lhs.data)?;
        require_finite::<Self>(1, rhs.data)?;
        // `[u32::MAX, u32::MAX, 0, 0]` is a legal shape pair: an empty output
        // reached through an enormous batch. Skipping the batch loop is what
        // keeps the call proportional to the work it actually has.
        if writes_nothing(&out_shape) {
            return Ok(());
        }

        let batch = lhs.layout.shape().dims()[0] as usize;
        let heads = lhs.layout.shape().dims()[1] as usize;
        let rows = lhs.layout.shape().dims()[2] as usize;
        let inner = lhs.layout.shape().dims()[3] as usize;
        let columns = rhs.layout.shape().dims()[3] as usize;

        // One independent GEMM per (batch, head) slice: the batch loop adds
        // no accumulation, so it cannot change any element's bit pattern.
        for slice in 0..batch * heads {
            let left = &lhs.data[slice * rows * inner..][..rows * inner];
            let right = &rhs.data[slice * inner * columns..][..inner * columns];
            let out = &mut output[slice * rows * columns..][..rows * columns];
            gemm(left, right, out, rows, inner, columns);
        }
        Ok(())
    }
}

/// Row-major `[m,k] x [k,n] -> [m,n]`, one output element at a time with an
/// ascending `k` chain.
///
/// The slice lengths are already validated by the caller, so every index
/// below is in bounds by construction.
fn gemm(lhs: &[f32], rhs: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for column in 0..n {
            let mut accumulation = 0.0f32;
            for contracted in 0..k {
                accumulation += lhs[row * k + contracted] * rhs[contracted * n + column];
            }
            output[row * n + column] = accumulation;
        }
    }
}

/// 2-D transpose, `[R, C] -> [C, R]` — a pure permutation of row-major
/// elements: `out[c * r + r_i] = in[r_i * c_total + c]`, walked once,
/// single-threaded. Bit-exact by construction (no arithmetic at all).
impl Executable for Transpose2D {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(values: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;
        let [rows, cols] = values.layout.shape().dims() else {
            // Unreachable in practice: the Transpose2D shape rule rejects
            // non-rank-2 inputs at `validate` above. Kept as an exhaustive
            // guard over the slice pattern.
            return Err(TensorError::RankMismatch {
                op: <Transpose2D as TensorOp>::NAME,
                port: 0,
                expected: "rank 2",
                got: values.layout.shape().rank(),
            });
        };
        // Empty work is constant-time, not proportional to the non-zero
        // dimension: [1e9, 0] must not iterate a billion empty rows
        // (review r2 P2). AFTER the complete gate — the contract checks
        // above still run.
        if writes_nothing(values.layout.shape()) {
            return Ok(());
        }
        for r in 0..*rows {
            for c in 0..*cols {
                let (out_at, in_at) = (
                    transpose_index(r, c, *rows, *cols),
                    r as u64 * *cols as u64 + c as u64,
                );
                output[out_at as usize] = values.data[in_at as usize];
            }
        }
        Ok(())
    }
}

/// The output-side transpose permutation index, `out[c, r] = in[r, c]`
/// flattened: widened to `u64` BEFORE multiplying — `c * rows` overflows
/// `u32` for valid shapes like `[65536, 65537]` at `c = 65536` (review r1
/// P1-2: the exact-arithmetic boundary case is pinned in the contract
/// tests without allocating).
pub const fn transpose_index(r: u32, c: u32, rows: u32, _cols: u32) -> u64 {
    c as u64 * rows as u64 + r as u64
}
