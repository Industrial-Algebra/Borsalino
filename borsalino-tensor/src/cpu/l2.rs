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
use crate::ops::l2::{BatchedMatMul, MatMul};

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
