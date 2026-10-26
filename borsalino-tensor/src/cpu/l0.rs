// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! CPU reference for L0 — elementwise and buffer ops.
//!
//! These are bit-exact by construction: no accumulation, so no summation
//! order to lose. The nonfinite gate still runs, because "bit-exact" says
//! nothing about whether a NaN should be allowed to travel (it should not —
//! handover §5, Amari's W3 wish).
//!
//! Broadcast indexing walks the output in row-major order with an odometer
//! over the output's coordinates and per-axis effective strides (0 on an
//! axis the operand repeats), so a broadcast operand is never re-strided and
//! never re-divided per element.

use crate::error::TensorError;
use crate::exec::{Executable, View, check_output, check_span, require_finite, validate};
use crate::layout::Shape;
use crate::ops::l0::{Add, BroadcastDiv, Copy, Gather, Mul};

impl Executable for Copy {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(source: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[source.layout])?;
        check_span::<Self>(0, source.layout, source.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, source.data)?;
        for (slot, &value) in output.iter_mut().zip(source.data) {
            *slot = value;
        }
        Ok(())
    }
}

impl Executable for Add {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (lhs, rhs): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        binary::<Self>(lhs, rhs, output, |left, right| left + right)
    }
}

impl Executable for Mul {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (lhs, rhs): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        binary::<Self>(lhs, rhs, output, |left, right| left * right)
    }
}

impl Executable for BroadcastDiv {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (lhs, rhs): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        // Division by zero is not trapped: T0 defines the result as the
        // IEEE-754 one (`inf`/`NaN`), and callers who need masking use
        // `masked_mean` instead. The output is deliberately not policed.
        binary::<Self>(lhs, rhs, output, |left, right| left / right)
    }
}

impl Executable for Gather {
    type Operands<'a> = (View<'a, f32>, View<'a, u32>);
    type Output = f32;

    fn execute(
        (table, indices): (View<'_, f32>, View<'_, u32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[table.layout, indices.layout])?;
        check_span::<Self>(0, table.layout, table.data.len())?;
        check_span::<Self>(1, indices.layout, indices.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, table.data)?;

        let rows = table.layout.shape().dims()[0];
        let width = table.layout.shape().dims()[1] as usize;
        // The whole index list is validated before a single row is copied, so
        // a rejected call leaves the output buffer untouched instead of
        // half-gathered.
        for (position, &index) in indices.data.iter().enumerate() {
            if index >= rows {
                return Err(TensorError::IndexOutOfRange {
                    op: <Self as crate::op::TensorOp>::NAME,
                    position: position as u64,
                    index,
                    rows,
                });
            }
        }
        for (position, &index) in indices.data.iter().enumerate() {
            let source = &table.data[index as usize * width..][..width];
            let target = &mut output[position * width..][..width];
            target.copy_from_slice(source);
        }
        Ok(())
    }
}

/// The shared broadcast elementwise core: one op function, three ops.
///
/// Index order is the output's row-major order, so the same inputs and the
/// same op name always read the same elements in the same sequence — which
/// is what makes L0 trivially replayable even though nothing here
/// accumulates.
fn binary<O: crate::op::TensorOp>(
    lhs: View<'_, f32>,
    rhs: View<'_, f32>,
    output: &mut [f32],
    combine: fn(f32, f32) -> f32,
) -> Result<(), TensorError> {
    let out_shape = validate::<O>(&[lhs.layout, rhs.layout])?;
    check_span::<O>(0, lhs.layout, lhs.data.len())?;
    check_span::<O>(1, rhs.layout, rhs.data.len())?;
    check_output::<O>(&out_shape, output.len())?;
    require_finite::<O>(0, lhs.data)?;
    require_finite::<O>(1, rhs.data)?;

    let dims = out_shape.dims();
    let rank = dims.len();
    let left_stride = broadcast_strides(lhs.layout.shape(), rank);
    let right_stride = broadcast_strides(rhs.layout.shape(), rank);
    let mut index = vec![0u32; rank];
    let mut left_offset = 0i64;
    let mut right_offset = 0i64;
    for slot in output.iter_mut() {
        *slot = combine(
            lhs.data[left_offset as usize],
            rhs.data[right_offset as usize],
        );
        advance(
            dims,
            &left_stride,
            &right_stride,
            &mut index,
            &mut left_offset,
            &mut right_offset,
        );
    }
    Ok(())
}

/// Per-axis strides for an operand broadcast against an output of `rank`
/// dimensions: `0` where the operand repeats along that axis.
///
/// A missing leading axis broadcasts, and a `1` broadcasts — both become a
/// stride of zero, which is what turns "repeat this element" into plain
/// indexing.
fn broadcast_strides(shape: &Shape, rank: usize) -> Vec<i64> {
    let offset = rank - shape.rank();
    let strides = shape.row_major_strides();
    let mut effective = vec![0i64; rank];
    for (axis, stride) in effective.iter_mut().enumerate() {
        if axis >= offset && shape.dims()[axis - offset] != 1 {
            *stride = strides.as_slice()[axis - offset] as i64;
        }
    }
    effective
}

/// Advance the row-major odometer after writing one output element, keeping
/// both operand offsets in step.
///
/// Offsets are `i64` so the carry can subtract a whole axis' worth of stride
/// without wrapping; they are non-negative by construction because every
/// coordinate reached here is inside its dimension.
fn advance(
    dims: &[u32],
    left_stride: &[i64],
    right_stride: &[i64],
    index: &mut [u32],
    left_offset: &mut i64,
    right_offset: &mut i64,
) {
    for axis in (0..dims.len()).rev() {
        index[axis] += 1;
        *left_offset += left_stride[axis];
        *right_offset += right_stride[axis];
        if index[axis] < dims[axis] {
            return;
        }
        index[axis] = 0;
        *left_offset -= left_stride[axis] * i64::from(dims[axis]);
        *right_offset -= right_stride[axis] * i64::from(dims[axis]);
    }
}
