// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! CPU reference for L1 — reductions.
//!
//! **Summation order is part of these ops' contract**, and it is fixed here
//! as strict left-to-right over the logical row-major element order, in a
//! single thread, accumulating in f32 (see the
//! [module policy](crate::cpu)). No tree, no lanes, no atomics: the same
//! inputs produce the same bits on every run and every platform, and any
//! implementation that walks the same indices reproduces them exactly.
//!
//! Ties and identities, all deliberate:
//!
//! - `sum` of an empty tensor is `0` — the addition identity.
//! - `max` of an empty tensor is an error, not `-inf`: `max` has no
//!   representable identity, and inventing a non-finite one would push an
//!   infinity into a pipeline whose inputs were all finite.
//! - `max` keeps the **first** of equal elements, so `-0.0` and `0.0`
//!   resolve to whichever appears first in row-major order (they compare
//!   equal, so strict `>` never replaces the incumbent).
//! - `rowwise_l2_norm` of an empty row is `0` (the square root of the empty
//!   sum), and `masked_mean` of an all-zero mask row is `0` rather than
//!   `NaN` (T0's shape rule already declares the shape legal).

use crate::error::TensorError;
use crate::exec::{
    Executable, View, check_output, check_span, require_finite, validate, writes_nothing,
};
use crate::op::TensorOp;
use crate::ops::l1::{MaskedMean, Max, RowwiseL2Norm, Sum};

impl Executable for Sum {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(values: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;

        let mut total = 0.0f32;
        for &value in values.data {
            total += value;
        }
        output[0] = total;
        Ok(())
    }
}

impl Executable for Max {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(values: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;

        let first = values
            .data
            .first()
            .ok_or(TensorError::EmptyReduction { op: Self::NAME })?;
        let mut best = *first;
        for &value in &values.data[1..] {
            // Strictly greater, so the earliest maximum wins a tie — that is
            // the -0.0 versus 0.0 case, and it must be replayable.
            if value > best {
                best = value;
            }
        }
        output[0] = best;
        Ok(())
    }
}

impl Executable for RowwiseL2Norm {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(values: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;

        let width = values.layout.shape().dims()[1] as usize;
        for (row, slot) in output.iter_mut().enumerate() {
            let mut sum_of_squares = 0.0f32;
            for &value in &values.data[row * width..][..width] {
                sum_of_squares += value * value;
            }
            // One correctly-rounded IEEE-754 square root on the reduced
            // value: no order left to vary, so the op stays deterministic.
            *slot = sum_of_squares.sqrt();
        }
        Ok(())
    }
}

impl Executable for MaskedMean {
    type Operands<'a> = (View<'a, f32>, View<'a, f32>);
    type Output = f32;

    fn execute(
        (values, mask): (View<'_, f32>, View<'_, f32>),
        output: &mut [f32],
    ) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout, mask.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_span::<Self>(1, mask.layout, mask.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;
        require_finite::<Self>(1, mask.data)?;
        // An empty output (a zero-width pool, or an empty token axis with a
        // huge batch) has nothing to write; the gate above still ran.
        if writes_nothing(&out_shape) {
            return Ok(());
        }

        let batch = values.layout.shape().dims()[0] as usize;
        let tokens = values.layout.shape().dims()[1] as usize;
        let width = values.layout.shape().dims()[2] as usize;

        for index in 0..batch {
            // The mask's own reduction uses the same left-to-right order as
            // the values' one: one contract, one order.
            let mut weight = 0.0f32;
            for token in 0..tokens {
                weight += mask.data[index * tokens + token];
            }
            for column in 0..width {
                let mut weighted = 0.0f32;
                for token in 0..tokens {
                    let at = index * tokens + token;
                    weighted += mask.data[at] * values.data[at * width + column];
                }
                output[index * width + column] = if weight == 0.0 {
                    // An all-zero mask row is a legal shape whose mean T0
                    // defines as zero, never NaN.
                    0.0
                } else {
                    weighted / weight
                };
            }
        }
        Ok(())
    }
}
