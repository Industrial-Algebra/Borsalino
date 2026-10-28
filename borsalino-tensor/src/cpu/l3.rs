// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! CPU reference for L3 — activations.
//!
//! Elementwise and order-free: the row-major element order is walked
//! once, single-threaded, and each output element depends on exactly one
//! input element (see the [module policy](crate::cpu)). Bit-exact by
//! construction — the same inputs produce the same bits on every run and
//! every platform.

use crate::error::TensorError;
use crate::exec::{Executable, View, check_output, check_span, require_finite, validate};
use crate::ops::l3::Relu;

impl Executable for Relu {
    type Operands<'a> = View<'a, f32>;
    type Output = f32;

    fn execute(values: View<'_, f32>, output: &mut [f32]) -> Result<(), TensorError> {
        let out_shape = validate::<Self>(&[values.layout])?;
        check_span::<Self>(0, values.layout, values.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;
        require_finite::<Self>(0, values.data)?;
        // The contract's exact form (see the op docs): the comparison, not
        // `f32::max` — `max` may return either operand for -0.0 vs +0.0,
        // which would not be bit-exact. NaN never reaches here
        // (`require_finite`); were the gate lifted, the comparison sends
        // NaN to +0.0 deterministically.
        for (slot, &value) in output.iter_mut().zip(values.data) {
            *slot = if value > 0.0 { value } else { 0.0 };
        }
        Ok(())
    }
}
