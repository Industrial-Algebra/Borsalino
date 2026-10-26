// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The CPU reference for [`Cast`], with in-crate f16/bf16 bit conversion.
//!
//! T0 pinned the cast *semantics* ([`CastContract`](crate::dtype::CastContract)) and deferred the
//! arithmetic. This module makes them executable, and it does so **without a
//! dependency**: both conversions are well-understood bit math (round to
//! nearest even, handle the subnormal ramp, overflow at the contracted
//! midpoint), about forty lines each way. Adding a `half`-style crate would
//! be a new dependency with no justification beyond convenience, against the
//! handover §8 rule that every dependency needs one.
//!
//! ## Where the contract lives in the code
//!
//! - `f32 -> f16`: f16's exponent field is 5 bits, so the rebias is
//!   `exp - 127 + 15`. Values whose f16 exponent is `<= 0` become f16
//!   subnormals (this includes plain f32 *normals* in `[2^-24, 2^-14)`: only
//!   f32 *sub* normals flush to zero, and they flush because they are below
//!   half of f16's minimum subnormal — the arithmetic says so, nothing is
//!   special-cased). Rounding carries out of the mantissa renormalize into
//!   the exponent, which is exactly the midpoint overflow the contract
//!   documents: 65504 stays 65504, 65520 becomes infinity.
//! - `f16 -> f32`, `bf16 -> f32`: exact, including subnormals, because f32's
//!   exponent range strictly contains both.
//! - `f32 -> bf16`: bf16 is the top 16 bits of an f32, so the conversion is
//!   truncate-and-round on the low half. Because bf16 keeps f32's 8-bit
//!   exponent, f32 subnormals land on bf16 subnormals down to 2^-133 and
//!   then round to zero — no exponent special case is needed either.
//!
//! The tests in `tests/cast_fixtures.rs` execute every threshold in the T0
//! contract, and roundtrip all 65,536 bit patterns of each narrow dtype.

use crate::dtype::{BF16, F16, F32};
use crate::error::TensorError;
use crate::exec::{Element, Executable, View, check_output, check_span, validate};
use crate::ops::l0::Cast;

/// The f32 bit pattern of 2^-24, constructed from its exponent so it is
/// exact by construction rather than by decimal literal.
const TWO_POW_M24: f32 = f32::from_bits((127 - 24) << 23);

/// Round an f32 to IEEE-754 binary16, returning the raw bit pattern.
///
/// Round-to-nearest-even, matching [`CastContract`](crate::dtype::CastContract) for `f32 -> f16`.
/// Non-finite inputs are outside the input gate (the cast rejects them
/// before compute); if one is passed anyway the result is the corresponding
/// f16 infinity or a NaN, never a wrong finite value.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    // f16's exponent field is all ones: infinity stays infinity, a NaN stays
    // a NaN (payload truncated, which is all binary16 can carry).
    if ((bits >> 23) & 0xff) == 0xff {
        let payload = if mantissa != 0 { 0x0200 } else { 0 };
        return (sign | 0x7c00 | payload) as u16;
    }

    if exponent <= 0 {
        // f16 subnormal ramp: shift the 24-bit significand right, keeping a
        // sticky remainder for round-to-nearest-even. Below -10 the value is
        // under half of the minimum subnormal (2^-25), hence zero.
        if exponent < -10 {
            return sign as u16;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - exponent) as u32;
        let mut result = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (result & 1) == 1) {
            result += 1;
        }
        return (sign | result) as u16;
    }

    if exponent >= 0x1f {
        // Beyond the largest binade: the contract's overflow threshold is
        // the midpoint, and everything past it is infinity.
        return (sign | 0x7c00) as u16;
    }

    let mut result = sign | ((exponent as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    let halfway = 0x1000;
    if remainder > halfway || (remainder == halfway && (result & 1) == 1) {
        // A carry out of the mantissa is a renormalization: 65504 + half an
        // ulp rounds to 2^16, which is f16 infinity.
        result += 1;
    }
    result as u16
}

/// Widen an IEEE-754 binary16 bit pattern to f32. Exact for every pattern.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);

    if exponent == 0x1f {
        // Infinity or NaN: rebuild the f32 field, payload shifted up.
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    if exponent == 0 {
        // Subnormal (or zero): the value is mantissa * 2^-24, exact in f32
        // because the scale is a power of two and the mantissa is 10 bits.
        let magnitude = (mantissa as f32) * TWO_POW_M24;
        return f32::from_bits(magnitude.to_bits() | sign);
    }
    let widened = (exponent + (127 - 15)) << 23;
    f32::from_bits(sign | widened | (mantissa << 13))
}

/// Round an f32 to bfloat16, returning the raw bit pattern.
///
/// Round-to-nearest-even on the truncated low half, matching
/// [`CastContract`](crate::dtype::CastContract) for `f32 -> bf16`.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let low = bits & 0xffff;
    let mut high = (bits >> 16) as u16;

    // Keep a NaN a NaN: if truncating the low half would leave the infinity
    // pattern, set the smallest significand bit instead. Infinity itself
    // passes through, and overflow past bf16's own maximum renormalizes to
    // infinity by the carry below.
    let is_nan = (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0;
    if is_nan {
        let mut result = high;
        if result & 0x007f == 0 {
            result |= 0x0001;
        }
        return result;
    }

    if low > 0x8000 || (low == 0x8000 && (high & 1) == 1) {
        high = high.wrapping_add(1);
    }
    high
}

/// Widen a bfloat16 bit pattern to f32. Exact: bf16 *is* f32's upper half.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl Element for F32 {
    type Repr = f32;

    fn to_f32(repr: Self::Repr) -> f32 {
        repr
    }

    fn from_f32(value: f32) -> Self::Repr {
        value
    }
}

impl Element for F16 {
    type Repr = u16;

    fn to_f32(repr: Self::Repr) -> f32 {
        f16_to_f32(repr)
    }

    fn from_f32(value: f32) -> Self::Repr {
        f32_to_f16(value)
    }
}

impl Element for BF16 {
    type Repr = u16;

    fn to_f32(repr: Self::Repr) -> f32 {
        bf16_to_f32(repr)
    }

    fn from_f32(value: f32) -> Self::Repr {
        f32_to_bf16(value)
    }
}

impl<From: Element, To: Element> Executable for Cast<From, To> {
    type Operands<'a> = View<'a, From::Repr>;
    type Output = To::Repr;

    fn execute(source: View<'_, From::Repr>, output: &mut [To::Repr]) -> Result<(), TensorError> {
        // The gate runs `Cast`'s own override, so an undefined pair
        // (`f32 -> f32`, `f16 -> bf16`, an index target) is rejected before
        // any shape is compared: the type-level marker set is wider than the
        // defined-pair set, and the contract, not the type, decides.
        let out_shape = validate::<Self>(&[source.layout])?;
        check_span::<Self>(0, source.layout, source.data.len())?;
        check_output::<Self>(&out_shape, output.len())?;

        // Non-finite data is refused here too, in the *source* dtype: a NaN
        // has no cast-contract answer, so it never reaches the rounding.
        for (index, &element) in source.data.iter().enumerate() {
            if !From::to_f32(element).is_finite() {
                return Err(TensorError::NonFiniteInput {
                    op: <Self as crate::op::TensorOp>::NAME,
                    port: 0,
                    index: index as u64,
                });
            }
        }

        for (slot, &element) in output.iter_mut().zip(source.data) {
            *slot = To::from_f32(From::to_f32(element));
        }
        Ok(())
    }
}
