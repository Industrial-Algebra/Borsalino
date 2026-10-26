// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 — the T0 cast contract, executed.
//!
//! T0 pinned `CastContract` as data: the overflow midpoint, the subnormal
//! policy, the roundtrip class. This file turns every one of those claims
//! into an assertion against the real conversion, so the contract cannot
//! drift from the arithmetic. The boundary values are the ones T0's review
//! settled on (65504 / 65505 / 65520, the bf16 midpoint, the subnormal
//! floors), plus a full-pattern roundtrip of both narrow dtypes.

mod common;

use borsalino_tensor::dtype::{BF16, CastContract, Dtype, F16, F32};
use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::{Executable, View};
use borsalino_tensor::ops::l0::Cast;
use common::{f32_layout, layout};

/// 2^`exp` as an exact f64, constructed from exponent bits.
fn pow2(exp: i32) -> f64 {
    f64::from_bits(((1023 + exp) as u64) << 52)
}

fn to_f16(values: &[f32]) -> Result<Vec<u16>, TensorError> {
    let source = f32_layout(&[values.len() as u32]);
    let mut output = vec![0u16; values.len()];
    <Cast<F32, F16> as Executable>::execute(View::new(&source, values), &mut output)?;
    Ok(output)
}

fn to_bf16(values: &[f32]) -> Result<Vec<u16>, TensorError> {
    let source = f32_layout(&[values.len() as u32]);
    let mut output = vec![0u16; values.len()];
    <Cast<F32, BF16> as Executable>::execute(View::new(&source, values), &mut output)?;
    Ok(output)
}

fn f16_to_f32(bits: &[u16]) -> Result<Vec<f32>, TensorError> {
    let source = layout(Dtype::F16, &[bits.len() as u32]);
    let mut output = vec![0.0f32; bits.len()];
    <Cast<F16, F32> as Executable>::execute(View::new(&source, bits), &mut output)?;
    Ok(output)
}

fn bf16_to_f32(bits: &[u16]) -> Result<Vec<f32>, TensorError> {
    let source = layout(Dtype::BF16, &[bits.len() as u32]);
    let mut output = vec![0.0f32; bits.len()];
    <Cast<BF16, F32> as Executable>::execute(View::new(&source, bits), &mut output)?;
    Ok(output)
}

#[test]
fn f16_overflows_at_the_midpoint_not_at_the_maximum() {
    let contract = CastContract::for_dtypes(Dtype::F32, Dtype::F16).unwrap();
    assert_eq!(contract.max_finite, 65504.0);
    assert_eq!(contract.overflow_threshold, 65520.0);

    // At the maximum, and anywhere below the midpoint, the value rounds back
    // down to the maximum finite value.
    assert_eq!(to_f16(&[65504.0]).unwrap(), vec![0x7bff]);
    assert_eq!(to_f16(&[65505.0]).unwrap(), vec![0x7bff]);
    assert_eq!(to_f16(&[65519.0]).unwrap(), vec![0x7bff]);
    assert_eq!(to_f16(&[-65504.0]).unwrap(), vec![0xfbff]);
    assert_eq!(to_f16(&[-65505.0]).unwrap(), vec![0xfbff]);

    // At the midpoint and beyond, it is infinity — the carry out of the
    // significand renormalizes into the exponent field.
    assert_eq!(to_f16(&[65520.0]).unwrap(), vec![0x7c00]);
    assert_eq!(to_f16(&[1.0e30]).unwrap(), vec![0x7c00]);
    assert_eq!(to_f16(&[f32::MAX]).unwrap(), vec![0x7c00]);
    assert_eq!(to_f16(&[-65520.0]).unwrap(), vec![0xfc00]);
}

#[test]
fn f16_boundary_values_match_the_ieee_754_encodings() {
    let cases: [(f32, u16); 12] = [
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (0.5, 0x3800),
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0 / 3.0, 0x3555),
        // The minimum normal, the minimum subnormal, half of it (a tie to
        // even, hence zero), and three halves of it (rounds up to two).
        (pow2(-14) as f32, 0x0400),
        (pow2(-24) as f32, 0x0001),
        (pow2(-25) as f32, 0x0000),
        (3.0 * pow2(-25) as f32, 0x0002),
        (-(pow2(-24) as f32), 0x8001),
        (65504.0, 0x7bff),
    ];

    for (value, bits) in cases {
        assert_eq!(
            to_f16(&[value]).unwrap(),
            vec![bits],
            "{value} should encode as {bits:#06x}"
        );
    }
}

#[test]
fn every_f32_subnormal_flushes_to_zero_in_f16() {
    let contract = CastContract::for_dtypes(Dtype::F32, Dtype::F16).unwrap();

    // The executable form of the T0 sentence: f16's minimum subnormal sits
    // 102 binades above f32's largest subnormal, so all of them are below
    // half of it.
    let largest_f32_subnormal = f64::from(f32::from_bits(0x007f_ffff));
    assert!(
        largest_f32_subnormal < contract.min_subnormal / 2.0,
        "the flush-to-zero claim depends on this inequality"
    );

    let subnormals = [
        f32::from_bits(1),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x0000_ffff),
        f32::from_bits(0x007f_ffff),
        f32::from_bits(0x8000_0001),
    ];
    assert_eq!(
        to_f16(&subnormals).unwrap(),
        vec![0x0000, 0x0000, 0x0000, 0x0000, 0x8000],
        "f32 subnormals flush to signed zero, never to a wrong small number"
    );

    // The smallest f16 subnormal is reachable from a *normal* f32.
    assert_eq!(to_f16(&[pow2(-24) as f32]).unwrap(), vec![0x0001]);
}

#[test]
fn f16_values_below_the_ramp_round_to_zero() {
    // Half of the minimum subnormal is the last value that rounds to zero
    // (ties to even), and just above it rounds to one ulp.
    let half = pow2(-25) as f32;
    let above = f32::from_bits(half.to_bits() + 1);
    assert_eq!(to_f16(&[half]).unwrap(), vec![0x0000]);
    assert_eq!(to_f16(&[above]).unwrap(), vec![0x0001]);
}

#[test]
fn bf16_overflows_at_its_own_midpoint() {
    let contract = CastContract::for_dtypes(Dtype::F32, Dtype::BF16).unwrap();
    // (2 - 2^-7) * 2^127 — smaller than f32's maximum, which is why a
    // narrowing to bf16 *can* overflow where a widening cannot.
    let max_finite = 255.0 * pow2(120);
    assert_eq!(contract.max_finite, max_finite);
    assert!(max_finite < f64::from(f32::MAX));
    assert_eq!(contract.overflow_threshold, (max_finite + pow2(128)) / 2.0);

    let max_finite = max_finite as f32;
    // f32's ulp in that binade, and the midpoint to the next power of two.
    let ulp = pow2(104) as f32;
    let midpoint_offset = pow2(119) as f32;

    assert_eq!(to_bf16(&[max_finite]).unwrap(), vec![0x7f7f]);
    assert_eq!(to_bf16(&[max_finite + ulp]).unwrap(), vec![0x7f7f]);
    assert_eq!(
        to_bf16(&[max_finite + (midpoint_offset - ulp)]).unwrap(),
        vec![0x7f7f],
        "just below the midpoint rounds back to the maximum finite value"
    );
    assert_eq!(
        to_bf16(&[max_finite + midpoint_offset]).unwrap(),
        vec![0x7f80],
        "exactly at the midpoint the tie rounds up, which is infinity"
    );
    assert_eq!(to_bf16(&[f32::MAX]).unwrap(), vec![0x7f80]);
}

#[test]
fn bf16_reaches_its_own_subnormal_floor() {
    let contract = CastContract::for_dtypes(Dtype::F32, Dtype::BF16).unwrap();
    assert_eq!(contract.min_subnormal, pow2(-133));

    // Every f32 subnormal down to 2^-133 survives; below that the ramp
    // rounds to zero (ties to even), and three halves round up.
    let smallest = pow2(-133) as f32;
    let below = pow2(-134) as f32;
    assert_eq!(to_bf16(&[smallest]).unwrap(), vec![0x0001]);
    assert_eq!(to_bf16(&[below]).unwrap(), vec![0x0000]);
    assert_eq!(to_bf16(&[3.0 * below]).unwrap(), vec![0x0002]);
    assert_eq!(to_bf16(&[2.0 * smallest]).unwrap(), vec![0x0002]);
    assert_eq!(to_bf16(&[-smallest]).unwrap(), vec![0x8001]);
}

#[test]
fn bf16_rounds_ties_to_even() {
    let cases: [(f32, u16); 7] = [
        (1.0, 0x3f80),
        (2.0, 0x4000),
        (0.5, 0x3f00),
        (std::f32::consts::PI, 0x4049),
        // 1 + 2^-8 is an exact tie with an even significand: stays.
        (1.0 + pow2(-8) as f32, 0x3f80),
        // 1 + 3*2^-8 is an exact tie with an odd significand: rounds up.
        (1.0 + 3.0 * pow2(-8) as f32, 0x3f82),
        // Just above a tie rounds up.
        (
            f32::from_bits((1.0 + pow2(-8) as f32).to_bits() + 1),
            0x3f81,
        ),
    ];

    for (value, bits) in cases {
        assert_eq!(
            to_bf16(&[value]).unwrap(),
            vec![bits],
            "{value} should encode as {bits:#06x}"
        );
    }
}

#[test]
fn widening_casts_are_exact() {
    // f16 -> f32, including its subnormals.
    assert_eq!(
        f16_to_f32(&[0x3c00, 0x0001, 0x7bff, 0x8001, 0x0400]).unwrap(),
        vec![
            1.0,
            pow2(-24) as f32,
            65504.0,
            -(pow2(-24) as f32),
            pow2(-14) as f32
        ]
    );
    // bf16 -> f32 is a bit shift: exact by construction.
    assert_eq!(
        bf16_to_f32(&[0x3f80, 0x0001, 0x7f7f, 0x8000]).unwrap(),
        vec![1.0, pow2(-133) as f32, 255.0 * pow2(120) as f32, -0.0]
    );
}

#[test]
fn every_f16_pattern_roundtrips_through_f32() {
    // The source gate reads the *source* dtype, so a non-finite pattern is
    // rejected on the way in; the classification below is structural (the
    // exponent field is all ones) rather than read back out of the op.
    let mut finite = Vec::new();
    let mut widened = Vec::new();
    let mut non_finite = 0usize;
    for pattern in 0u16..=u16::MAX {
        if (pattern >> 10) & 0x1f == 0x1f {
            non_finite += 1;
            assert!(
                matches!(
                    f16_to_f32(&[pattern]),
                    Err(TensorError::NonFiniteInput { port: 0, .. })
                ),
                "pattern {pattern:#06x} is non-finite and must be rejected"
            );
            continue;
        }
        let value = f16_to_f32(&[pattern]).expect("finite patterns widen")[0];
        assert!(value.is_finite());
        finite.push(pattern);
        widened.push(value);
    }

    assert_eq!(finite.len() + non_finite, 65_536);
    // One all-ones exponent field: 2^10 significands (infinity included)
    // under both signs.
    assert_eq!(non_finite, 2 * 1024);

    assert_eq!(
        to_f16(&widened).unwrap(),
        finite,
        "widening is exact, so every finite pattern must narrow back to itself"
    );
}

#[test]
fn every_bf16_pattern_roundtrips_through_f32() {
    let mut finite = Vec::new();
    let mut widened = Vec::new();
    let mut non_finite = 0usize;
    for pattern in 0u16..=u16::MAX {
        if (pattern >> 7) & 0xff == 0xff {
            non_finite += 1;
            assert!(matches!(
                bf16_to_f32(&[pattern]),
                Err(TensorError::NonFiniteInput { .. })
            ));
            continue;
        }
        let value = bf16_to_f32(&[pattern]).expect("finite patterns widen")[0];
        assert!(value.is_finite());
        finite.push(pattern);
        widened.push(value);
    }

    assert_eq!(non_finite, 2 * 128);
    assert_eq!(to_bf16(&widened).unwrap(), finite);
}

#[test]
fn non_finite_values_are_rejected_in_both_directions() {
    // The gate is uniform: a non-finite value never enters a cast, whichever
    // side it is on. Only a *finite* input may produce a non-finite output
    // (the narrowing overflow the contract documents), so no fixture here
    // relies on carrying an infinity through.
    for pattern in [0x7c00u16, 0xfc00, 0x7e00] {
        let error = f16_to_f32(&[pattern]);
        assert!(
            matches!(error, Err(TensorError::NonFiniteInput { port: 0, .. })),
            "f16 pattern {pattern:#06x} is rejected: {error:?}"
        );
    }
    for pattern in [0x7f80u16, 0xff80, 0x7fc0] {
        let error = bf16_to_f32(&[pattern]);
        assert!(matches!(error, Err(TensorError::NonFiniteInput { .. })));
    }

    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(matches!(
            to_f16(&[value]),
            Err(TensorError::NonFiniteInput {
                op: "cast",
                port: 0,
                index: 0,
            })
        ));
        assert!(matches!(
            to_bf16(&[value]),
            Err(TensorError::NonFiniteInput { .. })
        ));
    }
}
