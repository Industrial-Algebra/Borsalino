// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T1 CPU reference — L3 activations (issue #71).
//!
//! `relu`'s contract is the comparison form `if x > 0.0 { x } else { 0.0 }`
//! (NOT `f32::max`, whose `-0.0` vs `+0.0` result is unspecified): every
//! fixture here is written in that form, independently of the
//! implementation under test.

mod common;

use borsalino_tensor::error::TensorError;
use borsalino_tensor::exec::Executable;
use borsalino_tensor::ops::l3::Relu;
use common::{SENTINEL, bits, f32_layout, view};

/// The contract's exact arithmetic, written independently.
fn reference_relu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| if *value > 0.0 { *value } else { 0.0 })
        .collect()
}

#[test]
fn relu_matches_the_comparison_form_on_adversarial_values() {
    // Subnormals, -0.0/+0.0, the finite extremes, and negatives — the
    // classes where `f32::max` and the comparison form would disagree
    // on bits (`max` may pick either zero).
    let input = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1), // smallest subnormal
        -f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        1e-40,
    ];
    let layout = f32_layout(&[input.len() as u32]);
    let mut output = vec![SENTINEL; input.len()];
    Relu::execute(view(&layout, &input), &mut output).unwrap();
    let expected = reference_relu(&input);
    assert_eq!(bits(&output), bits(&expected));
    // -0.0 must come out +0.0 (bit pattern 0), not -0.0.
    assert_eq!(output[3].to_bits(), 0);
    // Subnormal positives pass through bit-identically.
    assert_eq!(output[5].to_bits(), f32::from_bits(1).to_bits());
}

#[test]
fn relu_rejects_nonfinite_inputs() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let layout = f32_layout(&[2]);
        let mut output = vec![SENTINEL; 2];
        let input = [1.0, bad];
        match Relu::execute(view(&layout, &input), &mut output) {
            Err(TensorError::NonFiniteInput { .. }) => {}
            other => panic!("non-finite {bad} must be rejected, got {other:?}"),
        }
    }
}

#[test]
fn relu_passes_shapes_through_unchanged() {
    let layout = f32_layout(&[2, 3]);
    let input = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let mut output = vec![SENTINEL; 6];
    Relu::execute(view(&layout, &input), &mut output).unwrap();
    assert_eq!(bits(&output), bits(&reference_relu(&input)));
}

#[test]
fn relu_writes_nothing_on_empty_shapes() {
    let layout = f32_layout(&[0, 4]);
    let input: [f32; 0] = [];
    let mut output: Vec<f32> = vec![];
    Relu::execute(view(&layout, &input), &mut output).unwrap();
    assert!(output.is_empty());
}
