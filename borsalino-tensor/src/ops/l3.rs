// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! L3 — activation and composition ops (issue #71).
//!
//! The handover's L3 starts with composition (layernorm, softmax,
//! attention) and the fused 4-D mask; those need the transcendental
//! tolerance classes and land after this module's start. What ships now
//! is the first activation named by Katakuri's `Activation` enum —
//! `relu` — so `Activation::Relu` maps onto a contracted op instead of
//! per-consumer arithmetic. `tanh`/`gelu_*` follow with the tolerance
//! machinery.

use crate::op::{DeterminismClass, OpLevel, PortSpec, ShapeRule, TensorOp};

/// Rectified linear unit, `relu(x) = max(x, 0)` with a fully
/// deterministic definition:
///
/// - positive inputs (including subnormals) pass through bit-identically,
/// - everything else — zero, `-0.0`, negatives, NaN — maps to `+0.0`.
///
/// The comparison form (`if x > 0.0 { x } else { 0.0 }`) is deliberate:
/// `f32::max` may return either operand for `-0.0` vs `+0.0`, which would
/// not be bit-exact. Elementwise and order-free, so a kernel is
/// [`DeterminismClass::BitExact`] by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relu;

impl TensorOp for Relu {
    const NAME: &'static str = "relu";
    const LEVEL: OpLevel = OpLevel::L3;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Identity;
}
