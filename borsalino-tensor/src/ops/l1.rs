// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Level 1 — reductions (handover §3).
//!
//! Reductions are the determinism-sensitive class: a floating-point sum
//! depends on its order, so the order is part of these ops' contracts. Every
//! L1 kernel must use a fixed or tree order and pass
//! `determinism.rs`'s repeat-dispatch bit-for-bit check — unordered atomics
//! are forbidden, not merely discouraged.
//!
//! All L1 ops take f32 and produce f32.

use crate::op::{DeterminismClass, OpLevel, PortSpec, ShapeRule, TensorOp};

/// Full sum of every element: any shape -> rank-0 scalar.
///
/// Requires a fixed reduction order (the T2 implementation's tree shape is
/// part of the contract, not an optimization detail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sum;

impl TensorOp for Sum {
    const NAME: &'static str = "sum";
    const LEVEL: OpLevel = OpLevel::L1;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderReduction;
    const SHAPE_RULE: ShapeRule = ShapeRule::FullReduction;
}

/// Attention-masked mean over the token axis.
///
/// `values [B,S,D]` + `mask [B,S]` -> `[B,D]`: the pooling step of a
/// sentence-embedding head, where pad tokens must not contribute. The mask
/// is f32 data (not a boolean dtype) so it can also act as a weight.
///
/// Fixed-order reduction, and the mask's summation is part of the same
/// order: an all-zero mask row is a legal shape, whose mean T1 defines as
/// zero rather than `NaN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskedMean;

impl TensorOp for MaskedMean {
    const NAME: &'static str = "masked_mean";
    const LEVEL: OpLevel = OpLevel::L1;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderReduction;
    const SHAPE_RULE: ShapeRule = ShapeRule::MaskedPool;
}

/// Rowwise L2 norm: `[R,D]` -> `[R]`.
///
/// `out[r] = sqrt(sum_d values[r,d]^2)`. The sum of squares is a
/// fixed-order reduction; `sqrt` is a single IEEE-754 operation on the
/// reduced value, so the whole op stays deterministic.
///
/// Used for embedding normalization, where the precision of the reduction
/// order shows up directly in downstream cosine similarity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowwiseL2Norm;

impl TensorOp for RowwiseL2Norm {
    const NAME: &'static str = "rowwise_l2_norm";
    const LEVEL: OpLevel = OpLevel::L1;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderReduction;
    const SHAPE_RULE: ShapeRule = ShapeRule::RowReduction;
}

/// Full maximum of every element: any shape -> rank-0 scalar.
///
/// Max is order-insensitive in the reals but not in the presence of `NaN`
/// or signed zeros: the T2 kernel must still declare its order so the
/// tie-breaking (which element wins, and whether `-0.0` or `0.0` is
/// returned) is replayable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Max;

impl TensorOp for Max {
    const NAME: &'static str = "max";
    const LEVEL: OpLevel = OpLevel::L1;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderReduction;
    const SHAPE_RULE: ShapeRule = ShapeRule::FullReduction;
}
