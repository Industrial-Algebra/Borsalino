// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Level 0 — elementwise and buffer operations (handover §3).
//!
//! These ops are bit-exact by construction: no accumulation, no reduction
//! order, no atomics. A T2 kernel for an L0 op must reproduce that — which
//! is what makes them the cheap half of the determinism gate.
//!
//! Every op here requires row-major contiguous operands ([`TensorLayout`])
//! and f32 data ports, except [`Cast`], which carries its dtype in its own
//! type, and [`Gather`]'s index port, which is `u32`.

use core::marker::PhantomData;

use crate::dtype::{CastContract, DtypeMarker};
use crate::error::TensorError;
use crate::layout::TensorLayout;
use crate::op::{DeterminismClass, OpLevel, PortSpec, ShapeRule, TensorOp};

/// Elementwise copy — identity on shape, strides, and dtype.
///
/// The only way to say "the same values, a different buffer". A copy is the
/// canonical bit-exact op: `dst[i] = src[i]`, and the kernel's
/// repeat-dispatch output must be byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy;

impl TensorOp for Copy {
    const NAME: &'static str = "copy";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Identity;
}

/// An explicit dtype conversion, `Cast<From, To>`.
///
/// The **only** place a dtype changes (handover §4.3). The pair is
/// compile-time data: `Cast<F32, F16>` is a narrowing cast because its
/// types say so, and the runtime semantics are the
/// [`CastContract`] returned by [`Cast::contract`].
///
/// Defined pairs are `f32 <-> f16` and `f32 <-> bf16`. `f32 -> f32` is
/// [`struct@Copy`], and `f16 <-> bf16` is two explicit steps through f32 — both
/// are rejected by [`CastContract::for_dtypes`] rather than silently
/// accepted. Shape and strides pass through unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cast<From: DtypeMarker, To: DtypeMarker> {
    /// The dtype pair is carried entirely by the type; no value is needed.
    _pair: PhantomData<fn() -> (From, To)>,
}

impl<From: DtypeMarker, To: DtypeMarker> Cast<From, To> {
    /// The documented semantics of this conversion.
    ///
    /// # Errors
    ///
    /// [`TensorError::CastNotDefined`] when the pair is not part of the
    /// dtype policy (identity casts, `f16 <-> bf16`, or an index dtype).
    pub fn contract() -> Result<CastContract, TensorError> {
        CastContract::for_dtypes(From::DTYPE, To::DTYPE)
    }
}

impl<From: DtypeMarker, To: DtypeMarker> TensorOp for Cast<From, To> {
    const NAME: &'static str = "cast";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::data(From::DTYPE)];
    const OUTPUT: PortSpec = PortSpec::data(To::DTYPE);
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Identity;

    /// The dtype policy is part of the input gate, not a separate question.
    ///
    /// A pair that [`Cast::contract`] refuses (`f32 -> f32`, `f16 -> bf16`, an
    /// index target) has no defined conversion, so its ports must not
    /// validate either: the contract and the gate answer with one voice, for
    /// this op and for every dtype pair in the marker set.
    fn check_inputs(inputs: &[TensorLayout]) -> Result<(), TensorError> {
        Self::contract()?;
        crate::op::check_ports::<Self>(inputs)
    }
}

/// Elementwise addition with NumPy broadcasting.
///
/// `out[i] = lhs[i] + rhs[i]`, where a dimension of 1 (or a missing leading
/// dimension) is repeated to match the other operand. Both operands are f32:
/// the op never narrows, and an f16 weight must go through [`Cast`] first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Add;

impl TensorOp for Add {
    const NAME: &'static str = "add";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Broadcast;
}

/// Elementwise multiplication with NumPy broadcasting.
///
/// `out[i] = lhs[i] * rhs[i]`, broadcast as for [`Add`]. This is the
/// scaling/masking workhorse; it is bit-exact because multiplication has no
/// summation order to lose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mul;

impl TensorOp for Mul {
    const NAME: &'static str = "mul";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Broadcast;
}

/// Elementwise division where the **divisor** broadcasts.
///
/// `out[i] = lhs[i] / rhs[broadcast(i)]` — the softmax normalization,
/// layer-scale, and mean-subtraction shape. Named for the broadcast divisor
/// rather than for elementwise symmetry, because the fleet's usage is
/// always "a tensor over a per-row or per-column scale".
///
/// The divisor must not be zero: T1 defines division by zero as producing
/// the IEEE-754 result (`inf`/`NaN`) rather than trapping, and callers who
/// need masking use [`MaskedMean`](crate::ops::l1::MaskedMean) instead.
///
/// Naming note (issue #71): [`Add`](struct@Add) and [`Mul`] broadcast too
/// — the `Broadcast` prefix here marks the DIVISOR position (the right
/// operand broadcasts against the left, never the reverse); it is not the
/// only broadcasting op, just the only one whose name must say which side
/// is fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BroadcastDiv;

impl TensorOp for BroadcastDiv {
    const NAME: &'static str = "broadcast_div";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Broadcast;
}

/// Embedding-table row lookup.
///
/// `table [V,D]` + `indices [N]` -> `[N,D]`: output row `n` is table row
/// `indices[n]`, copied verbatim. Indices are `u32` and are **computed at
/// run time**, never loaded from a checkpoint — which is why a manifest can
/// never bind this op's second port (see
/// [`bind_to_op`](crate::manifest::bind_to_op)).
///
/// Out-of-range indices are checked by T1's implementation; T0 pins the
/// dtype and the shape relation. The copy is bit-exact and order-free, so
/// gather is L0's determinism baseline for table access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gather;

impl TensorOp for Gather {
    const NAME: &'static str = "gather";
    const LEVEL: OpLevel = OpLevel::L0;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::indices()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::RowSelect;
}
