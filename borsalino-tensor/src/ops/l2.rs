// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Level 2 — GEMM (handover §3).
//!
//! The gate for this class is DeepReinforce exact-match against the CPU
//! oracle, plus `determinism.rs` repeat-dispatch bit-for-bit equality. Both
//! hold only if the accumulation order is fixed, so "fixed tiling order" is
//! part of the contract: a T2 kernel that reorders its K-loop by workgroup
//! size still owes the same bit pattern on every dispatch of the same shape.
//!
//! Adversarial shape classes this class must survive (handover §5, and the
//! kani harness classes): subnormal-magnitude inputs, non-power-of-2
//! dimensions, alignment-boundary buffer offsets, and empty batch/head dims.

use crate::op::{DeterminismClass, OpLevel, PortSpec, ShapeRule, TensorOp};

/// 2-D matrix multiplication: `[M,K] x [K,N]` -> `[M,N]`.
///
/// The contracted K dimensions must agree exactly — there is no broadcasting
/// on the reduction axis, because a broadcast inner dimension has no
/// meaning. Operands are f32 and row-major contiguous, so the kernel reads
/// `lhs[m*K + k]` and `rhs[k*N + n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMul;

impl TensorOp for MatMul {
    const NAME: &'static str = "matmul";
    const LEVEL: OpLevel = OpLevel::L2;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderGemm;
    const SHAPE_RULE: ShapeRule = ShapeRule::MatMul;
}

/// Batched 4-D matrix multiplication: `[B,H,S,D] x [B,H,D,T]` ->
/// `[B,H,S,T]`.
///
/// The attention-projection shape: `B` batches, `H` heads, and an
/// independent `S x D` by `D x T` GEMM inside each. Batch and head
/// dimensions must agree exactly (heads are not broadcast against each
/// other), and any of `B`, `H`, `S`, `D`, `T` may be zero — an empty batch
/// or head is a legal shape whose output is empty, not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedMatMul;

impl TensorOp for BatchedMatMul {
    const NAME: &'static str = "batched_matmul";
    const LEVEL: OpLevel = OpLevel::L2;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32(), PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::FixedOrderGemm;
    const SHAPE_RULE: ShapeRule = ShapeRule::BatchedMatMul;
}

/// 2-D transpose, `[R, C] -> [C, R]` (issue #71).
///
/// Katakuri's canonical weight layout is `[out, in]`; `matmul` contracts
/// `[M, K] x [K, N]`, so a weight consumed as the right operand needs its
/// axes swapped. Declaring the swap as an op (rather than every consumer
/// hand-rolling a strided view) gives it a contract, a CPU reference, and
/// — at T2 — a kernel with the same semantics. A pure permutation:
/// [`DeterminismClass::BitExact`] by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transpose2D;

impl TensorOp for Transpose2D {
    const NAME: &'static str = "transpose_2d";
    const LEVEL: OpLevel = OpLevel::L2;
    const INPUTS: &'static [PortSpec] = &[PortSpec::f32()];
    const OUTPUT: PortSpec = PortSpec::f32();
    const DETERMINISM: DeterminismClass = DeterminismClass::BitExact;
    const SHAPE_RULE: ShapeRule = ShapeRule::Transpose2D;
}
