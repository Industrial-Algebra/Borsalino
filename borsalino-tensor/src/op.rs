// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The op contract: what an operation *is*, before any kernel exists.
//!
//! Every op in the T0 inventory is a type implementing [`TensorOp`]. The
//! trait is deliberately declarative — there is no `execute`, no `dispatch`,
//! no buffer. At T0 an op is:
//!
//! - its **name** and **level** (the handover §3 inventory, L0–L2),
//! - its **ports**: dtype and role for each input and for the output,
//! - its **determinism class**, the gate the kernel must pass in T2,
//! - its **shape rule**, which infers the output shape from the inputs and
//!   rejects an ill-formed call.
//!
//! That is the whole contract a kernel author in T1/T2 codes against, and it
//! is why the inventory can be reviewed (and pinned by tests) before a
//! single WGSL line exists.
//!
//! Explicitly not here (handover §3, NOT-scope): autodiff, training,
//! optimizers, gradients, model specs (Katakuri's half), f64 (Amari's half),
//! graphics.

use crate::dtype::Dtype;
use crate::error::TensorError;
use crate::layout::{Shape, TensorLayout};

/// Which layer of the handover §3 inventory an op belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpLevel {
    /// Elementwise and buffer ops: copy, cast, add, mul, broadcast_div,
    /// gather.
    L0,
    /// Reductions: sum, attention-masked mean, rowwise L2 norm, max.
    ///
    /// This is the determinism-sensitive class: a reduction's summation
    /// order is part of its contract, so the kernel must use a fixed or
    /// tree order proven by `determinism.rs` (handover §3).
    L1,
    /// GEMM: matmul (2D) and batched matmul `[B,H,S,D]`.
    ///
    /// Gated by DeepReinforce exact-match against the CPU oracle, on
    /// adversarial shape classes (subnormal spreads, non-power-of-2 dims,
    /// alignment-boundary offsets, empty batch/head dims).
    L2,
}

/// The determinism a kernel must prove (handover §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeterminismClass {
    /// Bit-identical by construction: no accumulation, no atomics, no
    /// reduction order to get wrong.
    BitExact,
    /// Requires a fixed-order or tree reduction. Unordered atomics are
    /// forbidden: they fail repeat-dispatch bit-for-bit checks.
    FixedOrderReduction,
    /// Requires a fixed tiling order for the accumulation chain, so the
    /// same inputs produce the same bit pattern on every dispatch.
    FixedOrderGemm,
}

/// What a port carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortRole {
    /// Element data — the contract dtype (f32) for every op at T0.
    Data,
    /// Indices into another operand's leading dimension. Always `u32`, never
    /// loaded from a checkpoint (they are computed at run time), therefore
    /// never bindable from a manifest.
    Indices,
}

/// One operand port: its dtype and its role.
///
/// Ports are `const` data on the op type, so a consumer can inspect an op
/// without instantiating anything and the manifest binder can validate a
/// checkpoint without running a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSpec {
    /// The dtype the port requires.
    pub dtype: Dtype,
    /// What the port carries.
    pub role: PortRole,
}

impl PortSpec {
    /// A data port of the given dtype.
    #[must_use]
    pub const fn data(dtype: Dtype) -> Self {
        Self {
            dtype,
            role: PortRole::Data,
        }
    }

    /// A data port of the contract dtype (f32).
    #[must_use]
    pub const fn f32() -> Self {
        Self::data(Dtype::F32)
    }

    /// An index port (`u32`).
    #[must_use]
    pub const fn indices() -> Self {
        Self {
            dtype: Dtype::U32,
            role: PortRole::Indices,
        }
    }
}

/// How an op derives its output shape from its inputs.
///
/// The rule owns the inference, so an op cannot declare one shape relation
/// and check another: [`TensorOp::infer_shape`] and [`TensorOp::check_inputs`]
/// both delegate here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeRule {
    /// Output is input 0 unchanged (`copy`, `cast`).
    Identity,
    /// NumPy-style broadcasting over two operands (`add`, `mul`,
    /// `broadcast_div`).
    Broadcast,
    /// Table row lookup: `[V,D]` table + `[N]` indices -> `[N,D]`
    /// (`gather`).
    RowSelect,
    /// Full reduction to the rank-0 scalar (`sum`, `max`).
    FullReduction,
    /// Reduction over the trailing axis: `[R,D]` -> `[R]`
    /// (`rowwise_l2_norm`).
    RowReduction,
    /// Attention-masked pooling over the token axis: `[B,S,D]` values +
    /// `[B,S]` mask -> `[B,D]` (`masked_mean`).
    MaskedPool,
    /// 2-D GEMM: `[M,K] x [K,N]` -> `[M,N]` (`matmul`).
    MatMul,
    /// Batched 4-D GEMM: `[B,H,S,D] x [B,H,D,T]` -> `[B,H,S,T]`
    /// (`batched_matmul`).
    BatchedMatMul,
}

impl ShapeRule {
    /// The number of operands the rule reads.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            ShapeRule::Identity | ShapeRule::FullReduction | ShapeRule::RowReduction => 1,
            ShapeRule::Broadcast
            | ShapeRule::RowSelect
            | ShapeRule::MaskedPool
            | ShapeRule::MatMul
            | ShapeRule::BatchedMatMul => 2,
        }
    }

    /// Infer the output shape, rejecting ill-formed operand shapes.
    ///
    /// # Errors
    ///
    /// The variant that names the violation: arity, rank, broadcast
    /// incompatibility, disagreeing inner or batch dimensions, or a shape
    /// whose element count overflows.
    pub fn infer(self, op: &'static str, inputs: &[&Shape]) -> Result<Shape, TensorError> {
        if inputs.len() != self.arity() {
            return Err(TensorError::ArityMismatch {
                op,
                expected: self.arity(),
                got: inputs.len(),
            });
        }
        match self {
            ShapeRule::Identity => Ok(inputs[0].clone()),
            ShapeRule::Broadcast => broadcast(op, inputs[0], inputs[1]),
            ShapeRule::RowSelect => row_select(op, inputs[0], inputs[1]),
            ShapeRule::FullReduction => Shape::new(&[]),
            ShapeRule::RowReduction => row_reduction(op, inputs[0]),
            ShapeRule::MaskedPool => masked_pool(op, inputs[0], inputs[1]),
            ShapeRule::MatMul => matmul(op, inputs[0], inputs[1]),
            ShapeRule::BatchedMatMul => batched_matmul(op, inputs[0], inputs[1]),
        }
    }

    /// Infer the output shape and check it against a declared one.
    ///
    /// # Errors
    ///
    /// Whatever [`infer`](Self::infer) rejects, plus
    /// [`TensorError::OutputShapeMismatch`] when the declared output shape is
    /// not the inferred one.
    pub fn check(
        self,
        op: &'static str,
        inputs: &[&Shape],
        output: &Shape,
    ) -> Result<(), TensorError> {
        let inferred = self.infer(op, inputs)?;
        if inferred == *output {
            Ok(())
        } else {
            Err(TensorError::OutputShapeMismatch {
                op,
                expected: inferred.dims().to_vec(),
                got: output.dims().to_vec(),
            })
        }
    }
}

/// A tensor operation from the T0 inventory.
///
/// Implementors are zero-sized types (or, for [`Cast`](crate::ops::l0::Cast),
/// dtype markers): the type *is* the op, and every property a kernel needs is
/// an associated const, inspectable without constructing anything.
pub trait TensorOp {
    /// The op's contract name, as it appears in errors and manifests.
    const NAME: &'static str;
    /// The handover §3 level the op belongs to.
    const LEVEL: OpLevel;
    /// The op's input ports, in argument order.
    const INPUTS: &'static [PortSpec];
    /// The op's output port.
    const OUTPUT: PortSpec;
    /// The determinism class a T2 kernel must prove for this op.
    const DETERMINISM: DeterminismClass;
    /// The op's output-shape relation.
    const SHAPE_RULE: ShapeRule;

    /// Infer the output shape from the operand shapes.
    ///
    /// # Errors
    ///
    /// Whatever the op's [`ShapeRule`] rejects.
    fn infer_shape(inputs: &[&Shape]) -> Result<Shape, TensorError> {
        Self::SHAPE_RULE.infer(Self::NAME, inputs)
    }

    /// Check a call's operands against arity, dtype, and shape rule.
    ///
    /// This is the gate every T1/T2 entry point runs first: it keeps f16 out
    /// of f32 arithmetic (no silent narrowing), keeps index ports on `u32`,
    /// and rejects shapes the op cannot honor — before any buffer is bound.
    ///
    /// # Errors
    ///
    /// [`TensorError::ArityMismatch`], [`TensorError::DtypeMismatch`], or
    /// whatever the op's [`ShapeRule`] rejects.
    fn check_inputs(inputs: &[TensorLayout]) -> Result<(), TensorError> {
        check_ports::<Self>(inputs)
    }
}

/// The shared operand gate: arity, per-port dtype, then the op's shape rule.
///
/// [`TensorOp::check_inputs`] delegates here so an op that overrides it (to
/// enforce an extra policy, as [`Cast`](crate::ops::l0::Cast) does) runs
/// exactly the same checks as every other op instead of re-implementing them.
///
/// # Errors
///
/// [`TensorError::ArityMismatch`], [`TensorError::DtypeMismatch`], or
/// whatever the op's [`ShapeRule`] rejects.
pub(crate) fn check_ports<O: TensorOp + ?Sized>(
    inputs: &[TensorLayout],
) -> Result<(), TensorError> {
    if inputs.len() != O::INPUTS.len() {
        return Err(TensorError::ArityMismatch {
            op: O::NAME,
            expected: O::INPUTS.len(),
            got: inputs.len(),
        });
    }
    for (port, (spec, layout)) in O::INPUTS.iter().zip(inputs).enumerate() {
        if layout.dtype() != spec.dtype {
            return Err(TensorError::DtypeMismatch {
                op: O::NAME,
                port,
                expected: spec.dtype,
                got: layout.dtype(),
            });
        }
    }
    let shapes: Vec<&Shape> = inputs.iter().map(TensorLayout::shape).collect();
    O::SHAPE_RULE.infer(O::NAME, &shapes).map(|_| ())
}

/// The dimension at `axis`, counting from the trailing end, or `1` when the
/// operand does not reach that far — the alignment rule broadcasting needs.
fn aligned_dim(shape: &Shape, rank: usize, axis: usize) -> u32 {
    let offset = rank - shape.rank();
    if axis < offset {
        1
    } else {
        shape.dims()[axis - offset]
    }
}

/// NumPy broadcasting: align on the trailing dimension, then require equal
/// dimensions or a `1`.
fn broadcast(op: &'static str, lhs: &Shape, rhs: &Shape) -> Result<Shape, TensorError> {
    let rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left = aligned_dim(lhs, rank, axis);
        let right = aligned_dim(rhs, rank, axis);
        let dim = if left == right {
            left
        } else if left == 1 {
            right
        } else if right == 1 {
            left
        } else {
            return Err(TensorError::BroadcastIncompatible {
                op,
                lhs: lhs.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        };
        dims.push(dim);
    }
    Shape::new(&dims)
}

/// `[V,D]` table + `[N]` indices -> `[N,D]`: embedding-table row lookup.
fn row_select(op: &'static str, table: &Shape, indices: &Shape) -> Result<Shape, TensorError> {
    expect_rank(op, 0, table, 2)?;
    expect_rank(op, 1, indices, 1)?;
    Shape::new(&[indices.dims()[0], table.dims()[1]])
}

/// `[R,D]` -> `[R]`: the trailing axis is the one summed, squared, or
/// maximized over.
fn row_reduction(op: &'static str, values: &Shape) -> Result<Shape, TensorError> {
    expect_rank(op, 0, values, 2)?;
    Shape::new(&[values.dims()[0]])
}

/// `[B,S,D]` values + `[B,S]` mask -> `[B,D]`.
///
/// The mask must be rank 2 (batch and token axes), its batch must equal the
/// values' batch, and its token axis must match the values' token axis. An
/// all-zero mask row is legal at the shape level; the T1 op contract defines
/// its mean as zero rather than NaN.
fn masked_pool(op: &'static str, values: &Shape, mask: &Shape) -> Result<Shape, TensorError> {
    expect_rank(op, 0, values, 3)?;
    expect_rank(op, 1, mask, 2)?;
    if mask.dims()[0] != values.dims()[0] {
        return Err(TensorError::BatchDimensionMismatch {
            op,
            lhs: vec![values.dims()[0]],
            rhs: vec![mask.dims()[0]],
        });
    }
    if mask.dims()[1] != values.dims()[1] {
        return Err(TensorError::BroadcastIncompatible {
            op,
            lhs: values.dims().to_vec(),
            rhs: mask.dims().to_vec(),
        });
    }
    Shape::new(&[values.dims()[0], values.dims()[2]])
}

/// `[M,K] x [K,N]` -> `[M,N]`.
fn matmul(op: &'static str, lhs: &Shape, rhs: &Shape) -> Result<Shape, TensorError> {
    expect_rank(op, 0, lhs, 2)?;
    expect_rank(op, 1, rhs, 2)?;
    contract(op, lhs.dims()[1], rhs.dims()[0])?;
    Shape::new(&[lhs.dims()[0], rhs.dims()[1]])
}

/// `[B,H,S,D] x [B,H,D,T]` -> `[B,H,S,T]`.
///
/// Batch and head dimensions must agree exactly (no broadcast: attention
/// heads are not broadcast against each other), and empty batch/head dims
/// are legal — they are an adversarial shape class, not an invalidity
/// (handover §5).
fn batched_matmul(op: &'static str, lhs: &Shape, rhs: &Shape) -> Result<Shape, TensorError> {
    expect_rank(op, 0, lhs, 4)?;
    expect_rank(op, 1, rhs, 4)?;
    if lhs.dims()[..2] != rhs.dims()[..2] {
        return Err(TensorError::BatchDimensionMismatch {
            op,
            lhs: lhs.dims()[..2].to_vec(),
            rhs: rhs.dims()[..2].to_vec(),
        });
    }
    contract(op, lhs.dims()[3], rhs.dims()[2])?;
    Shape::new(&[lhs.dims()[0], lhs.dims()[1], lhs.dims()[2], rhs.dims()[3]])
}

/// Check that an operand has the rank its port requires.
fn expect_rank(
    op: &'static str,
    port: usize,
    shape: &Shape,
    expected: usize,
) -> Result<(), TensorError> {
    if shape.rank() == expected {
        Ok(())
    } else {
        Err(TensorError::RankMismatch {
            op,
            port,
            expected: match expected {
                1 => "rank 1",
                2 => "rank 2",
                3 => "rank 3",
                4 => "rank 4",
                _ => "rank",
            },
            got: shape.rank(),
        })
    }
}

/// Check that a GEMM's contracted dimensions agree.
fn contract(op: &'static str, lhs: u32, rhs: u32) -> Result<(), TensorError> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(TensorError::InnerDimensionMismatch { op, lhs, rhs })
    }
}
