// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The T0 error taxonomy.
//!
//! Every way a tensor operation can be rejected at the contract boundary
//! has exactly one variant here, and each variant names the operation, the
//! port, or the manifest entry that failed. A contract that reports "invalid
//! argument" is a contract no kernel author can implement against, so the
//! variants carry the data a caller needs to fix the call site.
//!
//! The taxonomy is deliberately *shallow*: it describes shapes, strides,
//! dtypes, and byte spans. It says nothing about device loss, queue
//! submission, or dispatch failure — those belong to the substrate
//! (Borsalino/Zunesha), not to the tensor layer.

use crate::dtype::Dtype;

/// A violation of the tensor op-set contract.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    /// A shape's element count does not fit in a `u64`.
    ///
    /// Raised by [`Shape::new`](crate::layout::Shape::new) before any
    /// arithmetic can overflow downstream: a shape that cannot be counted
    /// cannot be allocated, bound, or dispatched.
    #[error("shape {shape:?} has an element count that overflows u64")]
    ShapeOverflow {
        /// The rejected dimensions, in row-major order.
        shape: Vec<u32>,
    },

    /// A layout's byte footprint does not fit in a `u64`.
    ///
    /// The element count can fit while `element count × dtype size` does
    /// not (e.g. `[u32::MAX, u32::MAX]` f32 is 73,786,976,260,478,468,100
    /// bytes). Saturating the product would make the layout accept a byte
    /// length it cannot represent, so construction rejects it instead.
    #[error("shape {shape:?} of {dtype} has a byte footprint that overflows u64")]
    ByteLenOverflow {
        /// The offending dimensions.
        shape: Vec<u32>,
        /// The dtype that pushed the footprint past `u64`.
        dtype: Dtype,
    },

    /// A layout's strides are not the canonical row-major strides.
    ///
    /// T0 binds row-major, contiguous, unpadded buffers only (handover
    /// §4.4): padded and transposed strides are a later slice, and they
    /// arrive as an explicit layout, never as an accident.
    #[error("shape {shape:?} with strides {strides:?} is not row-major contiguous")]
    NonContiguous {
        /// The dimensions the strides were checked against.
        shape: Vec<u32>,
        /// The rejected strides, in element units.
        strides: Vec<u64>,
    },

    /// The number of operands does not match the op's declared ports.
    #[error("op `{op}` takes {expected} input(s), got {got}")]
    ArityMismatch {
        /// The op's contract name, e.g. `"matmul"`.
        op: &'static str,
        /// Ports the op declares.
        expected: usize,
        /// Operands actually supplied.
        got: usize,
    },

    /// An operand's dtype differs from its port's declared dtype.
    ///
    /// This is the variant that keeps f16 out of f32 arithmetic: narrowing
    /// and widening happen only through [`Cast`](crate::ops::l0::Cast).
    #[error("op `{op}` port {port} expects {expected}, got {got}")]
    DtypeMismatch {
        /// The op's contract name.
        op: &'static str,
        /// Zero-based input port index.
        port: usize,
        /// The port's declared dtype.
        expected: Dtype,
        /// The operand's dtype.
        got: Dtype,
    },

    /// An operand has the wrong rank for its port.
    #[error("op `{op}` port {port} expects {expected}, got rank {got}")]
    RankMismatch {
        /// The op's contract name.
        op: &'static str,
        /// Zero-based input port index.
        port: usize,
        /// The rank the port requires, e.g. `"rank 2"`.
        expected: &'static str,
        /// The operand's actual rank.
        got: usize,
    },

    /// Two matmul operands disagree on the contracted dimension.
    #[error("op `{op}` inner dimensions disagree: lhs {lhs} vs rhs {rhs}")]
    InnerDimensionMismatch {
        /// The op's contract name.
        op: &'static str,
        /// The left operand's contracted dimension.
        lhs: u32,
        /// The right operand's contracted dimension.
        rhs: u32,
    },

    /// Two operands disagree on a batch or head dimension.
    #[error("op `{op}` batch dimensions disagree: lhs {lhs:?} vs rhs {rhs:?}")]
    BatchDimensionMismatch {
        /// The op's contract name.
        op: &'static str,
        /// The left operand's batch dimensions.
        lhs: Vec<u32>,
        /// The right operand's batch dimensions.
        rhs: Vec<u32>,
    },

    /// Two operands cannot be broadcast together.
    #[error("op `{op}` cannot broadcast {lhs:?} with {rhs:?}")]
    BroadcastIncompatible {
        /// The op's contract name.
        op: &'static str,
        /// The left operand's shape.
        lhs: Vec<u32>,
        /// The right operand's shape.
        rhs: Vec<u32>,
    },

    /// A declared output shape disagrees with the op's shape rule.
    #[error("op `{op}` infers output {expected:?}, caller declared {got:?}")]
    OutputShapeMismatch {
        /// The op's contract name.
        op: &'static str,
        /// The shape the op's rule infers.
        expected: Vec<u32>,
        /// The shape the caller declared.
        got: Vec<u32>,
    },

    /// A reduction axis is not within the operand's rank.
    #[error("op `{op}` axis {axis} is out of range for rank {rank}")]
    AxisOutOfRange {
        /// The op's contract name.
        op: &'static str,
        /// The requested axis.
        axis: usize,
        /// The operand's rank.
        rank: usize,
    },

    /// A cast pair is not part of the dtype policy.
    ///
    /// Identity casts (`f32 -> f32`) and narrowed two-step casts
    /// (`f16 -> bf16`) are rejected: the former is
    /// [`Copy`](crate::ops::l0::Copy), the latter is two explicit steps
    /// through f32.
    #[error("cast from {from} to {to} is not defined; use copy or two steps through f32")]
    CastNotDefined {
        /// The source dtype.
        from: Dtype,
        /// The target dtype.
        to: Dtype,
    },

    /// A byte span does not match the shape × dtype footprint.
    #[error("`{name}` with shape {shape:?} of {dtype} is {expected} bytes, not {got}")]
    ByteLenMismatch {
        /// The manifest entry or layout label.
        name: String,
        /// The tensor's dimensions.
        shape: Vec<u32>,
        /// The tensor's dtype.
        dtype: Dtype,
        /// The footprint implied by shape × dtype size.
        expected: u64,
        /// The byte length that was declared.
        got: u64,
    },

    /// A manifest entry's byte span does not fit the data section.
    #[error("`{name}` spans {offset}..{end}, past the {available}-byte data section")]
    OutOfBounds {
        /// The manifest entry's name.
        name: String,
        /// The entry's start offset.
        offset: u64,
        /// The entry's exclusive end offset.
        end: u64,
        /// The size of the data section it must fit in.
        available: u64,
    },

    /// Two manifest entries claim intersecting byte spans.
    ///
    /// Entries are distinct parameters: aliasing a byte range would make two
    /// "different" tensors the same memory, which is never what a checkpoint
    /// means (weight tying is expressed by naming one entry twice).
    #[error("`{first}` and `{second}` overlap in {offset}..{end}")]
    OverlappingEntries {
        /// The entry with the lower byte offset.
        first: String,
        /// The entry that starts inside it.
        second: String,
        /// Start of the intersection.
        offset: u64,
        /// Exclusive end of the intersection.
        end: u64,
    },

    /// The manifest entry count does not match the op's declared ports.
    #[error("op `{op}` expects {expected} manifest entr(ies), got {got}")]
    ManifestArityMismatch {
        /// The op's contract name.
        op: &'static str,
        /// Ports the op declares.
        expected: usize,
        /// Entries the manifest supplies.
        got: usize,
    },

    /// The manifest's determinism pin disagrees with the manifest's dtype.
    ///
    /// The pin is what forbids a backend from silently substituting a
    /// narrower or faster element type (TF32, f16 accumulation); a manifest
    /// that pins one dtype and declares another cannot be honoured.
    #[error("manifest pins {pinned} but declares {manifest}")]
    DeterminismDtypeMismatch {
        /// The dtype the determinism policy pins.
        pinned: Dtype,
        /// The dtype the manifest declares.
        manifest: Dtype,
    },

    /// A dtype that is not the f32 contract reached a contract boundary.
    ///
    /// v1 manifests are f32-only, so this guard is expected to be
    /// unreachable today; it fires the day a manifest type carries a
    /// non-contract dtype without an explicit `Cast` at the load boundary.
    #[error("{got} is not the f32 contract dtype; narrow explicitly via cast")]
    NonContractDtype {
        /// The rejected dtype.
        got: Dtype,
    },
}
