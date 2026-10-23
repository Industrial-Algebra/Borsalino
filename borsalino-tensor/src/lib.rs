// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! # borsalino-tensor — the typed tensor op-set contract
//!
//! > Borsalino is the workhorse; Amari is the precision.
//!
//! This crate declares **what tensor operations the Borsalino substrate will
//! perform**, as types, before any kernel exists. It is slice T0 of
//! `docs/plans/2026-10-16-borsalino-tensor-extension-handover.md`: the
//! measured candle surface the Industrial Algebra fleet actually uses, cut
//! into an op inventory with an explicit dtype policy, a determinism class
//! per op, a shape rule per op, and a shared layout manifest with Katakuri.
//!
//! ## What T0 is
//!
//! - [`ops`] — the L0/L1/L2 inventory from handover §3: elementwise and
//!   buffer ops, reductions, and GEMM. Each op is a type implementing
//!   [`op::TensorOp`]; the trait carries its name, level, ports, determinism
//!   class, and shape rule as associated constants.
//! - [`layout`] — shapes, strides, and the row-major contiguous layout
//!   contract that makes Katakuri weights zero-copy.
//! - [`dtype`] — f32 as the contract dtype, with f16/bf16 reachable only
//!   through an explicit [`ops::l0::Cast`] whose semantics are documented.
//! - [`error`] — one variant per way a call can be rejected, each naming the
//!   op, port, or manifest entry at fault.
//! - [`manifest`] — the Katakuri layout-manifest interop: validate a
//!   checkpoint's declared layout and bind its entries to an op's inputs.
//!
//! ## What T0 is not
//!
//! No kernels, no CPU reference, no compute of any kind — those are T1 (CPU
//! reference for L0–L2) and T2 (WGSL kernels behind the DeepReinforce and
//! determinism gates). Level 3 (activations, softmax, the fused 4-D
//! attention mask) is not declared here: it composes from these.
//!
//! Also out of scope by decision (handover §3): autodiff, training,
//! optimizers, gradients; model specs, tokenization, and checkpoint loading
//! (Katakuri's half); f64 exactness (Amari's half); graphics.
//!
//! ## The three contracts every op carries
//!
//! 1. **Dtype.** Data ports are f32. Nothing narrows silently: f16 and bf16
//!    enter only via [`ops::l0::Cast`], whose rounding, overflow, and
//!    subnormal behaviour are pinned by [`dtype::CastContract`].
//! 2. **Determinism.** Every op declares a [`op::DeterminismClass`].
//!    Reductions must use a fixed or tree order; GEMM must fix its
//!    accumulation tiling; no kernel may select a bit-pattern-changing fast
//!    path. This is what Katakuri's manifest pin enforces from the data side.
//! 3. **Shape.** Every op declares a [`op::ShapeRule`], and inference lives
//!    in the rule, so a call cannot declare one shape relation and check
//!    another.
//!
//! Zunesha buffer binding is deliberately absent: the manifest is data, and
//! mapping an entry onto a `zunesha::Buffer` belongs to the consumer
//! (handover §4.4). This crate names no buffer type.
//!
//! ```
//! use borsalino_tensor::layout::Shape;
//! use borsalino_tensor::op::TensorOp;
//! use borsalino_tensor::ops::l2::MatMul;
//!
//! let lhs = Shape::new(&[4, 8]).unwrap();
//! let rhs = Shape::new(&[8, 16]).unwrap();
//! let out = MatMul::infer_shape(&[&lhs, &rhs]).unwrap();
//! assert_eq!(out, Shape::new(&[4, 16]).unwrap());
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod dtype;
pub mod error;
pub mod layout;
pub mod manifest;
pub mod op;
pub mod ops;
