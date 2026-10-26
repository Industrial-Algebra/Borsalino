// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! The CPU reference implementation of L0–L2.
//!
//! This is **first-class, not a fallback** (handover §4.2). It is two things
//! at once:
//!
//! 1. Ijima's execution path — Ijima runs on GPU-less hardware, so the
//!    forward pass has to be computable without a device.
//! 2. The **equivalence oracle** for every WGSL and Metal kernel T2 adds:
//!    the reference is what a kernel is measured against, bit for bit
//!    ([`oracle::check_bit_identical`](crate::oracle::check_bit_identical)).
//!
//! Every op here implements [`Executable`](crate::exec::Executable), so a
//! kernel and this reference are interchangeable behind one test harness.
//!
//! ## Determinism policy — the load-bearing decision
//!
//! **Every reduction sums in strict left-to-right order over its logical
//! (row-major) operand, in a single thread.** For `matmul` that is the
//! `k`-loop ascending, one output element at a time; for `sum`, `max`,
//! `rowwise_l2_norm` and `masked_mean` it is the flattened element order;
//! for `masked_mean` the mask's own sum uses the same order.
//!
//! Why left-to-right and not a tree:
//!
//! - It is the only order that presumes nothing about the parallelism T2
//!   will use. A tree order is fixed only once its shape is fixed, and a GPU
//!   tree's shape is the kernel's workgroup size — a T2 decision. Pinning a
//!   serial order now gives an unconditional determinism guarantee (same
//!   inputs, same bits, any run, any platform) and leaves T2 to decide
//!   whether to serialize the reduction to match, or to migrate this
//!   contract to a pinned tree.
//! - It is trivially reproducible by any implementation that walks the same
//!   indices, which is what makes the oracle meaningful.
//!
//! There are no threads, no atomics, no hash-map iteration, and no
//! order-dependent fast paths anywhere in this module; the tests assert
//! repeat-run bit equality and pin the order with value sets where a
//! different order changes the bits.
//!
//! ## Accumulation precision — f32 throughout, deliberately
//!
//! Every accumulation is in f32, including reduction intermediates. f64
//! accumulation was considered and rejected:
//!
//! - **The oracle must live in the arithmetic domain of the kernels it
//!   judges.** WGSL has no f64: a kernel accumulating in f32 could never
//!   bit-match an f64-accumulating oracle, so an f64 oracle would be
//!   un-matchable by construction and useless as an equivalence gate.
//! - f64 accumulation buys accuracy this layer does not owe. f64 exactness
//!   is Amari's half (handover §3); Borsalino's contract is f32, and the
//!   dtype policy forbids silently widening for speed *or* accuracy.
//!
//! The tests pin this decision: a sum where f32 accumulation provably
//! differs from f64 asserts the f32 result.
//!
//! ## Non-finite policy
//!
//! NaN and ±inf inputs are rejected with
//! [`TensorError::NonFiniteInput`](crate::error::TensorError::NonFiniteInput)
//! before compute, for every op (see
//! [`exec::require_finite`](crate::exec::require_finite)). Outputs are not
//! policed: a cast overflows to infinity exactly where its
//! [`CastContract`](crate::dtype::CastContract) says it does, and
//! `broadcast_div` by zero produces the IEEE-754 result.
//!
//! ## Allocation
//!
//! The output buffer is the caller's: nothing here allocates one. Three
//! things do allocate, all documented at their site:
//!
//! - the operand gate clones the layouts into the T0 `check_inputs` slice
//!   ([`exec::validate`](crate::exec::validate)) — `O(rank)`, not
//!   `O(elements)`, and the price of having one gate implementation;
//! - broadcast indexing precomputes per-axis strides and a coordinate
//!   odometer (`O(rank)` each) so the element loop never divides;
//! - error paths allocate their message payloads.

pub mod cast;
pub mod l0;
pub mod l1;
pub mod l2;
