// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Shared fixtures for the T1 CPU-reference tests.
//!
//! Each integration-test binary compiles this module separately, so a helper
//! used by only one of them reads as dead code in the others.
#![allow(dead_code)]

use borsalino_tensor::dtype::Dtype;
use borsalino_tensor::exec::View;
use borsalino_tensor::layout::{Shape, TensorLayout};

/// A shape, panicking when the fixture itself is invalid.
pub fn shape(dims: &[u32]) -> Shape {
    Shape::new(dims).expect("fixture shapes are valid")
}

/// A row-major layout of the given dtype.
pub fn layout(dtype: Dtype, dims: &[u32]) -> TensorLayout {
    TensorLayout::row_major(dtype, shape(dims)).expect("fixture layouts are valid")
}

/// A row-major f32 layout.
pub fn f32_layout(dims: &[u32]) -> TensorLayout {
    layout(Dtype::F32, dims)
}

/// A row-major u32 layout (an index operand).
pub fn u32_layout(dims: &[u32]) -> TensorLayout {
    layout(Dtype::U32, dims)
}

/// An f32 operand view.
pub fn view<'a>(layout: &'a TensorLayout, data: &'a [f32]) -> View<'a, f32> {
    View::new(layout, data)
}

/// The raw bit patterns of an f32 slice — the determinism comparison unit.
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The sentinel a "the output must not be touched" assertion fills a buffer
/// with. Distinct from every fixture value used in these tests.
pub const SENTINEL: f32 = -1234.5678;

/// Run `work` on a worker thread and fail the test if it has not finished
/// within `limit`.
///
/// Nothing in the reference may take proportional-to-a-declared-dimension
/// time when the work itself is empty, so a timeout is the assertion: a hung
/// loop fails the test instead of wedging the suite. The worker is detached
/// rather than joined — the process exit kills a straggler.
pub fn within(limit: std::time::Duration, work: impl FnOnce() + Send + 'static) {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        work();
        let _ = sender.send(());
    });
    assert!(
        matches!(receiver.recv_timeout(limit), Ok(())),
        "the op did not finish within {limit:?}: an empty dimension is being iterated"
    );
}
