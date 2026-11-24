// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! WGSL kernels for the tensor op set (T2a).
//!
//! One kernel per contracted op, embedded as source — compiled through
//! Borsalino's naga pipeline per backend. Every kernel is
//! [`DeterminismClass::BitExact`](crate::op::DeterminismClass::BitExact):
//! elementwise index math or pure permutation, no reduction order to fix.
//!
//! Binding layout (all kernels): operands in port order, output LAST.
//! Each kernel bounds-guards itself (the pipelines compile with
//! `Unchecked` buffer bounds — the guard is load-bearing, the same
//! contract the substrate's own examples carry).
//!
//! Workgroup size is 256 (1-D dispatch, `(ceil(n/256), 1, 1)`); the
//! executor computes it from the output element count.

/// `relu(x) = if x > 0.0 { x } else { 0.0 }` — the comparison form, the
/// same contract the CPU reference pins (NOT `f32::max`, whose signed-zero
/// result is unspecified). Bindings: input, output.
pub const RELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    output[i] = select(0.0, input[i], input[i] > 0.0);
}
"#;

/// `add(a, b)` for same-shape operands (T2a; broadcast variants are T2b).
/// Bindings: lhs, rhs, output.
pub const ADD: &str = r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    output[i] = lhs[i] + rhs[i];
}
"#;

/// `mul(a, b)` for same-shape operands (T2a; broadcast variants are T2b).
/// Bindings: lhs, rhs, output.
pub const MUL: &str = r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    output[i] = lhs[i] * rhs[i];
}
"#;

/// Embedding-table row gather: `[V, D]` table + `[N]` u32 indices →
/// `[N, D]` rows. One thread per OUTPUT ELEMENT (not per row — the D
/// axis rides the same dispatch). Bindings: table, indices, output.
pub const GATHER: &str = r#"
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let n = arrayLength(&indices);
    let d = arrayLength(&output) / max(n, 1u);
    if (flat >= arrayLength(&output)) { return; }
    let row = flat / d;
    let col = flat % d;
    let index = indices[row];
    // Out-of-range indices are a caller error (the CPU reference
    // rejects them at the gate); the kernel clamps defensively so a
    // contract violation reads garbage, never out-of-bounds memory.
    let clamped = min(index, (arrayLength(&table) / max(d, 1u)) - 1u);
    output[flat] = table[clamped * d + col];
}
"#;

/// `[R, C] -> [C, R]` transpose — pure permutation with the same widened
/// index math the CPU reference uses (`u32` products computed in `u64`;
/// WGSL has no u32 overflow traps, so the casts are correctness, not
/// defense). One thread per output element.
///
/// Bindings: input, output, then an op-params buffer (a single `u32`
/// carrying the input's row count R — the shape is NOT recoverable from
/// binding lengths alone, since input and output have the same element
/// count). Params buffers come AFTER the output, a convention every
/// shape-carrying kernel follows.
pub const TRANSPOSE_2D: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn transpose_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = params[0];
    let cols = arrayLength(&input) / max(rows, 1u);
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    // Output is [C, R]: index i = c * rows + r.
    let c = i / rows;
    let r = i % rows;
    let in_index = u64(r) * u64(cols) + u64(c);
    output[i] = input[u32(in_index)];
}
"#;
