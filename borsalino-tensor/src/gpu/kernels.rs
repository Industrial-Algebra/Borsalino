// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! WGSL kernels for the tensor op set (T2a + T2b).
//!
//! One kernel per contracted op, embedded as source — compiled through
//! Borsalino's naga pipeline per backend. The T2a set is
//! [`DeterminismClass::BitExact`](crate::op::DeterminismClass::BitExact):
//! elementwise index math or pure permutation. The T2b broadcast kernels
//! are the same class (stride-mapped data movement plus one IEEE op);
//! the T2b reductions carry
//! [`DeterminismClass::FixedOrderReduction`](crate::op::DeterminismClass::FixedOrderReduction):
//! each accumulating loop runs strictly left-to-right over its axis,
//! one invocation per output element, with one explicit `fma` per
//! mul-into-add term — WGSL promises nothing about contraction, so the
//! fusion is explicit on both sides (see the module docs in
//! [`super`]).
//! That forecloses intra-axis parallelism (a tree or chunked split is
//! a different parenthesization, hence different bits); parallelism is
//! across output rows/columns, where it is free.
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
/// `broadcast_div(a, b)` for same-shape operands (the elementwise arm of
/// T2b's BroadcastDiv). Division by zero passes IEEE `inf`/`NaN` through.
/// Bindings: lhs, rhs, output.
pub const DIV: &str = r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    output[i] = lhs[i] / rhs[i];
}
"#;

/// `gather(a, idx)`: embedding-table row lookup with a defensive clamp
/// (the executor validates indices host-side first — the clamp is the
/// last line, not the contract). Bindings: table, indices, output.
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

/// The shared broadcast body: `OP(lhs_at, rhs_at)`. A `const fn` can't
/// build the source, so this macro stamps the one body per op.
macro_rules! BROADCAST_KERNEL {
    ($entry:ident, $op:expr) => {
        concat!(
            r#"
@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn "#,
            stringify!($entry),
            r#"(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    let rank = params[0u];
    var rem = i;
    var lo: u32 = 0u;
    var ro: u32 = 0u;
    // Decompose from the LAST axis down: row-major's fastest-varying
    // axis is the least significant digit of the flat index.
    for (var d = rank; d > 0u; d = d - 1u) {
        let dim = params[d];
        let c = rem % dim;
        rem = rem / dim;
        lo = lo + c * params[rank + d];
        ro = ro + c * params[2u * rank + d];
    }
    let a = lhs[lo];
    let b = rhs[ro];
    output[i] = "#,
            $op,
            r#";
}
"#
        )
    };
}
/// Broadcast binary ops (T2b): NumPy-style broadcasting, index-mapped —
/// the executor passes `rank`, the output dims, and both operands'
/// broadcast strides (stride 0 on broadcast axes) in the params buffer.
/// Per element: de-linearize the flat output index, map to both operand
/// offsets, apply one IEEE op. Bindings: lhs, rhs, output, params.
///
/// Params layout (`u32`s): `[rank, out_dims..., lhs_strides...,
/// rhs_strides...]`. All offsets stay below the output element count,
/// which the gate caps at `u32::MAX`.
/// Broadcast `add` (same body, `a + b`). See [`BROADCAST_ADD`]'s layout note.
pub const BROADCAST_ADD: &str = BROADCAST_KERNEL!(broadcast_add, "a + b");
/// Broadcast `mul` (same body, `a * b`).
pub const BROADCAST_MUL: &str = BROADCAST_KERNEL!(broadcast_mul, "a * b");
/// Broadcast `div` (same body, `a / b`; division by zero passes IEEE
/// `inf`/`NaN` through — T0's deliberate contract).
pub const BROADCAST_DIV: &str = BROADCAST_KERNEL!(broadcast_div, "a / b");

/// `sum` (T2b): full reduction to a scalar, strictly sequential
/// left-to-right — one invocation, `workgroup_size(1)`. Bindings:
/// input, output.
pub const SUM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn sum() {
    var total = 0.0;
    for (var i = 0u; i < arrayLength(&input); i = i + 1u) {
        total = total + input[i];
    }
    output[0] = total;
}
"#;

/// `max` (T2b): full reduction to a scalar, strictly-`>` with
/// earliest-wins ties — the CPU reference's replayable order
/// (that is the -0.0 vs 0.0 case). One invocation. Bindings: input,
/// output.
pub const MAX: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn max() {
    var best = input[0];
    for (var i = 1u; i < arrayLength(&input); i = i + 1u) {
        let v = input[i];
        if (v > best) { best = v; }
    }
    output[0] = best;
}
"#;

/// `rowwise_l2_norm` (T2b): `[R, C] -> [R]`, one invocation per row
/// (parallel across rows), sequential accumulation across the row,
/// `sqrt` at the end (architecture-precision — see the module docs).
/// Per term: one EXPLICIT `fma` — matching the CPU reference's
/// `mul_add`; drivers contract mul-into-add anyway, so the fusion is
/// made explicit on both sides to pin identical bits. Params: `[C]`.
/// Bindings: input, output, params.
pub const ROWWISE_L2_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn rowwise_l2_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= arrayLength(&output)) { return; }
    let width = params[0u];
    var sos = 0.0;
    let base = row * width;
    for (var c = 0u; c < width; c = c + 1u) {
        let v = input[base + c];
        sos = fma(v, v, sos);
    }
    output[row] = sqrt(sos);
}
"#;

/// `masked_mean` (T2b): `[B,S,D] + [B,S] -> [B,D]`, one invocation per
/// `(batch, column)` pair. Both the mask weight and the weighted sum
/// accumulate strictly left-to-right over the token axis — the same
/// single order the CPU reference pins for both — one explicit `fma`
/// per term (the same cross-device fusion pin as rowwise_l2_norm). An all-zero mask row
/// defines the mean as zero, never NaN. Params: `[S, D]`. Bindings:
/// values, mask, output, params.
pub const MASKED_MEAN: &str = r#"
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn masked_mean(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&output)) { return; }
    let tokens = params[0u];
    let width = params[1u];
    let index = i / width;
    let column = i % width;
    var weight = 0.0;
    for (var s = 0u; s < tokens; s = s + 1u) {
        weight = weight + mask[index * tokens + s];
    }
    var weighted = 0.0;
    for (var s = 0u; s < tokens; s = s + 1u) {
        let at = index * tokens + s;
        weighted = fma(mask[at], values[at * width + column], weighted);
    }
    output[i] = select(0.0, weighted / weight, weight != 0.0);
}
"#;
