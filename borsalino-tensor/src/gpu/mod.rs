// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! GPU execution of the tensor op set through Borsalino (T2a + T2b).
//!
//! [`TensorGpu`] wraps any [`GpuBackend`] and executes contracted ops as
//! WGSL kernels: compile-once-per-op (cached), operands bound in port
//! order with the output last and any op-params buffers after it, one
//! dispatch sized from the output element count. The CPU reference
//! ([`crate::cpu`]) remains the equivalence oracle — every kernel test
//! in this crate runs both and compares per
//! [`DeterminismClass`](crate::op::DeterminismClass).
//!
//! T2b adds broadcast execution (stride-mapped in the kernel, never a
//! materialized expansion — equal shapes keep the elementwise pipeline)
//! and the fixed-order reductions (`sum`, `max`, `rowwise_l2_norm`,
//! `masked_mean`). Reduction kernels preserve the CPU reference's exact
//! parenthesization: one invocation per output element, accumulating
//! strictly left-to-right, one **explicit fused multiply-add per term**
//! (`fma` in WGSL, `mul_add` on the CPU — drivers contract mul-into-add
//! anyway, so the fusion is made explicit on both sides to pin identical
//! bits). That forecloses intra-axis parallelism (a tree or chunked
//! split is a different parenthesization, hence different bits);
//! parallelism is across output rows, where it is free.
//!
//! The layer is thin by design: buffer lifetimes are per-call (created,
//! dispatched, read back), matching the verification harness. Buffer
//! reuse, manifest-driven zero-copy weight binding, and the Katakuri
//! checkpoint path land with the consumer-facing execution API (T2b+).
//!
//! # Architecture-precision ops — the second GPU equivalence seam
//!
//! On this stack (RTX 5080, Vulkan via naga) two f32 operations are NOT
//! correctly rounded where the CPU's are:
//!
//! - **division** — finite quotients can round one ulp off the
//!   correctly-rounded result (`1e10 / 7` observed);
//! - **`sqrt`** — up to two ulp off (RTX 5080, observed).
//!
//! Affected final results: `masked_mean` (ends in `weighted / weight`)
//! and `rowwise_l2_norm` (ends in `sqrt`). Their oracles assert
//! ulp-distance bounds instead of bit-equality; **dispatch-to-dispatch
//! determinism is unconditional** (the same approximation replays
//! identically). Additionally, **NaN payloads are platform-defined**:
//! GPU `0/0` emits `0x7FFFFFFF` where the CPU emits the canonical quiet
//! NaN — NaN results compare by class, never by bits. Everything else
//! (elementwise ops, broadcast mapping, gather, transpose, `sum`,
//! `max`) is bit-exact against the CPU oracle on the normal range.
//!
//! # Subnormal policy (FTZ/DAZ) — the first GPU equivalence seam
//!
//! [`DeterminismClass::BitExact`](crate::op::DeterminismClass::BitExact)
//! means two things here, and the split is load-bearing:
//!
//! 1. **Dispatch-to-dispatch**: identical inputs produce identical bits —
//!    unconditionally true (the flush behaviour below is itself
//!    deterministic).
//! 2. **GPU vs the CPU oracle**: bit-equal **on the normal range**.
//!    Subnormals do not survive GPU arithmetic: Vulkan and Metal permit
//!    flush-to-zero and denormals-are-zero, WGSL exposes no way to
//!    disable them, and the hardware exercises both — arithmetic
//!    flushes subnormal inputs AND results to a **sign-preserving zero**,
//!    and comparisons read subnormals as zero (so `relu` of a positive
//!    subnormal yields `+0.0`, not the value).
//!
//! The equivalence tests pin both halves: normal-range fixtures require
//! oracle bit-equality; subnormal probes pin the flush rule itself.
//! Katakuri's `TrustPolicy` keeps subnormal weights (finite and legal) —
//! consumers moving such weights to GPU execution accept the flush; the
//! seam is this policy, stated once, here.

use std::collections::HashMap;

use borsalino::{ComputePipeline, GpuBackend, GpuBuffer};

pub mod kernels;

use crate::error::TensorError;
use crate::op::TensorOp;
use crate::ops::l0::Gather;

/// WGSL source and entry point for a contracted op's kernel.
pub trait GpuKernel: TensorOp {
    /// The kernel source (see [`crate::gpu::kernels`]).
    const SOURCE: &'static str;
    /// The kernel entry-point function name.
    const ENTRY: &'static str;
    /// Op-params carried in a trailing `u32` buffer (shape data the
    /// kernel cannot recover from binding lengths). Empty = none.
    fn params(input_shapes: &[&crate::layout::Shape]) -> Result<Vec<u32>, TensorError>;
}

/// A GPU executor for the tensor op set over any Borsalino backend.
pub struct TensorGpu<B: GpuBackend> {
    backend: B,
    pipelines: HashMap<&'static str, ComputePipeline>,
    /// Broadcast variants, keyed by the same op names (T2b).
    broadcast_pipelines: HashMap<&'static str, ComputePipeline>,
}

impl<B: GpuBackend> TensorGpu<B> {
    /// Wrap a backend (the caller owns backend selection and strategy).
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            pipelines: HashMap::new(),
            broadcast_pipelines: HashMap::new(),
        }
    }

    /// Borrow the underlying backend (for Borsalino-native verification
    /// drivers — determinism sweeps, numerical checks).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The compiled-pipeline cache (test observability).
    pub fn pipelines(&self) -> &HashMap<&'static str, ComputePipeline> {
        &self.pipelines
    }

    /// Disjoint field borrows: `backend` is shared, `pipelines` is
    /// mutable — one method, no clone needed.
    fn ensure_pipeline<K: GpuKernel>(
        backend: &B,
        pipelines: &mut HashMap<&'static str, ComputePipeline>,
    ) -> Result<(), TensorError> {
        if !pipelines.contains_key(K::NAME) {
            let compiled =
                backend
                    .compile(K::ENTRY, K::SOURCE)
                    .map_err(|e| TensorError::GpuDispatch {
                        op: K::NAME,
                        message: format!("compile: {e}"),
                    })?;
            pipelines.insert(K::NAME, compiled);
        }
        Ok(())
    }

    /// Compile-once cache for broadcast kernels (mirrors
    /// [`Self::ensure_pipeline`], keyed by the same op name in a
    /// separate map).
    fn ensure_broadcast_pipeline<K: BroadcastKernel>(
        backend: &B,
        pipelines: &mut HashMap<&'static str, ComputePipeline>,
    ) -> Result<(), TensorError> {
        if !pipelines.contains_key(K::NAME) {
            let compiled = backend
                .compile(K::BROADCAST_ENTRY, K::BROADCAST_SOURCE)
                .map_err(gpu_err::<K>)?;
            pipelines.insert(K::NAME, compiled);
        }
        Ok(())
    }

    /// Execute a unary op over flat f32 operands: binds `input` then the
    /// `output` buffer, dispatches, reads the output back.
    ///
    /// The contract gate runs first (r1): shapes through the op's rule,
    /// slice lengths against shapes, output length against the inferred
    /// shape — before any buffer exists.
    pub fn execute_unary<K: GpuKernel>(
        &mut self,
        input: &[f32],
        output_len: usize,
        input_shapes: &[&crate::layout::Shape],
    ) -> Result<Vec<f32>, TensorError> {
        // Entry arity (r3 P1): the gate checks shapes-vs-INPUTS, but
        // `execute_unary::<Add>` with BOTH shapes passed it — the zip of
        // layouts with a one-element lens slice silently dropped the
        // missing operand's span check and reached compilation. Each
        // entry serves exactly its own arity.
        if K::INPUTS.len() != 1 {
            return Err(TensorError::GpuDispatch {
                op: K::NAME,
                message: format!(
                    "execute_unary serves 1-input ops; {} takes {} — use the matching entry",
                    K::NAME,
                    K::INPUTS.len()
                ),
            });
        }
        require_f32_ports::<K>()?;
        let out_shape = gate::<K>(input_shapes, &[input.len()], output_len)?;
        require_elementwise::<K>(input_shapes, &out_shape)?;
        Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
        let in_buf = self
            .backend
            .create_buffer(input)
            .map_err(|e| gpu_err::<K>(e))?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(output_len)
            .map_err(|e| gpu_err::<K>(e))?;
        let params = K::params(input_shapes)?;
        let params_buf = if params.is_empty() {
            None
        } else {
            Some(
                self.backend
                    .create_buffer(&params)
                    .map_err(|e| gpu_err::<K>(e))?,
            )
        };

        let mut buffers: Vec<&GpuBuffer> = vec![&in_buf, &out_buf];
        if let Some(pb) = params_buf.as_ref() {
            buffers.push(pb);
        }

        let workgroups = workgroup_count(output_len);
        self.backend
            .dispatch(pipeline, &buffers, workgroups)
            .map_err(|e| gpu_err::<K>(e))?;
        let result: Vec<f32> = self
            .backend
            .read_buffer(&out_buf)
            .map_err(|e| gpu_err::<K>(e))?;
        Ok(result)
    }

    /// Execute a binary op over flat f32 operands (same-shape class):
    /// binds lhs, rhs, then the output buffer.
    pub fn execute_binary<K: GpuKernel + BroadcastKernel>(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
        output_len: usize,
        input_shapes: &[&crate::layout::Shape],
    ) -> Result<Vec<f32>, TensorError> {
        if K::INPUTS.len() != 2 {
            return Err(TensorError::GpuDispatch {
                op: K::NAME,
                message: format!(
                    "execute_binary serves 2-input ops; {} takes {} — use the matching entry",
                    K::NAME,
                    K::INPUTS.len()
                ),
            });
        }
        require_f32_ports::<K>()?;
        let out_shape = gate::<K>(input_shapes, &[lhs.len(), rhs.len()], output_len)?;
        // T2b: unequal counts now EXECUTE for Broadcast-rule ops —
        // stride-mapped in the kernel, never a materialized expansion.
        // Equal counts keep the plain elementwise pipeline (no params
        // buffer, no per-element de-linearization).
        let broadcast = input_shapes
            .iter()
            .any(|shape| shape.element_count() != out_shape.element_count());
        if broadcast && K::SHAPE_RULE != crate::op::ShapeRule::Broadcast {
            return Err(TensorError::GpuDispatch {
                op: K::NAME,
                message: format!(
                    "op's rule is {:?}; unequal shapes are not broadcast for it",
                    K::SHAPE_RULE
                ),
            });
        }
        let (pipeline, params) = if broadcast {
            Self::ensure_broadcast_pipeline::<K>(&self.backend, &mut self.broadcast_pipelines)?;
            let pipeline = self.broadcast_pipelines.get(K::NAME).expect("just ensured");
            (pipeline, broadcast_params(input_shapes, &out_shape)?)
        } else {
            Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
            let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
            (pipeline, K::params(input_shapes)?)
        };
        let lhs_buf = self
            .backend
            .create_buffer(lhs)
            .map_err(|e| gpu_err::<K>(e))?;
        let rhs_buf = self
            .backend
            .create_buffer(rhs)
            .map_err(|e| gpu_err::<K>(e))?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(output_len)
            .map_err(|e| gpu_err::<K>(e))?;
        let params_buf = if params.is_empty() {
            None
        } else {
            Some(
                self.backend
                    .create_buffer(&params)
                    .map_err(|e| gpu_err::<K>(e))?,
            )
        };
        let mut buffers: Vec<&GpuBuffer> = vec![&lhs_buf, &rhs_buf, &out_buf];
        if let Some(pb) = params_buf.as_ref() {
            buffers.push(pb);
        }
        self.backend
            .dispatch(pipeline, &buffers, workgroup_count(output_len))
            .map_err(|e| gpu_err::<K>(e))?;
        let result: Vec<f32> = self
            .backend
            .read_buffer(&out_buf)
            .map_err(|e| gpu_err::<K>(e))?;
        Ok(result)
    }
}

/// Broadcast execution (T2b): ops whose `ShapeRule` is `Broadcast`
/// execute unequal-shape operands through a stride-mapped kernel instead
/// of materializing an expanded operand. One trait per op because the
/// broadcast source is distinct from the same-shape [`GpuKernel`]
/// source (both stay live: equal shapes take the elementwise pipeline).
pub trait BroadcastKernel: crate::op::TensorOp {
    /// The broadcast WGSL source for this op.
    const BROADCAST_SOURCE: &'static str;
    /// The broadcast shader's entry point — distinct from `NAME` (the
    /// compiled entry is `broadcast_<op>`; compiling by op name looks up
    /// an entry point the source does not contain and the driver
    /// rejects the module).
    const BROADCAST_ENTRY: &'static str;
}

impl BroadcastKernel for crate::ops::l0::Add {
    const BROADCAST_SOURCE: &'static str = kernels::BROADCAST_ADD;
    const BROADCAST_ENTRY: &'static str = "broadcast_add";
}

impl BroadcastKernel for crate::ops::l0::Mul {
    const BROADCAST_SOURCE: &'static str = kernels::BROADCAST_MUL;
    const BROADCAST_ENTRY: &'static str = "broadcast_mul";
}

impl BroadcastKernel for crate::ops::l0::BroadcastDiv {
    const BROADCAST_SOURCE: &'static str = kernels::BROADCAST_DIV;
    const BROADCAST_ENTRY: &'static str = "broadcast_div";
}

impl GpuKernel for crate::ops::l0::BroadcastDiv {
    const SOURCE: &'static str = kernels::DIV;
    const ENTRY: &'static str = "div";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

/// Host-side broadcast params (T2b): `[rank, out_dims..., lhs_strides...,
/// rhs_strides...]` as `u32`s. Strides are stride-0 on broadcast axes —
/// the same mapping the CPU reference's `broadcast_strides` computes, so
/// both sides walk identical element pairs.
fn broadcast_params(input_shapes: &[&Shape], out_shape: &Shape) -> Result<Vec<u32>, TensorError> {
    let rank = out_shape.rank();
    let mut params = Vec::with_capacity(1 + 3 * rank);
    params.push(rank as u32);
    params.extend(out_shape.dims().iter().copied());
    for shape in input_shapes {
        let strides = crate::cpu::l0::broadcast_strides(shape, rank);
        params.extend(strides.iter().map(|&s| s as u32));
    }
    Ok(params)
}

fn gpu_err<K: TensorOp>(e: borsalino::GpuError) -> TensorError {
    TensorError::GpuDispatch {
        op: K::NAME,
        message: e.to_string(),
    }
}

/// The generic f32 entry points refuse mixed-dtype ops (r2 P1): u32
/// and f32 are both 4 bytes, so byte-length checks cannot tell a
/// `[u32]`-shaped port fed an `&[f32]` slice apart — and that path
/// would bypass `execute_gather`'s index validation into Unchecked
/// reads. Mixed-dtype ops take dedicated entries.
fn require_f32_ports<K: TensorOp>() -> Result<(), TensorError> {
    for (port, input) in K::INPUTS.iter().enumerate() {
        if input.dtype != crate::dtype::Dtype::F32 {
            return Err(TensorError::GpuDispatch {
                op: K::NAME,
                message: format!(
                    "port {port} takes {:?}; this entry point is f32-only (use a dedicated entry)",
                    input.dtype
                ),
            });
        }
    }
    Ok(())
}

/// The executor's contract gate (review r1 P1): shapes must satisfy the
/// op's rule, operand slices must match their shapes' element counts,
/// and `output_len` must equal the inferred output — BEFORE any buffer
/// is created or dispatched. Kernels compile with `Unchecked` buffer
/// bounds, so a short operand against a declared shape would otherwise
/// be an unchecked GPU read.
///
/// Also rejects output counts the 1-D dispatch grid cannot cover
/// (r1 P2: `ceil(n/256)` saturates `u32` above `u32::MAX` elements and
/// would silently omit the tail).
fn gate<K: TensorOp>(
    input_shapes: &[&Shape],
    input_lens: &[usize],
    output_len: usize,
) -> Result<Shape, TensorError> {
    use crate::exec::{check_output, check_span, validate};

    // Arity first (r2 P1): a caller passing MORE shapes than the op has
    // ports would index `K::INPUTS` out of bounds; fewer shapes means a
    // port silently defaults to layout[0]'s dtype downstream.
    if input_shapes.len() != K::INPUTS.len() {
        return Err(TensorError::GpuDispatch {
            op: K::NAME,
            message: format!(
                "executor arity mismatch: op takes {} inputs, {} shapes given",
                K::INPUTS.len(),
                input_shapes.len()
            ),
        });
    }

    let layouts: Vec<crate::layout::TensorLayout> = input_shapes
        .iter()
        .enumerate()
        .map(|(port, shape)| {
            crate::layout::TensorLayout::row_major(K::INPUTS[port].dtype, (*shape).clone())
        })
        .collect::<Result<_, _>>()?;
    let layout_refs: Vec<&crate::layout::TensorLayout> = layouts.iter().collect();
    let out_shape = validate::<K>(&layout_refs)?;
    for (port, (layout, len)) in layouts.iter().zip(input_lens).enumerate() {
        check_span::<K>(port, layout, *len)?;
    }
    check_output::<K>(&out_shape, output_len)?;
    if output_len > u32::MAX as usize {
        return Err(TensorError::GpuDispatch {
            op: K::NAME,
            message: format!(
                "output count {output_len} exceeds the 1-D dispatch limit (u32::MAX elements)"
            ),
        });
    }
    Ok(out_shape)
}

/// Elementwise entries additionally require same-shape operands (r2 P1):
/// an op's rule may infer a larger output from broadcast dims — the CPU
/// reference expands; these kernels read elementwise and would need
/// Unchecked OOB reads to satisfy that. Broadcast execution is T2b;
/// reject rather than read past a short operand. (Gather is exempt:
/// its indices operand is `[N]` while the output is `[N, D]` by
/// design.)
fn require_elementwise<K: TensorOp>(
    input_shapes: &[&Shape],
    out_shape: &Shape,
) -> Result<(), TensorError> {
    for shape in input_shapes {
        if shape.element_count() != out_shape.element_count() {
            return Err(TensorError::GpuDispatch {
                op: K::NAME,
                message: format!(
                    "broadcasting is not yet supported by the GPU executor (input {:?} vs output {:?}); T2b",
                    shape, out_shape
                ),
            });
        }
    }
    Ok(())
}

/// 1-D dispatch grid for `n` elements at 256 threads per group.
impl<B: GpuBackend> TensorGpu<B> {
    /// Execute `gather`: `[V, D]` table + `[N]` u32 indices → `[N, D]`.
    ///
    /// Mirrors the CPU reference's contract exactly: the whole index
    /// list is validated BEFORE dispatch — an out-of-range index
    /// rejects with [`TensorError::IndexOutOfRange`] instead of the
    /// kernel's defensive clamp ever mattering (review r1: gather
    /// lacked executor integration; the kernel clamp is the last line,
    /// not the contract).
    pub fn execute_gather(
        &mut self,
        table: &[f32],
        indices: &[u32],
        output_len: usize,
        input_shapes: &[&Shape],
    ) -> Result<Vec<f32>, TensorError> {
        gate::<Gather>(input_shapes, &[table.len(), indices.len()], output_len)?;
        // Index validation (the CPU gate, host-side).
        let rows = input_shapes[0].dims()[0];
        for (position, &index) in indices.iter().enumerate() {
            if index >= rows {
                return Err(TensorError::IndexOutOfRange {
                    op: <Gather as TensorOp>::NAME,
                    position: position as u64,
                    index,
                    rows,
                });
            }
        }
        Self::ensure_pipeline::<Gather>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(Gather::NAME).expect("just ensured");
        let table_buf = self
            .backend
            .create_buffer(table)
            .map_err(gpu_err::<Gather>)?;
        let index_buf = self
            .backend
            .create_buffer(indices)
            .map_err(gpu_err::<Gather>)?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(output_len)
            .map_err(gpu_err::<Gather>)?;
        let buffers: Vec<&GpuBuffer> = vec![&table_buf, &index_buf, &out_buf];
        self.backend
            .dispatch(pipeline, &buffers, workgroup_count(output_len))
            .map_err(gpu_err::<Gather>)?;
        let result: Vec<f32> = self
            .backend
            .read_buffer(&out_buf)
            .map_err(gpu_err::<Gather>)?;
        Ok(result)
    }

    /// Execute `sum` (T2b): full reduction to a scalar. The kernel runs
    /// one sequential left-to-right accumulation — the CPU reference's
    /// exact parenthesization (an empty input sums to `0.0`, matching
    /// the CPU loop).
    pub fn execute_sum(&mut self, input: &[f32], shape: &Shape) -> Result<Vec<f32>, TensorError> {
        self.full_reduction::<crate::ops::l1::Sum>(input, &[shape])
    }

    /// Execute `max` (T2b): full reduction to a scalar, strictly-`>`
    /// with earliest-wins ties. An EMPTY input rejects with
    /// [`TensorError::EmptyReduction`] before dispatch — the CPU gate.
    pub fn execute_max(&mut self, input: &[f32], shape: &Shape) -> Result<Vec<f32>, TensorError> {
        if input.is_empty() {
            return Err(TensorError::EmptyReduction {
                op: crate::ops::l1::Max::NAME,
            });
        }
        self.full_reduction::<crate::ops::l1::Max>(input, &[shape])
    }

    /// Shared full-reduction dispatch: gate, one-invocation dispatch,
    /// scalar readback.
    fn full_reduction<K: GpuKernel>(
        &mut self,
        input: &[f32],
        input_shapes: &[&Shape],
    ) -> Result<Vec<f32>, TensorError> {
        require_f32_ports::<K>()?;
        gate::<K>(input_shapes, &[input.len()], 1)?;
        Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
        let in_buf = self.backend.create_buffer(input).map_err(gpu_err::<K>)?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(1)
            .map_err(gpu_err::<K>)?;
        let buffers: Vec<&GpuBuffer> = vec![&in_buf, &out_buf];
        // One sequential invocation: the parenthesization IS the
        // contract (FixedOrderReduction).
        self.backend
            .dispatch(pipeline, &buffers, (1, 1, 1))
            .map_err(gpu_err::<K>)?;
        let result: Vec<f32> = self.backend.read_buffer(&out_buf).map_err(gpu_err::<K>)?;
        Ok(result)
    }

    /// Execute `rowwise_l2_norm` (T2b): `[R, C] -> [R]`, one invocation
    /// per row, sequential accumulation within the row. A zero-row input
    /// (`[0, C]`) writes nothing — empty result, no dispatch.
    pub fn execute_rowwise_l2_norm(
        &mut self,
        input: &[f32],
        shape: &Shape,
    ) -> Result<Vec<f32>, TensorError> {
        type K = crate::ops::l1::RowwiseL2Norm;
        require_f32_ports::<K>()?;
        let out_shape = gate::<K>(&[shape], &[input.len()], shape.dims()[0] as usize)?;
        if out_shape.element_count() == 0 {
            return Ok(Vec::new());
        }
        let params = K::params(&[shape])?;
        let params_buf = self.backend.create_buffer(&params).map_err(gpu_err::<K>)?;
        let out_len = usize::try_from(out_shape.element_count()).expect("gate caps at u32::MAX");
        Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
        let in_buf = self.backend.create_buffer(input).map_err(gpu_err::<K>)?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(out_len)
            .map_err(gpu_err::<K>)?;
        let buffers: Vec<&GpuBuffer> = vec![&in_buf, &out_buf, &params_buf];
        self.backend
            .dispatch(pipeline, &buffers, workgroup_count(out_len))
            .map_err(gpu_err::<K>)?;
        let result: Vec<f32> = self.backend.read_buffer(&out_buf).map_err(gpu_err::<K>)?;
        Ok(result)
    }

    /// Execute `masked_mean` (T2b): `[B,S,D] + [B,S] -> [B,D]`, one
    /// invocation per `(batch, column)` pair, both accumulation loops
    /// strictly left-to-right over the token axis. A zero-width output
    /// writes nothing — empty result, no dispatch (the CPU gate).
    pub fn execute_masked_mean(
        &mut self,
        values: &[f32],
        mask: &[f32],
        input_shapes: &[&Shape],
    ) -> Result<Vec<f32>, TensorError> {
        type K = crate::ops::l1::MaskedMean;
        require_f32_ports::<K>()?;
        let out_shape = gate::<K>(
            input_shapes,
            &[values.len(), mask.len()],
            (input_shapes[0].dims()[0] * input_shapes[0].dims()[2]) as usize,
        )?;
        if out_shape.element_count() == 0 {
            return Ok(Vec::new());
        }
        let params = K::params(input_shapes)?;
        let params_buf = self.backend.create_buffer(&params).map_err(gpu_err::<K>)?;
        let out_len = usize::try_from(out_shape.element_count()).expect("gate caps at u32::MAX");
        Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
        let values_buf = self.backend.create_buffer(values).map_err(gpu_err::<K>)?;
        let mask_buf = self.backend.create_buffer(mask).map_err(gpu_err::<K>)?;
        let out_buf = self
            .backend
            .create_buffer_uninit::<f32>(out_len)
            .map_err(gpu_err::<K>)?;
        let buffers: Vec<&GpuBuffer> = vec![&values_buf, &mask_buf, &out_buf, &params_buf];
        self.backend
            .dispatch(pipeline, &buffers, workgroup_count(out_len))
            .map_err(gpu_err::<K>)?;
        let result: Vec<f32> = self.backend.read_buffer(&out_buf).map_err(gpu_err::<K>)?;
        Ok(result)
    }
}

fn workgroup_count(n: usize) -> (u32, u32, u32) {
    let n = u32::try_from(n).unwrap_or(u32::MAX);
    (n.div_ceil(256), 1, 1)
}

// ── GpuKernel implementations for the T2a op set ─────────────────────

use crate::layout::Shape;

impl GpuKernel for crate::ops::l3::Relu {
    const SOURCE: &'static str = kernels::RELU;
    const ENTRY: &'static str = "relu";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l0::Add {
    const SOURCE: &'static str = kernels::ADD;
    const ENTRY: &'static str = "add";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l0::Mul {
    const SOURCE: &'static str = kernels::MUL;
    const ENTRY: &'static str = "mul";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l0::Gather {
    const SOURCE: &'static str = kernels::GATHER;
    const ENTRY: &'static str = "gather";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l1::Sum {
    const SOURCE: &'static str = kernels::SUM;
    const ENTRY: &'static str = "sum";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l1::Max {
    const SOURCE: &'static str = kernels::MAX;
    const ENTRY: &'static str = "max";

    fn params(_input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        Ok(Vec::new())
    }
}

impl GpuKernel for crate::ops::l1::RowwiseL2Norm {
    const SOURCE: &'static str = kernels::ROWWISE_L2_NORM;
    const ENTRY: &'static str = "rowwise_l2_norm";

    fn params(input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        let Some(first) = input_shapes.first() else {
            return Err(TensorError::GpuDispatch {
                op: Self::NAME,
                message: "params called without input shapes".to_string(),
            });
        };
        let [_, cols] = first.dims() else {
            return Err(TensorError::RankMismatch {
                op: Self::NAME,
                port: 0,
                expected: "rank 2",
                got: first.rank(),
            });
        };
        Ok(vec![*cols])
    }
}

impl GpuKernel for crate::ops::l1::MaskedMean {
    const SOURCE: &'static str = kernels::MASKED_MEAN;
    const ENTRY: &'static str = "masked_mean";

    fn params(input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        let Some(first) = input_shapes.first() else {
            return Err(TensorError::GpuDispatch {
                op: Self::NAME,
                message: "params called without input shapes".to_string(),
            });
        };
        let [_, tokens, width] = first.dims() else {
            return Err(TensorError::RankMismatch {
                op: Self::NAME,
                port: 0,
                expected: "rank 3",
                got: first.rank(),
            });
        };
        Ok(vec![*tokens, *width])
    }
}

impl GpuKernel for crate::ops::l2::Transpose2D {
    const SOURCE: &'static str = kernels::TRANSPOSE_2D;
    const ENTRY: &'static str = "transpose_2d";

    fn params(input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        // The input's row count R: not recoverable from binding lengths
        // (input and output have the same element count). The executor
        // calls this only after the gate, but the trait method is
        // public — reject instead of indexing (r3 P1).
        let Some(first) = input_shapes.first() else {
            return Err(TensorError::GpuDispatch {
                op: Self::NAME,
                message: "params called without input shapes".to_string(),
            });
        };
        let [rows, _cols] = first.dims() else {
            return Err(TensorError::RankMismatch {
                op: Self::NAME,
                port: 0,
                expected: "rank 2",
                got: input_shapes[0].rank(),
            });
        };
        Ok(vec![*rows])
    }
}
