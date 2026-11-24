// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! GPU execution of the tensor op set through Borsalino (T2a).
//!
//! [`TensorGpu`] wraps any [`GpuBackend`] and executes contracted ops as
//! WGSL kernels: compile-once-per-op (cached), operands bound in port
//! order with the output last and any op-params buffers after it, one
//! dispatch sized from the output element count. The CPU reference
//! ([`crate::cpu`]) remains the equivalence oracle — every kernel test
//! in this crate runs both and compares per
//! [`DeterminismClass`](crate::op::DeterminismClass).
//!
//! The layer is thin by design: buffer lifetimes are per-call (created,
//! dispatched, read back), matching the T2a verification harness. Buffer
//! reuse, manifest-driven zero-copy weight binding, and the Katakuri
//! checkpoint path land with the consumer-facing execution API (T2b+).
//!
//! # Subnormal policy (FTZ/DAZ) — the GPU equivalence seam
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
}

impl<B: GpuBackend> TensorGpu<B> {
    /// Wrap a backend (the caller owns backend selection and strategy).
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            pipelines: HashMap::new(),
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

    /// Execute a unary op over flat f32 operands: binds `input` then the
    /// `output` buffer, dispatches, reads the output back.
    ///
    /// Shape validation is the caller's gate (the contract tests run
    /// `validate` first — the executor does not re-derive it from flat
    /// lengths, which is why the public execution API lands in T2b with
    /// typed operands).
    pub fn execute_unary<K: GpuKernel>(
        &mut self,
        input: &[f32],
        output_len: usize,
        input_shapes: &[&crate::layout::Shape],
    ) -> Result<Vec<f32>, TensorError> {
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
    pub fn execute_binary<K: GpuKernel>(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
        output_len: usize,
        input_shapes: &[&crate::layout::Shape],
    ) -> Result<Vec<f32>, TensorError> {
        require_f32_ports::<K>()?;
        let out_shape = gate::<K>(input_shapes, &[lhs.len(), rhs.len()], output_len)?;
        require_elementwise::<K>(input_shapes, &out_shape)?;
        Self::ensure_pipeline::<K>(&self.backend, &mut self.pipelines)?;
        let pipeline = self.pipelines.get(K::NAME).expect("just ensured");
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

impl GpuKernel for crate::ops::l2::Transpose2D {
    const SOURCE: &'static str = kernels::TRANSPOSE_2D;
    const ENTRY: &'static str = "transpose_2d";

    fn params(input_shapes: &[&Shape]) -> Result<Vec<u32>, TensorError> {
        // The input's row count R: not recoverable from binding lengths
        // (input and output have the same element count).
        let [rows, _cols] = input_shapes[0].dims() else {
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
