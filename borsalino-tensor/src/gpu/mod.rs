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
            .dispatch(&pipeline, &buffers, workgroups)
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
            .dispatch(&pipeline, &buffers, workgroup_count(output_len))
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

/// 1-D dispatch grid for `n` elements at 256 threads per group.
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
