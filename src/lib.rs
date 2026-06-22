// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! # Borsalino — Thin GPU Compute Abstraction
//!
//! > One trait, two backends, zero ceremony.
//!
//! Borsalino provides a minimal synchronous GPU compute interface
//! for the Industrial Algebra ecosystem. Write WGSL compute kernels,
//! dispatch them on Metal or Vulkan hardware, read the results
//! back — no bind groups, no pipeline layouts, no descriptor sets.
//!
//! ## Design
//!
//! - **Synchronous**: `dispatch()` blocks until the GPU finishes.
//!   No async runtime, no callback hell, no completion handlers.
//! - **WGSL-first**: Kernels are authored in WGSL (WebGPU Shading Language).
//!   The Metal backend translates to MSL via naga; the Vulkan backend
//!   translates to SPIR-V via naga.
//! - **Minimal surface area**: Four operations — create buffers, compile
//!   shaders, dispatch, read results. That's it.
//! - **Zero-cost abstraction**: No allocation, no validation, no safety
//!   checks beyond what the GPU driver provides.
//!
//! ## Quick Start
//!
//! ```ignore
//! use borsalino::GpuBackend;
//!
//! // WGSL compute kernel
//! let wgsl = r#"
//!     @group(0) @binding(0) var<storage, read> input: array<f32>;
//!     @group(0) @binding(1) var<storage, read_write> output: array<f32>;
//!
//!     @compute @workgroup_size(256)
//!     fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
//!         let i = gid.x;
//!         output[i] = input[i] + 1.0;
//!     }
//! "#;
//!
//! let mut gpu = borsalino::init()?;
//! let pipeline = gpu.compile("add_one", wgsl)?;
//! let input = gpu.create_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
//! let output = gpu.create_buffer_uninit::<f32>(4)?;
//! gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1))?;
//! let result = gpu.read_buffer(&output)?;
//! assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
//! # Ok::<(), borsalino::GpuError>(())
//! ```
//!
//! ## Backends
//!
//! | Feature    | Platform       | Status     |
//! |------------|----------------|------------|
//! | `metal`    | macOS          | ✅ Active  |
//! | `vulkan`   | Linux, Windows | ✅ Active  |
//!
//! The Metal backend requires no external dependencies beyond naga —
//! it calls the Metal framework directly via `objc_msgSend`.
//! The Vulkan backend uses `ash` for raw Vulkan FFI.
//!
//! ## Safety
//!
//! The GPU driver layer uses `unsafe` for FFI. The public API is
//! safe — buffer bounds are checked, shader compilation errors
//! are surfaced as `GpuError`, and dispatch parameters are validated
//! before hitting the driver.

#![warn(missing_docs)]
#![warn(clippy::all)]

mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(feature = "verify")]
pub mod verify;
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
mod vulkan;

pub use error::{GpuError, Result};

// ── Memory strategy ───────────────────────────────────────────────

/// Memory allocation strategy for GPU buffers.
///
/// Controls whether buffer data lives in unified memory (shared with CPU)
/// or dedicated GPU memory (VRAM), with automatic staging transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Let the backend choose based on hardware capabilities.
    /// Discrete GPUs get device-local memory; integrated/unified get host-visible.
    Auto,
    /// Force host-visible, host-coherent memory (unified memory systems).
    /// Best for Apple Silicon, AMD APUs, and GB10.
    Unified,
    /// Force device-local memory with staging transfers (discrete GPUs).
    /// Best for NVIDIA RTX, AMD RDNA, Intel Arc.
    DeviceLocal,
}

// ── Opaque handle types ───────────────────────────────────────────

use std::ffi::c_void;

/// Handle to a compiled compute pipeline.
///
/// Created by [`GpuBackend::compile`] from WGSL source. Wraps a
/// backend-specific pipeline object (Metal `MTLComputePipelineState`,
/// Vulkan `VkPipeline`). Opaque to callers.
///
/// Pipelines are cheap to clone — the underlying GPU objects
/// are reference-counted by the backend runtime.
///
/// # Drop behaviour
///
/// When dropped, the pipeline releases its GPU resources via
/// the backend-specific drop function stored at construction time.
pub struct ComputePipeline {
    pub(crate) raw: *mut c_void,
    pub(crate) drop_fn: fn(*mut c_void),
}

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("raw", &self.raw)
            .finish()
    }
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        (self.drop_fn)(self.raw);
    }
}

/// Handle to a GPU buffer.
///
/// Created by [`GpuBackend::create_buffer`] or
/// [`GpuBackend::create_buffer_uninit`]. Wraps a backend-specific
/// buffer object (Metal `MTLBuffer`, Vulkan `VkBuffer`).
///
/// # Drop behaviour
///
/// When dropped, the buffer releases its GPU resources.
pub struct GpuBuffer {
    pub(crate) raw: *mut c_void,
    pub(crate) len: usize,
    pub(crate) element_size: usize,
    pub(crate) drop_fn: fn(*mut c_void),
    #[allow(dead_code)] // Used via dynamic dispatch in backend read_buffer impls
    pub(crate) contents_fn: fn(*mut c_void) -> *const c_void,
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("raw", &self.raw)
            .field("len", &self.len)
            .field("element_size", &self.element_size)
            .finish()
    }
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        (self.drop_fn)(self.raw);
    }
}

/// Handle to an in-flight asynchronous dispatch.
///
/// Created by [`GpuBackend::dispatch_async`]. Call [`Pulse::wait`]
/// to block until the GPU completes the dispatch. Multiple pulses
/// can be in flight simultaneously.
///
/// # Drop behaviour
///
/// When dropped, blocks until the dispatch completes (implicit join).
pub struct Pulse {
    pub(crate) raw: *mut c_void,
    pub(crate) wait_fn: fn(*mut c_void),
    pub(crate) drop_fn: fn(*mut c_void),
}

impl std::fmt::Debug for Pulse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pulse").field("raw", &self.raw).finish()
    }
}

unsafe impl Send for Pulse {}
unsafe impl Sync for Pulse {}

impl Drop for Pulse {
    fn drop(&mut self) {
        // Implicit join: wait for the dispatch to complete before cleanup
        (self.wait_fn)(self.raw);
        (self.drop_fn)(self.raw);
    }
}

impl Pulse {
    /// Block until the dispatch completes.
    pub fn wait(&self) {
        (self.wait_fn)(self.raw);
    }
}

// ── Trait ─────────────────────────────────────────────────────────

/// Backend-agnostic GPU compute interface.
///
/// Each backend implements this trait. Callers use [`init`] to
/// get the right backend for the current platform.
///
/// # Buffer binding
///
/// Buffers are bound by position in the `dispatch` call's `buffers`
/// slice — `buffers[0]` maps to `@group(0) @binding(0)` in WGSL,
/// `buffers[1]` maps to `@group(0) @binding(1)`, etc.
///
/// # Thread groups
///
/// The `workgroups` parameter is `(groups_x, groups_y, groups_z)`.
/// Each workgroup contains `threads_per_group` threads (default 256).
/// Use [`dispatch_ex`](GpuBackend::dispatch_ex) to specify a custom
/// threadgroup size.
pub trait GpuBackend: Sized {
    /// Initialise the GPU backend.
    ///
    /// On macOS Metal, returns the system default Metal device.
    /// On Vulkan, returns the first available discrete GPU.
    /// Memory strategy is auto-detected from hardware capabilities.
    fn init() -> Result<Self>;

    /// Initialise with an explicit memory strategy.
    ///
    /// Overrides the automatic detection. Use [`MemoryStrategy::DeviceLocal`]
    /// to force VRAM allocation on discrete GPUs, or [`MemoryStrategy::Unified`]
    /// to force host-visible memory even when device-local is available.
    fn init_with_strategy(_strategy: MemoryStrategy) -> Result<Self> {
        Self::init()
    }

    /// Compile WGSL source into a compute pipeline.
    ///
    /// The `entry_point` is the kernel function name in the WGSL source.
    /// Compilation happens at call time — cache the [`ComputePipeline`]
    /// if dispatching repeatedly.
    fn compile(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline>;

    /// Compile with disk caching.
    ///
    /// On first call, behaves identically to [`compile`](GpuBackend::compile)
    /// and caches the compiled shader to disk. On subsequent calls with the
    /// same WGSL source, skips naga translation and loads the cached binary
    /// (SPIR-V on Vulkan, MSL on Metal).
    ///
    /// Cache location: `~/.cache/borsalino/` (respects `XDG_CACHE_HOME` on Linux).
    ///
    /// The default implementation delegates to [`compile`](GpuBackend::compile)
    /// without caching. Backends may override.
    fn compile_cached(
        &self,
        entry_point: &str,
        wgsl_source: &str,
    ) -> Result<ComputePipeline> {
        self.compile(entry_point, wgsl_source)
    }

    /// Allocate a GPU buffer and upload initial data.
    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer>;

    /// Allocate an uninitialised GPU buffer of `len` elements.
    fn create_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer>;

    /// Allocate a device-local GPU buffer and upload initial data.
    ///
    /// Unlike [`create_buffer`](GpuBackend::create_buffer), this buffer is
    /// allocated in device-local memory (VRAM on discrete GPUs) and persists
    /// across dispatches without CPU readback overhead. Use for model weights,
    /// activations, and other GPU-resident data.
    ///
    /// On unified memory systems (Apple Silicon, AMD APU, GB10), this is
    /// identical to [`create_buffer`](GpuBackend::create_buffer).
    ///
    /// Default implementation delegates to [`create_buffer`](GpuBackend::create_buffer).
    fn create_device_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        self.create_buffer(data)
    }

    /// Allocate an uninitialised device-local GPU buffer.
    ///
    /// Default implementation delegates to
    /// [`create_buffer_uninit`](GpuBackend::create_buffer_uninit).
    fn create_device_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        self.create_buffer_uninit::<T>(len)
    }

    /// Dispatch a compute pipeline across `workgroups` thread groups.
    ///
    /// Each workgroup contains 256 threads (1D layout) unless overridden
    /// via [`dispatch_ex`](GpuBackend::dispatch_ex).
    ///
    /// Buffers are bound to the kernel in slice order: `buffers[0]`
    /// → `@group(0) @binding(0)`, `buffers[1]` → `@group(0) @binding(1)`, etc.
    ///
    /// Blocks until the GPU completes the dispatch and the results
    /// are visible to the CPU.
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Dispatch with explicit threadgroup size.
    ///
    /// Like [`dispatch`](GpuBackend::dispatch), but each workgroup contains
    /// `threads_per_group` threads rather than the default 256.
    fn dispatch_ex(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
        threads_per_group: (u32, u32, u32),
    ) -> Result<()>;

    /// Read the contents of a GPU buffer back to the CPU.
    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>>;

    /// Return a GPU-side timestamp in nanoseconds for profiling.
    ///
    /// Call before and after a dispatch to measure GPU execution time
    /// independent of CPU-side dispatch overhead:
    ///
    /// ```ignore
    /// let t0 = gpu.timestamp()?;
    /// gpu.dispatch(&pipeline, &buffers, (wgs, 1, 1))?;
    /// let elapsed_ns = gpu.timestamp()? - t0;
    /// ```
    fn timestamp(&self) -> Result<u64>;

    /// Dispatch multiple kernels in a single command buffer.
    ///
    /// Amortises command-buffer creation and GPU-sync overhead across
    /// multiple dispatches. All dispatches execute before any results
    /// are visible to the CPU.
    ///
    /// The default implementation calls [`dispatch`](GpuBackend::dispatch)
    /// for each spec. Backends may override with a batched implementation.
    fn dispatch_many(&self, dispatches: &[DispatchSpec<'_>]) -> Result<()> {
        for spec in dispatches {
            self.dispatch_ex(
                spec.pipeline,
                spec.buffers,
                spec.workgroups,
                spec.threads_per_group,
            )?;
        }
        Ok(())
    }

    /// Dispatch a compute pipeline asynchronously.
    ///
    /// Returns a [`Pulse`] handle that can be waited on later.
    /// Unlike [`dispatch`](GpuBackend::dispatch), this does not block
    /// the caller. Multiple async dispatches can be in flight simultaneously.
    ///
    /// The default implementation calls [`dispatch`](GpuBackend::dispatch)
    /// and wraps the result in a no-op pulse. Backends may override with
    /// true async behaviour.
    fn dispatch_async(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<Pulse> {
        self.dispatch(pipeline, buffers, workgroups)?;
        Ok(Pulse {
            raw: std::ptr::null_mut(),
            wait_fn: |_| {},
            drop_fn: |_| {},
        })
    }
}

/// Spec for a single dispatch within [`GpuBackend::dispatch_many`].
#[derive(Clone, Copy)]
pub struct DispatchSpec<'a> {
    /// The compiled compute pipeline.
    pub pipeline: &'a ComputePipeline,
    /// Buffers to bind, in `@group(0) @binding(N)` order.
    pub buffers: &'a [&'a GpuBuffer],
    /// Number of threadgroups in (x, y, z).
    pub workgroups: (u32, u32, u32),
    /// Threads per threadgroup (default: 256, 1, 1).
    pub threads_per_group: (u32, u32, u32),
}

// ── Stub backend (compile-time sentinel) ──────────────────────────

/// Stub backend — no GPU backend compiled for this target.
#[cfg(not(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "vulkan", not(target_os = "macos"))
)))]
pub struct NoBackendStub;

#[cfg(not(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "vulkan", not(target_os = "macos"))
)))]
impl GpuBackend for NoBackendStub {
    fn init() -> Result<Self> {
        Err(GpuError::NoBackend)
    }
    fn compile(&self, _entry: &str, _wgsl: &str) -> Result<ComputePipeline> {
        Err(GpuError::NoBackend)
    }
    fn create_buffer<T: bytemuck::Pod>(&self, _data: &[T]) -> Result<GpuBuffer> {
        Err(GpuError::NoBackend)
    }
    fn create_buffer_uninit<T: bytemuck::Pod>(&self, _len: usize) -> Result<GpuBuffer> {
        Err(GpuError::NoBackend)
    }
    fn dispatch(
        &self,
        _pipeline: &ComputePipeline,
        _buffers: &[&GpuBuffer],
        _workgroups: (u32, u32, u32),
    ) -> Result<()> {
        Err(GpuError::NoBackend)
    }
    fn dispatch_ex(
        &self,
        _pipeline: &ComputePipeline,
        _buffers: &[&GpuBuffer],
        _workgroups: (u32, u32, u32),
        _threads_per_group: (u32, u32, u32),
    ) -> Result<()> {
        Err(GpuError::NoBackend)
    }
    fn read_buffer<T: bytemuck::Pod>(&self, _buffer: &GpuBuffer) -> Result<Vec<T>> {
        Err(GpuError::NoBackend)
    }
    fn timestamp(&self) -> Result<u64> {
        Err(GpuError::NoBackend)
    }
}

// ── Top-level initialiser ─────────────────────────────────────────

/// Initialise the best available GPU backend for the current platform.
///
/// - macOS: Metal (requires `metal` feature, enabled by default on macOS)
/// - Linux/Windows: Vulkan (requires `vulkan` feature)
/// - Otherwise: returns [`GpuError::NoBackend`]
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn init() -> Result<metal::MetalBackend> {
    metal::MetalBackend::init()
}

/// Initialise the Vulkan backend (Linux/Windows).
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
pub fn init() -> Result<vulkan::VulkanBackend> {
    vulkan::VulkanBackend::init()
}

/// Initialise with device-local memory (forces VRAM allocation on discrete GPUs).
///
/// Equivalent to `init_with_strategy(MemoryStrategy::DeviceLocal)`.
/// Use when you know you're on discrete hardware and want maximum throughput.
#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
pub fn init_device_local() -> Result<vulkan::VulkanBackend> {
    vulkan::VulkanBackend::init_with_strategy(MemoryStrategy::DeviceLocal)
}

/// Stub backend — returned by `init()` when no GPU backend is available.
/// Exists solely to give `Result<impl GpuBackend>` a concrete type in
/// the fallback compilation path.
#[cfg(not(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "vulkan", not(target_os = "macos"))
)))]
pub fn init() -> Result<NoBackendStub> {
    Err(GpuError::NoBackend)
}
