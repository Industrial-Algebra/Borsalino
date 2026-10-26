// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU backend for Apple Silicon, delegating device/queue/buffer
//! ownership to [`zunesha::metal::MetalDevice`][zdm] (Phase 3 of the
//! staged Zunesha migration — mirrors the Vulkan backend's shape).
//!
//! Borsalino keeps the consumer half: naga WGSL→MSL translation, the
//! `_mslBufferSizes` ABI, pipeline compilation, dispatch encoding, and
//! epoch accounting around its own submissions. Zunesha owns the
//! MTLDevice, MTLCommandQueue, and every MTLBuffer (allocation,
//! storage-mode strategy, staging transfers, teardown quiescence).
//!
//! ## Raw surfaces used (Zunesha R4 contract)
//!
//! - [`MetalDevice::raw_device`] — MTLDevice for library/pipeline
//!   creation (retained borrow; never released here),
//! - [`MetalDevice::command_queue`] — the substrate's MTLCommandQueue
//!   (consumers dispatch on it; never create their own),
//! - [`MetalDevice::raw_buffer`] — the MTLBuffer behind a
//!   `zunesha::Buffer` for dispatch-time binding.
//!
//! Uses the `objc` crate for `msg_send!` dispatch on ARM64.
//!
//! ## Borsalino-owned Metal objects
//!
//! Two classes remain this backend's to manage (explicitly NOT the
//! substrate's): the `MTLComputePipelineState` products (owned +1 from
//! `new…`, released in `drop_pipeline`) and the transient
//! `_mslBufferSizes` constants (owned +1 from `newBufferWithBytes` in
//! `bind_buffer_sizes`, released after `endEncoding` — Metal retains
//! resources referenced by encoded commands, so the early release is
//! safe even for in-flight async command buffers).
//!
//! [zdm]: https://docs.rs/zunesha

use std::ffi::c_void;

use naga::back::msl;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use zunesha::Device as _;

use crate::{ComputePipeline, GpuBackend, GpuBuffer, GpuError, MemoryStrategy, Pulse, Result};

/// Map Borsalino's strategy onto the substrate's (identical shapes —
/// the strategy vocabulary is shared across backends).
fn to_zunesha_strategy(s: MemoryStrategy) -> zunesha::MemoryStrategy {
    match s {
        MemoryStrategy::Auto => zunesha::MemoryStrategy::Auto,
        MemoryStrategy::Unified => zunesha::MemoryStrategy::Unified,
        MemoryStrategy::DeviceLocal => zunesha::MemoryStrategy::DeviceLocal,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

/// Cast a raw Metal object pointer to `*const Object` for `msg_send!`.
unsafe fn obj(ptr: *mut c_void) -> *const Object {
    ptr as *const Object
}

/// Create an NSString from a Rust string. Returns an autoreleased
/// object — callers must NOT release it; the autorelease pool handles it.
unsafe fn nsstring(s: &str) -> *mut c_void {
    let c_str = std::ffi::CString::new(s).expect("string contains null byte");
    msg_send![class!(NSString), stringWithUTF8String: c_str.as_ptr()]
}

/// Read an NSString into a Rust String.
unsafe fn nsstring_read(ns: *mut c_void) -> String {
    // Safety: `ns` is an NSString (or null-checked before use).
    let utf8: *const std::ffi::c_char = unsafe { msg_send![obj(ns), UTF8String] };
    if utf8.is_null() {
        return "(null)".into();
    }
    unsafe { std::ffi::CStr::from_ptr(utf8) }
        .to_string_lossy()
        .into_owned()
}

// ═══════════════════════════════════════════════════════════════════
// Internal Metal handles
// ═══════════════════════════════════════════════════════════════════

/// Internal state for a Metal compute pipeline, stored behind the opaque
/// `ComputePipeline.raw` pointer: the owned `MTLComputePipelineState` and
/// the naga buffer-sizes layout (see [`sizes_buffer_bindings`]).
struct MetalPipelineInner {
    /// Owned (+1 from `newComputePipelineState…`) pipeline state.
    pipeline: *mut c_void,
    /// Binding numbers of runtime-sized-array globals, in module
    /// declaration order — the field order of naga's synthesized
    /// `_mslBufferSizes` constant. Empty for kernels without runtime
    /// arrays (naga emits no sizes constant for those).
    sizes_bindings: Vec<u32>,
}

/// The owned `MTLComputePipelineState` behind a [`ComputePipeline`].
fn mtl_pipeline(raw: *mut c_void) -> *mut c_void {
    debug_assert!(!raw.is_null());
    // Safety: `raw` was produced by `Box::into_raw::<MetalPipelineInner>`
    // and remains valid while the pipeline is alive.
    unsafe { (*(raw as *const MetalPipelineInner)).pipeline }
}

fn drop_pipeline(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            // Safety: reclaim the box, release the owned state, drop layout.
            let inner = Box::from_raw(raw as *mut MetalPipelineInner);
            let _: () = msg_send![obj(inner.pipeline), release];
        }
    }
}

// ── MSL post-processing ──────────────────────────────────────────

/// Post-process naga-generated MSL to fix Metal 3 compatibility.
/// Naga emits `device type_N const&` / `device type_N&` (references to
/// fixed-size arrays), but Metal 3's pipeline creation crashes with this
/// syntax. Converts to pointer syntax and strips unused structs.
/// First-line marker embedded in cached MSL carrying the buffer-sizes
/// layout (`// borsalino:sizes 0,1,2` — binding numbers in module
/// declaration order). Compilation caches without it are pre-header and
/// treated as misses.
const SIZES_HEADER_PREFIX: &str = "// borsalino:sizes ";

fn make_sizes_header(sizes_bindings: &[u32]) -> String {
    let list = sizes_bindings
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{SIZES_HEADER_PREFIX}{list}")
}

/// Split a cached MSL into `(sizes_bindings, body)`. Returns `None` when
/// the header is absent or malformed (cache miss).
fn split_sizes_header(cached: &str) -> Option<(Vec<u32>, &str)> {
    let first_line_end = cached.find('\n')?;
    let (first, rest) = cached.split_at(first_line_end);
    let rest = &rest[1..]; // drop the newline
    let list = first.strip_prefix(SIZES_HEADER_PREFIX)?;
    if list.is_empty() {
        return Some((Vec::new(), rest));
    }
    let bindings = list
        .split(',')
        .map(|n| n.trim().parse::<u32>().ok())
        .collect::<Option<Vec<u32>>>()?;
    Some((bindings, rest))
}

/// Binding numbers of module globals carrying runtime-sized arrays, in
/// **module declaration order** — this is the field order of naga's
/// synthesized `struct _mslBufferSizes` (one `uint sizeN;` per such
/// global; `N` is the module-global index, and the fields are laid out in
/// module order — verified against naga 27's writer). Kernels translated
/// from WGSL `arrayLength` dereference that constant, so dispatch must
/// bind one **byte** size per entry, in THIS order, at the sizes slot.
///
/// Mirrors naga's `needs_array_length`: a global qualifies when its type
/// is a dynamic-sized array, or a struct whose LAST member is one (the
/// storage-buffer block shape).
fn sizes_buffer_bindings(module: &naga::Module) -> Vec<u32> {
    module
        .global_variables
        .iter()
        .filter(|(_, var)| match module.types[var.ty].inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Dynamic,
                ..
            } => true,
            naga::TypeInner::Struct { ref members, .. } => members.last().is_some_and(|m| {
                matches!(
                    module.types[m.ty].inner,
                    naga::TypeInner::Array {
                        size: naga::ArraySize::Dynamic,
                        ..
                    }
                )
            }),
            _ => false,
        })
        .filter_map(|(_, var)| var.binding.map(|b| b.binding))
        .collect()
}

fn naga_msl_fixup(msl: &str) -> String {
    let mut out = String::with_capacity(msl.len());

    for line in msl.lines() {
        let trimmed = line.trim();

        // Skip `typedef float type_N[1];` lines
        if trimmed.starts_with("typedef ") && trimmed.contains("type_") && trimmed.ends_with("];") {
            continue;
        }

        // Skip empty struct declarations like `struct add_oneInput {};`
        if trimmed.starts_with("struct ") && trimmed.ends_with(" {};") {
            continue;
        }

        // NOTE: the `_mslBufferSizes` struct and its `_buffer_sizes`
        // kernel parameter are KEPT — naga emits them unconditionally
        // (slot 30, per `sizes_buffer` in `compile`), and kernels using
        // `arrayLength` dereference them. The dispatch paths bind the
        // byte-size constant at slot 30 (`bind_buffer_sizes`). Stripping
        // them used to corrupt any arrayLength kernel's signature (the
        // macOS CI failure on the first cut of the batched-kernel guard).

        // Fix `metal::uint3` → `uint3`
        let line = line.replace("metal::uint3", "uint3");

        // Fix `device type_N const& name` → `device const float* name`
        if let Some(fixed) = fix_device_line(&line, false) {
            out.push_str(&fixed);
            out.push('\n');
            continue;
        }

        // Fix `device type_N& name` → `device float* name`
        if let Some(fixed) = fix_device_line(&line, true) {
            out.push_str(&fixed);
            out.push('\n');
            continue;
        }

        out.push_str(&line);
        out.push('\n');
    }

    out
}

fn fix_device_line(line: &str, mutable: bool) -> Option<String> {
    let type_start = line.find("device type_")?;
    let after_device = &line[type_start..];

    // Check if this line matches the requested mutability
    let has_const = after_device.contains("const&");
    if mutable && has_const {
        return None; // Mutable pass: skip lines with const&
    }
    if !mutable && !has_const {
        return None; // Const pass: skip lines without const&
    }

    let idx_after_type = after_device.find('&')? + 1;
    let rest = after_device[idx_after_type..].trim_start();
    let name_end = rest
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    let name = &rest[..name_end];
    let suffix = &rest[name_end..];
    let prefix = if mutable {
        "device float* "
    } else {
        "device const float* "
    };
    let before = &line[..type_start];
    Some(format!("{before}{prefix}{name}{suffix}"))
}

// ═══════════════════════════════════════════════════════════════════
// MetalBackend
// ═══════════════════════════════════════════════════════════════════

/// Metal GPU backend for Apple Silicon, on the Zunesha substrate.
pub struct MetalBackend {
    /// The substrate device (owns MTLDevice + MTLCommandQueue + buffers).
    z: std::sync::Arc<zunesha::metal::MetalDevice>,
    /// Epoch tracker for GC safety — counts Borsalino's in-flight
    /// dispatches. Shared with async `Pulse`s so they cannot outlive it
    /// (review finding: a raw `&'a GpuEpochTracker` in `MetalPulseInner`
    /// could dangle). Consumer-side accounting, mirroring Vulkan — the
    /// substrate's own tracker is not wired to consumer dispatches.
    epoch: std::sync::Arc<crate::epoch::GpuEpochTracker>,
}

/// Contents function for substrate-backed buffers — returns the
/// [`SUBSTRATE_BUFFER_TAG`] sentinel instead of a raw `contents()`
/// pointer (reads go through the substrate's `read_buffer`, which needs
/// the device — and a Private buffer's `contents()` would be stale
/// staging anyway). Phase 3 removed the last Borsalino-allocated buffer
/// kind (every buffer is substrate-backed now); the tag survives as the
/// public `contents()` value for substrate buffers and as the
/// delegation marker the Phase-3 tests assert (mirroring Vulkan).
const SUBSTRATE_BUFFER_TAG: *const c_void = std::ptr::without_provenance::<c_void>(1);

fn contents_metal_zunesha_buffer(_raw: *mut c_void) -> *const c_void {
    SUBSTRATE_BUFFER_TAG
}

/// Drop function for substrate-backed buffers — reclaims the box; the
/// `zunesha::Buffer` inside drops and the substrate releases the
/// MTLBuffer (and any staging twin) under its own teardown discipline.
fn drop_metal_zunesha_buffer(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            drop(Box::from_raw(raw as *mut MetalZuneshaBufferInner));
        }
    }
}

/// Inner state for a substrate-backed [`GpuBuffer`].
struct MetalZuneshaBufferInner {
    /// The substrate buffer (owns the MTLBuffer + staging).
    zbuf: zunesha::Buffer,
    /// Device share — buffers must not outlive the substrate device
    /// (documented Zunesha invariant; mirrored from the Vulkan backend).
    _device: std::sync::Arc<zunesha::metal::MetalDevice>,
}

// Send/Sync: auto-derived — zunesha::Buffer is Send + Sync (upstream
// unsafe impls over its raw handles) and Arc<MetalDevice> likewise
// (review r2 P3: the redundant local impls are gone; safety lives in
// Zunesha's documented invariants).

impl MetalBackend {
    /// The substrate MTLBuffer behind a [`GpuBuffer`] for dispatch-time
    /// binding (Zunesha R4: consumers must not release it).
    fn buffer_handle(&self, buf: &GpuBuffer) -> *mut c_void {
        // Safety: `raw` was produced by `Box::into_raw::<MetalZuneshaBufferInner>`
        // by this backend and remains valid while the buffer is alive.
        let inner = unsafe { &*(buf.raw as *const MetalZuneshaBufferInner) };
        self.z.raw_buffer(&inner.zbuf)
    }

    /// Wrap a substrate buffer in Borsalino's opaque `GpuBuffer`.
    fn wrap_zunesha_buffer(
        &self,
        zbuf: zunesha::Buffer,
        len: usize,
        element_size: usize,
    ) -> GpuBuffer {
        GpuBuffer {
            raw: Box::into_raw(Box::new(MetalZuneshaBufferInner {
                zbuf,
                _device: std::sync::Arc::clone(&self.z),
            })) as *mut c_void,
            len,
            element_size,
            drop_fn: drop_metal_zunesha_buffer,
            contents_fn: contents_metal_zunesha_buffer,
        }
    }
}

/// Inner state for an async Metal dispatch [`Pulse`].
struct MetalPulseInner {
    cmd: *mut std::ffi::c_void,
    /// Owned share of the backend's tracker — the pulse may outlive the
    /// backend, so it must not borrow it (P1 review finding).
    epoch: std::sync::Arc<crate::epoch::GpuEpochTracker>,
    /// Tracks whether end_dispatch has been called (prevents double-decrement
    /// when wait() + drop() both fire).
    epoch_completed: std::sync::atomic::AtomicBool,
}

fn wait_metal_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { &*(raw as *const MetalPulseInner) };
        let _: () = unsafe { msg_send![obj(inner.cmd), waitUntilCompleted] };
        // Balance the begin_dispatch from dispatch_async.
        if !inner
            .epoch_completed
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            inner.epoch.end_dispatch();
        }
    }
}

fn drop_metal_pulse(raw: *mut std::ffi::c_void) {
    if !raw.is_null() {
        let inner = unsafe { Box::from_raw(raw as *mut MetalPulseInner) };
        unsafe {
            // Ensure GPU completes before releasing the command buffer.
            let _: () = msg_send![obj(inner.cmd), waitUntilCompleted];
            // Balance the begin_dispatch from dispatch_async (only if
            // wait() hasn't already done so).
            if !inner
                .epoch_completed
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                inner.epoch.end_dispatch();
            }
            let _: () = msg_send![obj(inner.cmd), release];
        }
    }
}

/// Bind the naga `_mslBufferSizes` constant (slot 30 — must match
/// `sizes_buffer` in `compile`): one byte size per runtime-array global,
/// in the pipeline's **module declaration order** (the struct's field
/// order — naga 27 assigns `uint sizeN` per runtime-array global by
/// module-global index; review round 1 P1: binding order ≠ field order,
/// and the old fixed `[u32; 8]` truncated wider layouts).
///
/// Returns `None` (binding nothing) when the pipeline has no runtime-array
/// globals — naga emits no sizes constant for those kernels.
///
/// # Ownership
///
/// `newBufferWithBytes` is a `new…` method (+1 retain). The caller must
/// `release` the returned buffer after `endEncoding` — Metal retains
/// resources referenced by encoded commands in the command buffer, so the
/// release is safe even for async dispatches whose command buffers are
/// still in flight.
unsafe fn bind_buffer_sizes(
    dev: *mut std::ffi::c_void,
    encoder: *mut std::ffi::c_void,
    pipeline_raw: *mut std::ffi::c_void,
    buffers: &[&GpuBuffer],
) -> std::result::Result<Option<*mut std::ffi::c_void>, GpuError> {
    const SIZES_SLOT: u64 = 30;

    // Safety: `pipeline_raw` was produced by `Box::into_raw::<MetalPipelineInner>`
    // and remains valid while the pipeline is alive.
    let inner = unsafe { &*(pipeline_raw as *const MetalPipelineInner) };
    let layout: &[u32] = inner.sizes_bindings.as_slice();
    if layout.is_empty() {
        // Successful absence: this kernel has no runtime-array globals, so
        // naga emitted no sizes constant and there is nothing to bind.
        return Ok(None);
    }

    // One byte size per layout entry, in declaration order. Unbound
    // bindings contribute 0 (the kernel cannot validly use them).
    let sizes: Vec<u32> = layout
        .iter()
        .map(|&binding| {
            buffers
                .get(binding as usize)
                .map(|g| (g.len * g.element_size) as u32)
                .unwrap_or(0)
        })
        .collect();

    let buf: *mut std::ffi::c_void = unsafe {
        msg_send![
            obj(dev),
            newBufferWithBytes: sizes.as_ptr() as *const std::ffi::c_void
            length: (sizes.len() * std::mem::size_of::<u32>()) as u64
            options: 0u64
        ]
    };
    if buf.is_null() {
        return Err(GpuError::DispatchFailed {
            message: "failed to allocate _mslBufferSizes constant".into(),
        });
    }
    unsafe {
        let _: () = msg_send![obj(encoder), setBuffer: buf offset: 0u64 atIndex: SIZES_SLOT];
    }
    Ok(Some(buf))
}

/// Release a sizes buffer previously returned by [`bind_buffer_sizes`].
/// `None` (kernel without a sizes constant) releases nothing.
unsafe fn release_sizes_buffer(buf: Option<*mut std::ffi::c_void>) {
    if let Some(buf) = buf {
        if !buf.is_null() {
            unsafe {
                let _: () = msg_send![obj(buf), release];
            }
        }
    }
}

impl GpuBackend for MetalBackend {
    fn init() -> Result<Self> {
        Self::init_with_strategy(MemoryStrategy::Auto)
    }

    fn init_with_strategy(strategy: MemoryStrategy) -> Result<Self> {
        let z = zunesha::metal::MetalDevice::init_with_strategy(to_zunesha_strategy(strategy))
            .map_err(|e| GpuError::InitFailed(format!("zunesha device init: {e}")))?;
        Ok(Self {
            z: std::sync::Arc::new(z),
            epoch: std::sync::Arc::new(crate::epoch::GpuEpochTracker::new()),
        })
    }

    fn compile(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        // Step 0: Translate WGSL → MSL via naga
        let module = wgsl::parse_str(wgsl_source).map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: e.emit_to_string(wgsl_source),
        })?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator
            .validate(&module)
            .map_err(|e| GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            })?;

        // Build resource binding map: @group(0) @binding(N) → buffer(N)
        let mut resources = msl::BindingMap::new();
        for (_, global) in module.global_variables.iter() {
            if let Some(ref binding) = global.binding {
                let mutable = matches!(
                    global.space,
                    naga::AddressSpace::Storage { access }
                        if access.contains(naga::StorageAccess::STORE)
                );
                resources.insert(
                    naga::ResourceBinding {
                        group: binding.group,
                        binding: binding.binding,
                    },
                    msl::BindTarget {
                        buffer: Some(binding.binding as msl::Slot),
                        texture: None,
                        sampler: None,
                        external_texture: None,
                        mutable,
                    },
                );
            }
        }

        let entry_resources = msl::EntryPointResources {
            resources,
            push_constant_buffer: None,
            sizes_buffer: Some(30u8),
        };

        let mut msl_opts = msl::Options {
            fake_missing_bindings: false,
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Unchecked,
                buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                image_load: naga::proc::BoundsCheckPolicy::Unchecked,
                ..Default::default()
            },
            ..Default::default()
        };
        msl_opts
            .per_entry_point_map
            .insert(entry_point.into(), entry_resources);

        let (mut msl_source, _) =
            msl::write_string(&module, &info, &msl_opts, &msl::PipelineOptions::default())
                .map_err(|e| GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("MSL emission failed: {e}"),
                })?;

        // The naga buffer-sizes layout must be captured before MSL
        // emission (it is a property of the module, and the dispatch
        // paths bind per-pipeline layout entries — see
        // `sizes_buffer_bindings`).
        let sizes_bindings = sizes_buffer_bindings(&module);

        // Fix naga MSL for Metal 3 compatibility
        msl_source = naga_msl_fixup(&msl_source);

        // Best-effort disk cache (completes compile_cached's design — the
        // MSL carries the sizes header so the layout round-trips).
        {
            let dir = std::env::var("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| {
                    std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                        .join(".cache")
                })
                .join("borsalino");
            let _ = std::fs::create_dir_all(&dir);
            let mut hash: u64 = 0xcbf29ce484222325;
            for &b in wgsl_source.as_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            let path = dir.join(format!("{entry_point}_{hash:016x}.msl"));
            let _ = std::fs::write(
                &path,
                format!("{}\n{}", make_sizes_header(&sizes_bindings), msl_source),
            );
        }

        let dev = self.z.raw_device();

        unsafe {
            // Step 1: MTLLibrary from source
            let ns_src = nsstring(&msl_source);
            let mut err: *mut c_void = std::ptr::null_mut();
            let library: *mut c_void = msg_send![
                dev as *const objc::runtime::Object,
                newLibraryWithSource: ns_src
                options: std::ptr::null_mut::<c_void>()
                error: &mut err
            ];

            if library.is_null() {
                let msg = if !err.is_null() {
                    let desc: *mut c_void =
                        msg_send![err as *const objc::runtime::Object, localizedDescription];
                    let s = nsstring_read(desc);
                    // (err/perr is an autoreleased out-param — the pool owns it)
                    s
                } else {
                    "unknown compilation error".into()
                };
                return Err(GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // Step 2: MTLFunction
            let ns_entry = nsstring(entry_point);
            let func: *mut c_void =
                msg_send![library as *const objc::runtime::Object, newFunctionWithName: ns_entry];

            if func.is_null() {
                let _: () = msg_send![library as *const objc::runtime::Object, release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!("function '{entry_point}' not found in compiled library"),
                });
            }

            // Step 3: MTLComputePipelineState via descriptor path
            let desc: *mut c_void = msg_send![class!(MTLComputePipelineDescriptor), new];
            let _: () = msg_send![desc as *const objc::runtime::Object, setComputeFunction: func];
            let mut perr: *mut c_void = std::ptr::null_mut();
            let pipeline: *mut c_void = msg_send![
                dev as *const objc::runtime::Object,
                newComputePipelineStateWithDescriptor: desc
                options: 0u64
                reflection: std::ptr::null_mut::<c_void>()
                error: &mut perr
            ];

            if pipeline.is_null() {
                let msg = if !perr.is_null() {
                    let desc: *mut c_void = msg_send![obj(perr), localizedDescription];
                    let s = nsstring_read(desc);
                    // (err/perr is an autoreleased out-param — the pool owns it)
                    s
                } else {
                    "unknown pipeline error".into()
                };
                // desc is +1 from `new` — the failure path used to leak
                // it (review finding). func/library are also +1.
                let _: () = msg_send![obj(desc), release];
                let _: () = msg_send![obj(func), release];
                let _: () = msg_send![obj(library), release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // Release intermediates (desc may be retained by the pipeline)
            // `new` returns a retained object — release our reference.
            let _: () = msg_send![obj(desc), release];
            let _: () = msg_send![obj(func), release];
            let _: () = msg_send![obj(library), release];

            Ok(ComputePipeline {
                raw: Box::into_raw(Box::new(MetalPipelineInner {
                    pipeline,
                    sizes_bindings,
                })) as *mut c_void,
                drop_fn: drop_pipeline,
            })
        }
    }

    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let zbuf = self
            .z
            .create_buffer(data)
            .map_err(|e| GpuError::BufferCreationFailed {
                message: format!("zunesha create_buffer: {e}"),
            })?;
        Ok(self.wrap_zunesha_buffer(zbuf, data.len(), std::mem::size_of::<T>()))
    }

    fn create_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let zbuf =
            self.z
                .create_buffer_uninit::<T>(len)
                .map_err(|e| GpuError::BufferCreationFailed {
                    message: format!("zunesha create_buffer_uninit: {e}"),
                })?;
        Ok(self.wrap_zunesha_buffer(zbuf, len, std::mem::size_of::<T>()))
    }

    /// Phase 3 (mirrors the Vulkan §5.1 contract): delegates to the
    /// substrate's forced Private + persistent-staging path — no silent
    /// Shared fallback, exactly matching the Vulkan backend's
    /// post-0.1.1 contract.
    fn create_device_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let zbuf =
            self.z
                .create_device_buffer(data)
                .map_err(|e| GpuError::BufferCreationFailed {
                    message: format!("zunesha create_device_buffer: {e}"),
                })?;
        Ok(self.wrap_zunesha_buffer(zbuf, data.len(), std::mem::size_of::<T>()))
    }

    fn create_device_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let zbuf = self.z.create_device_buffer_uninit::<T>(len).map_err(|e| {
            GpuError::BufferCreationFailed {
                message: format!("zunesha create_device_buffer_uninit: {e}"),
            }
        })?;
        Ok(self.wrap_zunesha_buffer(zbuf, len, std::mem::size_of::<T>()))
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        self.dispatch_ex(pipeline, buffers, workgroups, (256, 1, 1))
    }

    fn dispatch_ex(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
        _threads_per_group: (u32, u32, u32),
    ) -> Result<()> {
        // Scoped autorelease pool: the command buffer and encoder are
        // autoreleased (+0) objc objects; without a pool on a plain Rust
        // worker thread they would never be reclaimed (and with one owned
        // by someone else, reclaimed at that pool's whim). Draining per
        // dispatch bounds them deterministically (P2 review finding).
        objc::rc::autoreleasepool(|| unsafe {
            let cmd: *mut c_void = msg_send![obj(self.z.command_queue()), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void = msg_send![obj(cmd), computeCommandEncoder];
            if encoder.is_null() {
                // cmd is autoreleased — do not release it here.
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            // Set pipeline
            let _: () =
                msg_send![obj(encoder), setComputePipelineState: mtl_pipeline(pipeline.raw)];

            // Bind user buffers
            for (i, buf) in buffers.iter().enumerate() {
                let _: () = msg_send![
                    obj(encoder),
                    setBuffer: self.buffer_handle(buf)
                    offset: 0u64
                    atIndex: i as u64
                ];
            }

            // Bind the naga buffer-sizes constant (arrayLength support).
            // Ok(None) = kernel has no sizes constant; only allocation
            // failure is an error here.
            let sizes_buf = bind_buffer_sizes(self.z.raw_device(), encoder, pipeline.raw, buffers)?;

            // Dispatch
            let _: () = msg_send![
                obj(encoder),
                dispatchThreadgroups: (workgroups.0 as u64, workgroups.1 as u64, workgroups.2 as u64)
                threadsPerThreadgroup: (_threads_per_group.0 as u64, _threads_per_group.1 as u64, _threads_per_group.2 as u64)
            ];

            // Finish
            let _: () = msg_send![obj(encoder), endEncoding];
            // Encoded resources are retained by the command buffer — the
            // sizes buffer can go now (objc ownership rule: new → release).
            release_sizes_buffer(sizes_buf);

            self.epoch.begin_dispatch();

            let _: () = msg_send![obj(cmd), commit];
            let _: () = msg_send![obj(cmd), waitUntilCompleted];

            self.epoch.end_dispatch();

            // `commandBuffer` returns an autoreleased object — this pool
            // owns it; no explicit release (the old release was the
            // over-release that SIGSEGV'd at drain).
            Ok(())
        })

        // The sync path waits for completion inside the pool, so nothing
        // escapes it.
    }

    fn dispatch_async(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<Pulse> {
        // Scoped pool: bounds the encoder and any intermediates. The
        // command buffer deliberately ESCAPES this pool — the explicit
        // retain below (+1) owns it past the drain, balanced by the
        // release in wait_metal_pulse / drop_metal_pulse.
        objc::rc::autoreleasepool(|| unsafe {
            let cmd: *mut c_void = msg_send![obj(self.z.command_queue()), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void = msg_send![obj(cmd), computeCommandEncoder];
            if encoder.is_null() {
                // cmd is autoreleased — do not release it here.
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            let _: () =
                msg_send![obj(encoder), setComputePipelineState: mtl_pipeline(pipeline.raw)];

            for (i, buf) in buffers.iter().enumerate() {
                let _: () = msg_send![
                    obj(encoder),
                    setBuffer: self.buffer_handle(buf)
                    offset: 0u64
                    atIndex: i as u64
                ];
            }

            // Bind the naga buffer-sizes constant (arrayLength support).
            // Ok(None) = kernel has no sizes constant; only allocation
            // failure is an error here.
            let sizes_buf = bind_buffer_sizes(self.z.raw_device(), encoder, pipeline.raw, buffers)?;

            let _: () = msg_send![
                obj(encoder),
                dispatchThreadgroups: (workgroups.0 as u64, workgroups.1 as u64, workgroups.2 as u64)
                threadsPerThreadgroup: (256u64, 1u64, 1u64)
            ];

            let _: () = msg_send![obj(encoder), endEncoding];
            // Encoded resources are retained by the (async) command buffer
            // until it completes — release our +1 now.
            release_sizes_buffer(sizes_buf);

            self.epoch.begin_dispatch();

            let _: () = msg_send![obj(cmd), commit];

            // +1 retain: the command buffer escapes the pool scope.
            // Balanced by the release in wait_metal_pulse /
            // drop_metal_pulse.
            let _: () = msg_send![obj(cmd), retain];

            // Store command buffer in Pulse; wait+release on demand.
            // The epoch tracker is shared (Arc) — the pulse owns a
            // reference and cannot dangle if the backend is dropped first.
            let inner = Box::new(MetalPulseInner {
                cmd,
                epoch: std::sync::Arc::clone(&self.epoch),
                epoch_completed: std::sync::atomic::AtomicBool::new(false),
            });

            Ok(Pulse {
                raw: Box::into_raw(inner) as *mut std::ffi::c_void,
                wait_fn: wait_metal_pulse,
                drop_fn: drop_metal_pulse,
            })
        })
    }

    /// Delegates to the substrate: `Private`-storage buffers (the
    /// `DeviceLocal` strategy and `create_device_buffer` products) are
    /// only readable through the substrate's staging-blit path — a raw
    /// `contents()` read would return stale staging bytes.
    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>> {
        // Safety: `raw` was produced by `Box::into_raw::<MetalZuneshaBufferInner>`
        // by this backend and remains valid while the buffer is alive.
        let inner = unsafe { &*(buffer.raw as *const MetalZuneshaBufferInner) };
        self.z
            .read_buffer::<T>(&inner.zbuf)
            .map_err(|e| GpuError::BufferReadFailed {
                message: format!("zunesha read_buffer: {e}"),
            })
    }

    fn timestamp(&self) -> Result<u64> {
        // CPU monotonic timestamp in nanoseconds.
        // On unified memory (Apple Silicon), GPU execution time closely
        // tracks CPU wall time. For Metal GPU-accurate timestamps,
        // use MTLCommandBuffer.gpuEndTime (requires command buffer liftetime).
        Ok(std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos() as u64)
    }

    fn compile_cached(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                std::path::PathBuf::from(home).join(".cache")
            })
            .join("borsalino");
        let _ = std::fs::create_dir_all(&cache_dir);

        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in wgsl_source.as_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let cache_path = cache_dir.join(format!("{entry_point}_{hash:016x}.msl"));

        if let Ok(cached_msl) = std::fs::read_to_string(&cache_path) {
            // The first line must carry the sizes header (see
            // `SIZES_HEADER_PREFIX`) — caches without it predate
            // arrayLength support and are treated as misses.
            if let Some((sizes_bindings, body)) = split_sizes_header(&cached_msl) {
                if !body.is_empty() {
                    return self.compile_msl(entry_point, body, sizes_bindings);
                }
            }
        }

        // Cache miss: compile normally
        let pipeline = self.compile(entry_point, wgsl_source)?;
        // MSL is generated internally by compile() — can't easily extract.
        // For now, cache miss compiles fresh each time.
        Ok(pipeline)
    }

    fn dispatch_many(&self, dispatches: &[crate::DispatchSpec<'_>]) -> Result<()> {
        if dispatches.is_empty() {
            return Ok(());
        }

        // Scoped autorelease pool — same ownership discipline as
        // dispatch_ex (P2 review finding).
        objc::rc::autoreleasepool(|| unsafe {
            let cmd: *mut c_void = msg_send![obj(self.z.command_queue()), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void = msg_send![obj(cmd), computeCommandEncoder];
            if encoder.is_null() {
                // cmd is autoreleased — do not release it here.
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            // Sizes buffers (one per spec) — released after endEncoding.
            let mut sizes_bufs: Vec<Option<*mut c_void>> = Vec::with_capacity(dispatches.len());

            for spec in dispatches {
                // Set pipeline
                let _: () = msg_send![
                    obj(encoder),
                    setComputePipelineState: mtl_pipeline(spec.pipeline.raw)
                ];

                // Bind buffers
                for (i, buf) in spec.buffers.iter().enumerate() {
                    let _: () = msg_send![
                        obj(encoder),
                        setBuffer: self.buffer_handle(buf)
                        offset: 0u64
                        atIndex: i as u64
                    ];
                }

                // Bind the naga buffer-sizes constant (arrayLength
                // support) — one per spec; slot 30 is overwritten per
                // dispatch, matching the encoder's sequential encoding.
                // Ok(None) = spec kernel has no sizes constant; release
                // handles absence as a no-op. On failure, release every
                // sizes buffer collected so far before propagating — the
                // final release loop is bypassed by `?` (review r3 P2).
                let sizes_buf = match bind_buffer_sizes(
                    self.z.raw_device(),
                    encoder,
                    spec.pipeline.raw,
                    spec.buffers,
                ) {
                    Ok(sb) => sb,
                    Err(e) => {
                        for sb in sizes_bufs.drain(..) {
                            release_sizes_buffer(sb);
                        }
                        return Err(e);
                    }
                };
                sizes_bufs.push(sizes_buf);

                // Dispatch
                let _: () = msg_send![
                    obj(encoder),
                    dispatchThreadgroups: (
                        spec.workgroups.0 as u64,
                        spec.workgroups.1 as u64,
                        spec.workgroups.2 as u64,
                    )
                    threadsPerThreadgroup: (
                        spec.threads_per_group.0 as u64,
                        spec.threads_per_group.1 as u64,
                        spec.threads_per_group.2 as u64,
                    )
                ];
            }

            let _: () = msg_send![obj(encoder), endEncoding];
            // Encoded resources are retained by the command buffer —
            // release our +1s now.
            for sb in sizes_bufs {
                release_sizes_buffer(sb);
            }

            self.epoch.begin_dispatch();

            let _: () = msg_send![obj(cmd), commit];
            let _: () = msg_send![obj(cmd), waitUntilCompleted];

            self.epoch.end_dispatch();

            // autoreleased command buffer — owned by this pool's drain.
            Ok(())
        })

        // Sync path: everything completed inside the pool.
    }

    fn in_flight(&self) -> u64 {
        self.epoch.in_flight()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

// ── Inherent methods (not part of the GpuBackend trait) ─────────────

impl MetalBackend {
    /// Compile pre-generated MSL directly (skips naga).
    fn compile_msl(
        &self,
        entry_point: &str,
        msl_source: &str,
        sizes_bindings: Vec<u32>,
    ) -> Result<ComputePipeline> {
        let dev = self.z.raw_device();

        unsafe {
            let ns_src = nsstring(msl_source);
            let mut err: *mut c_void = std::ptr::null_mut();
            let library: *mut c_void = msg_send![
                dev as *const Object,
                newLibraryWithSource: ns_src
                options: std::ptr::null_mut::<c_void>()
                error: &mut err
            ];

            if library.is_null() {
                let msg = if !err.is_null() {
                    let desc: *mut c_void = msg_send![err as *const Object, localizedDescription];
                    let s = nsstring_read(desc);
                    // (err/perr is an autoreleased out-param — the pool owns it)
                    s
                } else {
                    "unknown compilation error".into()
                };
                return Err(GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            let ns_entry = nsstring(entry_point);
            let func: *mut c_void =
                msg_send![library as *const Object, newFunctionWithName: ns_entry];

            if func.is_null() {
                let _: () = msg_send![library as *const Object, release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!("function '{entry_point}' not found in compiled library"),
                });
            }

            let desc: *mut c_void = msg_send![class!(MTLComputePipelineDescriptor), new];
            let _: () = msg_send![desc as *const Object, setComputeFunction: func];
            let mut perr: *mut c_void = std::ptr::null_mut();
            let pipeline: *mut c_void = msg_send![
                dev as *const Object,
                newComputePipelineStateWithDescriptor: desc
                options: 0u64
                reflection: std::ptr::null_mut::<c_void>()
                error: &mut perr
            ];

            if pipeline.is_null() {
                let msg = if !perr.is_null() {
                    let desc: *mut c_void = msg_send![perr as *const Object, localizedDescription];
                    let s = nsstring_read(desc);
                    // (err/perr is an autoreleased out-param — the pool owns it)
                    s
                } else {
                    "unknown pipeline error".into()
                };
                // desc is +1 from `new` — the failure path used to leak
                // it (review finding). func/library are also +1.
                let _: () = msg_send![obj(desc), release];
                let _: () = msg_send![obj(func), release];
                let _: () = msg_send![obj(library), release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // `new` returns a retained object — release our reference.
            let _: () = msg_send![obj(desc), release];
            let _: () = msg_send![obj(func), release];
            let _: () = msg_send![obj(library), release];

            Ok(ComputePipeline {
                raw: Box::into_raw(Box::new(MetalPipelineInner {
                    pipeline,
                    sizes_bindings,
                })) as *mut c_void,
                drop_fn: drop_pipeline,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CPU-only: naga buffer-sizes layout (review round 1, P1) ─────────

    /// The sizes constant's field order is MODULE DECLARATION ORDER, not
    /// binding order — a kernel declaring binding 1 before binding 0 must
    /// produce layout `[1, 0]`, or dispatch binds the wrong byte sizes and
    /// arrayLength guards mis-guard.
    #[test]
    fn sizes_bindings_follow_declaration_order_not_binding_order() {
        let wgsl = r#"
@group(0) @binding(1) var<storage, read> second: array<f32>;
@group(0) @binding(0) var<storage, read_write> first: array<f32>;

@compute @workgroup_size(4)
fn k(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&first)) { return; }
    first[gid.x] = second[gid.x];
}
"#;
        let module = naga::front::wgsl::parse_str(wgsl).unwrap();
        assert_eq!(sizes_buffer_bindings(&module), vec![1, 0]);
    }

    /// Struct-typed storage blocks (last member = dynamic array) qualify
    /// too, and fixed-size arrays do not.
    #[test]
    fn sizes_bindings_cover_struct_blocks_and_skip_fixed_arrays() {
        let wgsl = r#"
struct Block { prefix: u32, tail: array<f32>, };
struct Fixed { data: array<f32, 4>, };

@group(0) @binding(0) var<storage, read_write> block: Block;
@group(0) @binding(1) var<storage, read> fixed: Fixed;
@group(0) @binding(2) var<storage, read> plain: array<f32>;

@compute @workgroup_size(4)
fn k(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&plain)) { return; }
    block.tail[gid.x] = plain[gid.x] + fixed.data[gid.x % 4u];
}
"#;
        let module = naga::front::wgsl::parse_str(wgsl).unwrap();
        // block (struct with dynamic tail) and plain qualify; fixed does not.
        assert_eq!(sizes_buffer_bindings(&module), vec![0, 2]);
    }

    /// The cached-MSL sizes header round-trips exactly, and caches without
    /// it are treated as misses (pre-header caches predate the layout).
    #[test]
    fn sizes_header_round_trips() {
        for layout in [Vec::new(), vec![0u32], vec![1, 0], vec![3, 1, 0, 2]] {
            let body = "// language: metal1.0\nkernel void k() {}\n";
            let cached = format!("{}\n{}", make_sizes_header(&layout), body);
            let (parsed, rest) = split_sizes_header(&cached).unwrap();
            assert_eq!(parsed, layout);
            assert_eq!(rest, body);
        }
        assert!(split_sizes_header("// language: metal1.0\nno header").is_none());
    }

    /// Acquire a device for tests: skip politely on machines without one,
    /// but FAIL hard when `BORSALINO_REQUIRE_METAL` is set — the dedicated
    /// Apple Silicon CI job sets it, so that job can never pass without
    /// executing real kernels (P2 review finding).
    fn test_device() -> Option<MetalBackend> {
        match MetalBackend::init() {
            Ok(b) => Some(b),
            Err(e) => {
                assert!(
                    std::env::var("BORSALINO_REQUIRE_METAL").is_err(),
                    "BORSALINO_REQUIRE_METAL is set but Metal init failed: {e}"
                );
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn device_init() {
        let _backend = test_device();
    }

    #[test]
    #[serial_test::serial]
    fn add_one_kernel() {
        let Some(backend) = test_device() else { return };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                // The pipeline compiles with Unchecked buffer bounds, so
                // the shader itself must bound its accesses (review
                // finding: 256 threads over 4 elements read/wrote OOB).
                if (i >= 4u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;

        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

        // Normal destruction: with the ownership fixes (scoped pools, no
        // over-releases) the drop path is part of what this test verifies
        // (P2 review finding — the mem::forgets were the old workaround).
    }

    #[test]
    #[serial_test::serial]
    fn vector_scale_1024() {
        let Some(backend) = test_device() else { return };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                output[i] = input[i] * 2.5;
            }
        "#;

        let n: usize = 1024;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected: Vec<f32> = input_data.iter().map(|x| x * 2.5).collect();

        let pipeline = backend.compile("scale", wgsl).unwrap();
        let input = backend.create_buffer(&input_data).unwrap();
        let output = backend.create_buffer_uninit::<f32>(n).unwrap();

        backend
            .dispatch(&pipeline, &[&input, &output], (4, 1, 1))
            .unwrap();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-6,
                "mismatch at index {i}: got {r}, expected {e}"
            );
        }

        // Normal destruction — see add_one_kernel.
    }

    /// Regression (P2 review finding): an async `Pulse` must survive an
    /// autorelease-pool drain after `dispatch_async`. The command buffer
    /// escapes the pool via its explicit retain; the epoch tracker is a
    /// shared `Arc`, not a borrow of the backend.
    #[test]
    #[serial_test::serial]
    fn async_pulse_survives_pool_drain() {
        let Some(backend) = test_device() else { return };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                // The pipeline compiles with Unchecked buffer bounds, so
                // the shader itself must bound its accesses (review
                // finding: 256 threads over 4 elements read/wrote OOB).
                if (i >= 4u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();

        // dispatch_async INSIDE a pool scope; the Pulse (retained cmd)
        // crosses the drain by design.
        let pulse = objc::rc::autoreleasepool(|| {
            backend
                .dispatch_async(&pipeline, &[&input, &output], (1, 1, 1))
                .unwrap()
        });

        // Pool has drained. Wait, read back, verify — then normal drops.
        pulse.wait();
        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
        drop(pulse);
    }

    /// Phase 3 anchor (mirrors the Vulkan §5.1 contract):
    /// `create_device_buffer` delegates to the substrate — Private
    /// storage + persistent staging per Zunesha's documented contract,
    /// observable as the substrate tag from `contents()` and the
    /// substrate drop function. Round-trips through the substrate's
    /// staging-blit read (a raw `contents()` read would be stale).
    #[test]
    #[serial_test::serial]
    fn create_device_buffer_delegates_to_substrate() {
        let Some(backend) = test_device() else { return };
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buf = backend.create_device_buffer(&data).unwrap();

        // Substrate tag, not a Borsalino MTLBuffer and not null.
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "device buffer must be substrate-backed (sentinel tag)"
        );
        // Substrate drop path — the deleted Borsalino-allocated kind's
        // drop function must NOT be in use anymore.
        assert!(
            std::ptr::eq(
                buf.drop_fn as *const (),
                drop_metal_zunesha_buffer as *const ()
            ),
            "device buffer must use the substrate drop function"
        );
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data.to_vec());
    }

    /// Same delegation for the uninit variant: substrate-backed, correct
    /// length/element_size, readable (contents unspecified until written).
    #[test]
    #[serial_test::serial]
    fn create_device_buffer_uninit_delegates_to_substrate() {
        let Some(backend) = test_device() else { return };
        let buf = backend.create_device_buffer_uninit::<f32>(16).unwrap();
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "uninit device buffer must be substrate-backed"
        );
        assert_eq!(buf.len, 16);
        assert_eq!(buf.element_size, std::mem::size_of::<f32>());
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back.len(), 16);
    }

    /// The delegated contract holds under forced `Unified` (on Metal every
    /// strategy product must still round-trip through the substrate —
    /// `create_device_buffer` forces Private+staging regardless).
    #[test]
    #[serial_test::serial]
    fn device_buffer_roundtrips_under_forced_unified() {
        let backend = match MetalBackend::init_with_strategy(MemoryStrategy::Unified) {
            Ok(b) => Some(b),
            Err(e) => {
                assert!(
                    std::env::var("BORSALINO_REQUIRE_METAL").is_err(),
                    "BORSALINO_REQUIRE_METAL is set but Metal init failed: {e}"
                );
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        };
        let Some(backend) = backend else { return };
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25).collect();
        let buf = backend.create_device_buffer(&data).unwrap();
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data);
    }

    /// Zunesha R1 improvement, pinned: `MemoryStrategy::DeviceLocal` maps
    /// to Private storage + staging on Metal — Borsalino's pre-Phase-3
    /// Metal silently ignored the strategy (Shared always). The strategy
    /// path must round-trip through the substrate's blit read.
    #[test]
    #[serial_test::serial]
    fn device_local_strategy_buffers_roundtrip() {
        let backend = match MetalBackend::init_with_strategy(MemoryStrategy::DeviceLocal) {
            Ok(b) => Some(b),
            Err(e) => {
                assert!(
                    std::env::var("BORSALINO_REQUIRE_METAL").is_err(),
                    "BORSALINO_REQUIRE_METAL is set but Metal init failed: {e}"
                );
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        };
        let Some(backend) = backend else { return };
        let data: Vec<f32> = (0..256).map(|i| i as f32 - 128.0).collect();
        let buf = backend.create_buffer(&data).unwrap();
        assert!(
            std::ptr::eq((buf.contents_fn)(buf.raw), SUBSTRATE_BUFFER_TAG),
            "strategy-path buffer must be substrate-backed"
        );
        let back: Vec<f32> = backend.read_buffer(&buf).unwrap();
        assert_eq!(back, data);
    }

    /// Review r1 P2: Private buffers (create_device_buffer products)
    /// must read back through a FRESH device→staging blit — a stale-
    /// staging read would return the pre-kernel bytes. Writes via a real
    /// kernel dispatch, so the read path is exercised end-to-end.
    #[test]
    #[serial_test::serial]
    fn private_buffer_kernel_write_reads_back_fresh() {
        let Some(backend) = test_device() else { return };
        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= 8u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = backend.create_device_buffer(&data).unwrap();
        let output = backend.create_device_buffer_uninit::<f32>(8).unwrap();

        // Negotiated placement is observable at the substrate level only
        // (device-level `buffer_placement`); per-buffer storage mode is
        // the substrate's contract. On unified Apple Silicon the strategy
        // placement is HostVisible — but `create_device_buffer` forces
        // Private regardless (Zunesha R3).
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        // The fresh-blit pin: these must be the KERNEL's outputs, not the
        // pre-dispatch staging contents (the original `data`/zeros).
        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // And a second dispatch must read the SECOND blit (not a cached
        // first): feed output back through the kernel.
        backend
            .dispatch(&pipeline, &[&output, &output], (1, 1, 1))
            .unwrap();
        let result2: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result2, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    /// Review r1 P2: the `DeviceLocal` strategy path (Private +
    /// staging) must also round-trip kernel outputs — distinct from the
    /// forced-Private `create_device_buffer` path above, this exercises
    /// the strategy-negotiated allocation feeding dispatch + blit read.
    #[test]
    #[serial_test::serial]
    fn device_local_strategy_kernel_write_reads_back() {
        let backend = match MetalBackend::init_with_strategy(MemoryStrategy::DeviceLocal) {
            Ok(b) => Some(b),
            Err(e) => {
                assert!(
                    std::env::var("BORSALINO_REQUIRE_METAL").is_err(),
                    "BORSALINO_REQUIRE_METAL is set but Metal init failed: {e}"
                );
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        };
        let Some(backend) = backend else { return };
        // Placement assertion at the observable (substrate) level.
        assert!(matches!(
            backend.z.buffer_placement(),
            zunesha::MemoryPlacement::DeviceLocal
        ));

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= 4u) { return; }
                output[i] = input[i] * 2.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();
        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![20.0, 40.0, 60.0, 80.0]);
    }

    /// Review r1 P2, Shared side: the unified strategy's buffers are
    /// Shared storage — placement observable as HostVisible, and kernel
    /// outputs read back directly.
    #[test]
    #[serial_test::serial]
    fn unified_strategy_placement_is_host_visible() {
        let backend = match MetalBackend::init_with_strategy(MemoryStrategy::Unified) {
            Ok(b) => Some(b),
            Err(e) => {
                assert!(
                    std::env::var("BORSALINO_REQUIRE_METAL").is_err(),
                    "BORSALINO_REQUIRE_METAL is set but Metal init failed: {e}"
                );
                eprintln!("skipping: no Metal device ({e})");
                None
            }
        };
        let Some(backend) = backend else { return };
        assert!(matches!(
            backend.z.buffer_placement(),
            zunesha::MemoryPlacement::HostVisible
        ));
    }

    /// Regression (P2 review finding): batch dispatch through
    /// `dispatch_many` — a single command buffer carrying two encodes,
    /// verifying the last kernel's output plus normal destruction.
    #[test]
    #[serial_test::serial]
    fn dispatch_many_executes_batch() {
        use crate::DispatchSpec;

        let Some(backend) = test_device() else { return };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                // The pipeline compiles with Unchecked buffer bounds, so
                // the shader itself must bound its accesses (review
                // finding: 256 threads over 4 elements read/wrote OOB).
                if (i >= 4u) { return; }
                output[i] = input[i] + 1.0;
            }
        "#;
        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();

        let specs = [
            DispatchSpec {
                pipeline: &pipeline,
                buffers: &[&input, &output],
                workgroups: (1, 1, 1),
                threads_per_group: (256, 1, 1),
            },
            DispatchSpec {
                pipeline: &pipeline,
                buffers: &[&output, &output],
                workgroups: (1, 1, 1),
                threads_per_group: (256, 1, 1),
            },
        ];
        backend.dispatch_many(&specs).unwrap();

        // Second spec feeds the first spec's output back through add_one.
        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }
}
