// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Metal GPU backend for Apple Silicon.
//!
//! Uses the `objc` crate for safe `msg_send!` dispatch on ARM64.
//! All other FFI (MTLCreateSystemDefaultDevice, naga WGSL→MSL) is raw.
//!
//! ## Dependencies
//!
//! - `objc` 0.2 — `msg_send!` macro, correctly handles ARM64 calling convention
//! - `naga` 27 — WGSL → MSL translation

use std::ffi::c_void;
use std::ptr::NonNull;

use naga::back::msl;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use crate::{ComputePipeline, GpuBuffer, GpuBackend, GpuError, Result, DispatchSpec};

// ═══════════════════════════════════════════════════════════════════
// Metal C symbol
// ═══════════════════════════════════════════════════════════════════

#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
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
    msg_send![class!(NSString), stringWithUTF8String: c_str.as_ptr() as *const i8]
}

/// Read an NSString into a Rust String.
unsafe fn nsstring_read(ns: *mut c_void) -> String {
    let utf8: *const std::ffi::c_char = msg_send![obj(ns), UTF8String];
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

struct MetalDevice {
    ptr: NonNull<c_void>,
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![obj(self.ptr.as_ptr()), release];
        }
    }
}

struct MetalQueue {
    ptr: NonNull<c_void>,
}

impl Drop for MetalQueue {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![obj(self.ptr.as_ptr()), release];
        }
    }
}

fn drop_pipeline(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            let _: () = msg_send![obj(raw), release];
        }
    }
}

fn drop_buffer(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            let _: () = msg_send![obj(raw), release];
        }
    }
}

fn contents_of(raw: *mut c_void) -> *const c_void {
    if raw.is_null() {
        return std::ptr::null();
    }
    unsafe { msg_send![obj(raw), contents] }
}

// ── MSL post-processing ──────────────────────────────────────────

/// Post-process naga-generated MSL to fix Metal 3 compatibility.
/// Naga emits `device type_N const&` / `device type_N&` (references to
/// fixed-size arrays), but Metal 3's pipeline creation crashes with this
/// syntax. Converts to pointer syntax and strips unused structs.
fn naga_msl_fixup(msl: &str) -> String {
    let mut out = String::with_capacity(msl.len());
    let mut in_buffer_sizes = false;

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

        // Skip `_mslBufferSizes` struct (stateful: skip until closing };)
        if trimmed == "struct _mslBufferSizes {" {
            in_buffer_sizes = true;
            continue;
        }
        if in_buffer_sizes {
            if trimmed == "};" {
                in_buffer_sizes = false;
            }
            continue;
        }

        // Skip lines containing `_buffer_sizes` (the parameter)
        if trimmed.contains("_buffer_sizes") {
            continue;
        }

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
    let name_end = rest.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(rest.len());
    let name = &rest[..name_end];
    let suffix = &rest[name_end..];
    let prefix = if mutable { "device float* " } else { "device const float* " };
    let before = &line[..type_start];
    Some(format!("{before}{prefix}{name}{suffix}"))
}

// ═══════════════════════════════════════════════════════════════════
// MetalBackend
// ═══════════════════════════════════════════════════════════════════

/// Metal GPU backend for Apple Silicon.
pub struct MetalBackend {
    device: MetalDevice,
    queue: MetalQueue,
}

impl MetalBackend {
    const STORAGE_MODE_SHARED: u64 = 0;
}

impl GpuBackend for MetalBackend {
    fn init() -> Result<Self> {
        let device_ptr = unsafe { MTLCreateSystemDefaultDevice() };
        if device_ptr.is_null() {
            return Err(GpuError::InitFailed(
                "MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU"
                    .into(),
            ));
        }

        let queue_ptr: *mut c_void =
            unsafe { msg_send![obj(device_ptr), newCommandQueue] };
        if queue_ptr.is_null() {
            unsafe {
                let _: () = msg_send![obj(device_ptr), release];
            }
            return Err(GpuError::InitFailed(
                "failed to create MTLCommandQueue".into(),
            ));
        }

        Ok(Self {
            device: MetalDevice {
                ptr: NonNull::new(device_ptr).unwrap(),
            },
            queue: MetalQueue {
                ptr: NonNull::new(queue_ptr).unwrap(),
            },
        })
    }

    fn compile(
        &self,
        entry_point: &str,
        wgsl_source: &str,
    ) -> Result<ComputePipeline> {
        // Step 0: Translate WGSL → MSL via naga
        let module =
            wgsl::parse_str(wgsl_source).map_err(|e| GpuError::CompileFailed {
                entry: entry_point.into(),
                message: e.emit_to_string(wgsl_source),
            })?;

        let mut validator =
            Validator::new(ValidationFlags::all(), Capabilities::all());
        let info =
            validator.validate(&module).map_err(|e| GpuError::CompileFailed {
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

        let mut msl_opts = msl::Options::default();
        msl_opts.fake_missing_bindings = false;
        msl_opts.bounds_check_policies = naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
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

        // Fix naga MSL for Metal 3 compatibility
        msl_source = naga_msl_fixup(&msl_source);

        let dev = self.device.ptr.as_ptr();

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
                    let _: () = msg_send![err as *const objc::runtime::Object, release];
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
                    message: format!(
                        "function '{entry_point}' not found in compiled library"
                    ),
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
                    let desc: *mut c_void =
                        msg_send![obj(perr), localizedDescription];
                    let s = nsstring_read(desc);
                    let _: () = msg_send![obj(perr), release];
                    s
                } else {
                    "unknown pipeline error".into()
                };
                let _: () = msg_send![obj(func), release];
                let _: () = msg_send![obj(library), release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // Release intermediates (desc may be retained by the pipeline)
            // let _: () = msg_send![obj(desc), release];
            let _: () = msg_send![obj(func), release];
            let _: () = msg_send![obj(library), release];

            Ok(ComputePipeline {
                raw: pipeline,
                drop_fn: drop_pipeline,
            })
        }
    }

    fn create_buffer<T: bytemuck::Pod>(
        &self,
        data: &[T],
    ) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = data.len() * element_size;
        let dev = self.device.ptr.as_ptr();

        let buf: *mut c_void = unsafe {
            msg_send![
                obj(dev),
                newBufferWithBytes: data.as_ptr() as *const c_void
                length: byte_len as u64
                options: Self::STORAGE_MODE_SHARED
            ]
        };

        if buf.is_null() {
            return Err(GpuError::BufferCreationFailed {
                message: format!(
                    "failed to allocate {byte_len} bytes ({len} × {element_size}B)",
                    len = data.len()
                ),
            });
        }

        Ok(GpuBuffer {
            raw: buf,
            len: data.len(),
            element_size,
            drop_fn: drop_buffer,
            contents_fn: contents_of,
        })
    }

    fn create_buffer_uninit<T: bytemuck::Pod>(
        &self,
        len: usize,
    ) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = len * element_size;
        let dev = self.device.ptr.as_ptr();

        let buf: *mut c_void = unsafe {
            msg_send![
                obj(dev),
                newBufferWithLength: byte_len as u64
                options: Self::STORAGE_MODE_SHARED
            ]
        };

        if buf.is_null() {
            return Err(GpuError::BufferCreationFailed {
                message: format!("failed to allocate {byte_len} bytes (uninit)"),
            });
        }

        Ok(GpuBuffer {
            raw: buf,
            len,
            element_size,
            drop_fn: drop_buffer,
            contents_fn: contents_of,
        })
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
        unsafe {
            let cmd: *mut c_void =
                msg_send![obj(self.queue.ptr.as_ptr()), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void =
                msg_send![obj(cmd), computeCommandEncoder];
            if encoder.is_null() {
                let _: () = msg_send![obj(cmd), release];
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            // Set pipeline
            let _: () =
                msg_send![obj(encoder), setComputePipelineState: pipeline.raw];

            // Bind user buffers
            for (i, buf) in buffers.iter().enumerate() {
                let _: () = msg_send![
                    obj(encoder),
                    setBuffer: buf.raw
                    offset: 0u64
                    atIndex: i as u64
                ];
            }

            // Dispatch
            let _: () = msg_send![
                obj(encoder),
                dispatchThreadgroups: (workgroups.0 as u64, workgroups.1 as u64, workgroups.2 as u64)
                threadsPerThreadgroup: (_threads_per_group.0 as u64, _threads_per_group.1 as u64, _threads_per_group.2 as u64)
            ];

            // Finish
            let _: () = msg_send![obj(encoder), endEncoding];
            let _: () = msg_send![obj(cmd), commit];
            let _: () = msg_send![obj(cmd), waitUntilCompleted];
            let _: () = msg_send![obj(cmd), release];
        }

        Ok(())
    }

    fn read_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &GpuBuffer,
    ) -> Result<Vec<T>> {
        let contents = (buffer.contents_fn)(buffer.raw) as *const T;
        if contents.is_null() {
            return Err(GpuError::BufferReadFailed {
                message: "buffer contents pointer is null".into(),
            });
        }
        let slice =
            unsafe { std::slice::from_raw_parts(contents, buffer.len) };
        Ok(slice.to_vec())
    }

    fn timestamp(&self) -> Result<u64> {
        // CPU monotonic timestamp in nanoseconds.
        // On unified memory (Apple Silicon), GPU execution time closely
        // tracks CPU wall time. For Metal GPU-accurate timestamps,
        // use MTLCommandBuffer.gpuEndTime (requires command buffer liftetime).
        Ok(std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos() as u64)
    }

    fn dispatch_many(&self, dispatches: &[crate::DispatchSpec<'_>]) -> Result<()> {
        if dispatches.is_empty() {
            return Ok(());
        }

        unsafe {
            let cmd: *mut c_void =
                msg_send![obj(self.queue.ptr.as_ptr()), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void =
                msg_send![obj(cmd), computeCommandEncoder];
            if encoder.is_null() {
                let _: () = msg_send![obj(cmd), release];
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            for spec in dispatches {
                // Set pipeline
                let _: () = msg_send![
                    obj(encoder),
                    setComputePipelineState: spec.pipeline.raw
                ];

                // Bind buffers
                for (i, buf) in spec.buffers.iter().enumerate() {
                    let _: () = msg_send![
                        obj(encoder),
                        setBuffer: buf.raw
                        offset: 0u64
                        atIndex: i as u64
                    ];
                }

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
            let _: () = msg_send![obj(cmd), commit];
            let _: () = msg_send![obj(cmd), waitUntilCompleted];
            let _: () = msg_send![obj(cmd), release];
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_init() {
        match MetalBackend::init() {
            Ok(_) => {}
            Err(GpuError::InitFailed(msg)) => {
                eprintln!("Metal init failed (expected in CI/headless): {msg}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn add_one_kernel() {
        let backend = match MetalBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Metal device");
                return;
            }
        };

        let wgsl = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                output[i] = input[i] + 1.0;
            }
        "#;

        let pipeline = backend.compile("add_one", wgsl).unwrap();
        let input =
            backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output = backend.create_buffer_uninit::<f32>(4).unwrap();
        backend
            .dispatch(&pipeline, &[&input, &output], (1, 1, 1))
            .unwrap();

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

        let result: Vec<f32> = backend.read_buffer(&output).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

        // Prevent Drop: known Metal thread-cleanup SIGSEGV in test harness.
        // Examples (main thread) work correctly without this workaround.
        std::mem::forget(output);
        std::mem::forget(input);
        std::mem::forget(pipeline);
        std::mem::forget(backend);
    }

    #[test]
    fn vector_scale_1024() {
        let backend = match MetalBackend::init() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skipping: no Metal device");
                return;
            }
        };

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

        std::mem::forget(output);
        std::mem::forget(input);
        std::mem::forget(pipeline);
        std::mem::forget(backend);
    }
}
