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
use objc::runtime::Sel;
use objc::{class, msg_send, sel};

use crate::{ComputePipeline, GpuBuffer, GpuBackend, GpuError, Result};

// ═══════════════════════════════════════════════════════════════════
// Metal C symbol
// ═══════════════════════════════════════════════════════════════════

#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

// ═══════════════════════════════════════════════════════════════════
// Selector cache
// ═══════════════════════════════════════════════════════════════════

struct Selectors {
    new_buffer_with_bytes: Sel,
    new_library_with_source: Sel,
    new_function_with_name: Sel,
    new_compute_pipeline_state: Sel,
    new_command_queue: Sel,
    command_buffer: Sel,
    compute_command_encoder: Sel,
    set_compute_pipeline_state: Sel,
    set_buffer_offset_at_index: Sel,
    dispatch_threadgroups: Sel,
    end_encoding: Sel,
    commit: Sel,
    wait_until_completed: Sel,
    contents: Sel,
    retain: Sel,
    release: Sel,
    localized_description: Sel,
    utf8_string: Sel,
}

// SAFETY: Sel is a zero-sized wrapper around a registered selector pointer.
unsafe impl Send for Selectors {}
unsafe impl Sync for Selectors {}

fn selectors() -> &'static Selectors {
    use std::sync::OnceLock;
    static SEL: OnceLock<Selectors> = OnceLock::new();
    SEL.get_or_init(|| Selectors {
        new_buffer_with_bytes: sel!(newBufferWithBytes:length:options:),
        new_library_with_source: sel!(newLibraryWithSource:options:error:),
        new_function_with_name: sel!(newFunctionWithName:),
        new_compute_pipeline_state: sel!(newComputePipelineStateWithFunction:error:),
        new_command_queue: sel!(newCommandQueue),
        command_buffer: sel!(commandBuffer),
        compute_command_encoder: sel!(computeCommandEncoder),
        set_compute_pipeline_state: sel!(setComputePipelineState:),
        set_buffer_offset_at_index: sel!(setBuffer:offset:atIndex:),
        dispatch_threadgroups: sel!(dispatchThreadgroups:threadsPerThreadgroup:),
        end_encoding: sel!(endEncoding),
        commit: sel!(commit),
        wait_until_completed: sel!(waitUntilCompleted),
        contents: sel!(contents),
        retain: sel!(retain),
        release: sel!(release),
        localized_description: sel!(localizedDescription),
        utf8_string: sel!(UTF8String),
    })
}

// ═══════════════════════════════════════════════════════════════════
// NSString helpers
// ═══════════════════════════════════════════════════════════════════

unsafe fn nsstring(s: &str) -> *mut c_void {
    let ns: *mut c_void = msg_send![class!(NSString), stringWithUTF8String:s.as_ptr()];
    let _: *mut c_void = msg_send![ns, retain];
    ns
}

unsafe fn nsstring_read(ns: *mut c_void) -> String {
    let utf8: *const std::ffi::c_char = msg_send![ns, UTF8String];
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
            let _: () = msg_send![self.ptr.as_ptr(), release];
        }
    }
}

struct MetalQueue {
    ptr: NonNull<c_void>,
}

impl Drop for MetalQueue {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.ptr.as_ptr(), release];
        }
    }
}

fn drop_pipeline(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            let _: () = msg_send![raw, release];
        }
    }
}

fn drop_buffer(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            let _: () = msg_send![raw, release];
        }
    }
}

fn contents_of(raw: *mut c_void) -> *const c_void {
    if raw.is_null() {
        return std::ptr::null();
    }
    unsafe { msg_send![raw, contents] }
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
    const SIZES_BUFFER_SLOT: msl::Slot = 30;
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
            unsafe { msg_send![device_ptr, newCommandQueue] };
        if queue_ptr.is_null() {
            unsafe {
                let _: () = msg_send![device_ptr, release];
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
            sizes_buffer: Some(Self::SIZES_BUFFER_SLOT),
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

        let (msl_source, _) =
            msl::write_string(&module, &info, &msl_opts, &msl::PipelineOptions::default())
                .map_err(|e| GpuError::CompileFailed {
                    entry: entry_point.into(),
                    message: format!("MSL emission failed: {e}"),
                })?;

        let dev = self.device.ptr.as_ptr();

        unsafe {
            // Step 1: MTLLibrary from source
            let ns_src = nsstring(&msl_source);
            let mut err: *mut c_void = std::ptr::null_mut();
            let library: *mut c_void = msg_send![
                dev,
                newLibraryWithSource:ns_src
                options:std::ptr::null_mut::<c_void>()
                error:&mut err
            ];
            let _: () = msg_send![ns_src, release];

            if library.is_null() {
                let msg = if !err.is_null() {
                    let desc: *mut c_void = msg_send![err, localizedDescription];
                    let s = nsstring_read(desc);
                    let _: () = msg_send![err, release];
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
            let func: *mut c_void = msg_send![library, newFunctionWithName:ns_entry];
            let _: () = msg_send![ns_entry, release];

            if func.is_null() {
                let _: () = msg_send![library, release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!(
                        "function '{entry_point}' not found in compiled library"
                    ),
                });
            }

            // Step 3: MTLComputePipelineState
            let mut perr: *mut c_void = std::ptr::null_mut();
            let pipeline: *mut c_void = msg_send![
                dev,
                newComputePipelineStateWithFunction:func
                error:&mut perr
            ];

            if pipeline.is_null() {
                let msg = if !perr.is_null() {
                    let desc: *mut c_void = msg_send![perr, localizedDescription];
                    let s = nsstring_read(desc);
                    let _: () = msg_send![perr, release];
                    s
                } else {
                    "unknown pipeline error".into()
                };
                let _: () = msg_send![func, release];
                let _: () = msg_send![library, release];
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // Release intermediates
            let _: () = msg_send![func, release];
            let _: () = msg_send![library, release];

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
                dev,
                newBufferWithBytes:data.as_ptr() as *const c_void
                length:byte_len as u64
                options:Self::STORAGE_MODE_SHARED
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
                dev,
                newBufferWithBytes:std::ptr::null::<c_void>()
                length:byte_len as u64
                options:Self::STORAGE_MODE_SHARED
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
        threads_per_group: (u32, u32, u32),
    ) -> Result<()> {
        let dev = self.device.ptr.as_ptr();
        unsafe {
            let cmd: *mut c_void = msg_send![self.queue.ptr.as_ptr(), commandBuffer];
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder: *mut c_void = msg_send![cmd, computeCommandEncoder];
            if encoder.is_null() {
                let _: () = msg_send![cmd, release];
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            // Set pipeline
            let _: () =
                msg_send![encoder, setComputePipelineState:pipeline.raw];

            // Bind sizes buffer (naga runtime array element counts)
            let sizes: Vec<u32> = buffers.iter().map(|b| b.len as u32).collect();
            let sizes_buf: *mut c_void = msg_send![
                dev,
                newBufferWithBytes:sizes.as_ptr() as *const c_void
                length:(sizes.len() * 4) as u64
                options:Self::STORAGE_MODE_SHARED
            ];
            let _: () = msg_send![
                encoder,
                setBuffer:sizes_buf
                offset:0u64
                atIndex:Self::SIZES_BUFFER_SLOT as u64
            ];

            // Bind user buffers
            for (i, buf) in buffers.iter().enumerate() {
                let _: () = msg_send![
                    encoder,
                    setBuffer:buf.raw
                    offset:0u64
                    atIndex:i as u64
                ];
            }

            // Dispatch
            let _: () = msg_send![
                encoder,
                dispatchThreadgroups:(workgroups.0 as u64, workgroups.1 as u64, workgroups.2 as u64)
                threadsPerThreadgroup:(threads_per_group.0 as u64, threads_per_group.1 as u64, threads_per_group.2 as u64)
            ];

            // Finish
            let _: () = msg_send![encoder, endEncoding];
            let _: () = msg_send![cmd, commit];
            let _: () = msg_send![cmd, waitUntilCompleted];
            let _: () = msg_send![cmd, release];
            let _: () = msg_send![sizes_buf, release];
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
    }
}
