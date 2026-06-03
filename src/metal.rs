// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Metal GPU backend for Apple Silicon.
//!
//! Raw Objective-C FFI — no `metal-rs`, no `objc` crate, no `core-graphics`.
//! Just `extern "C"` calls to the Metal framework and Objective-C runtime.
//!
//! ## FFI surface
//!
//! 3 extern functions, 19 selectors, 1 crate dependency (naga for WGSL→MSL translation).
//!
//! | Function | Role |
//! |----------|------|
//! | `MTLCreateSystemDefaultDevice` | Get the default Metal GPU |
//! | `objc_getClass` | Look up an ObjC class |
//! | `sel_registerName` | Register a selector by name |
//! | `objc_msgSend` | Universal message dispatch |

use std::ffi::{c_void, CString};
use std::ptr::NonNull;

use naga::back::msl;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::{ComputePipeline, GpuBuffer, GpuBackend, GpuError, Result};

// ═══════════════════════════════════════════════════════════════════
// Objective-C / Metal extern declarations
// ═══════════════════════════════════════════════════════════════════

#[link(name = "Metal", kind = "framework")]
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    fn objc_getClass(name: *const std::ffi::c_char) -> *mut c_void;
    fn sel_registerName(name: *const std::ffi::c_char) -> *mut c_void;

    // Typed objc_msgSend variants — each includes self + _cmd + method args.
    // Properly declared for ARM64 ABI with explicit argument counts.
    #[link_name = "objc_msgSend"]
    fn msg_send_id(self_: *mut c_void, sel: *mut c_void) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn msg_send_id_id(self_: *mut c_void, sel: *mut c_void, a1: *mut c_void) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn msg_send_id_id_id(
        self_: *mut c_void,
        sel: *mut c_void,
        a1: *mut c_void,
        a2: *mut c_void,
    ) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn msg_send_id_buf(
        self_: *mut c_void,
        sel: *mut c_void,
        bytes: *const c_void,
        length: u64,
        options: u64,
    ) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn msg_send_id_lib(
        self_: *mut c_void,
        sel: *mut c_void,
        source: *mut c_void,
        opts: *mut c_void,
        error_out: *mut *mut c_void,
    ) -> *mut c_void;
    #[link_name = "objc_msgSend"]
    fn msg_send_void_id(self_: *mut c_void, sel: *mut c_void, a1: *mut c_void);
    #[link_name = "objc_msgSend"]
    fn msg_send_void(self_: *mut c_void, sel: *mut c_void);
    #[link_name = "objc_msgSend"]
    fn msg_send_set_buf(
        self_: *mut c_void,
        sel: *mut c_void,
        buffer: *mut c_void,
        offset: u64,
        index: u64,
    );
    #[link_name = "objc_msgSend"]
    fn msg_send_dispatch(
        self_: *mut c_void,
        sel: *mut c_void,
        gx: u64,
        gy: u64,
        gz: u64,
        tx: u64,
        ty: u64,
        tz: u64,
    );
    #[link_name = "objc_msgSend"]
    fn msg_send_string(
        self_: *mut c_void,
        sel: *mut c_void,
        s: *const u8,
    ) -> *mut c_void;
}

// ═══════════════════════════════════════════════════════════════════
// Selector cache
// ═══════════════════════════════════════════════════════════════════

struct Selectors {
    new_buffer_with_bytes: *mut c_void,
    new_library_with_source: *mut c_void,
    new_function_with_name: *mut c_void,
    new_compute_pipeline_state: *mut c_void,
    new_command_queue: *mut c_void,
    command_buffer: *mut c_void,
    compute_command_encoder: *mut c_void,
    set_compute_pipeline_state: *mut c_void,
    set_buffer_offset_at_index: *mut c_void,
    dispatch_threadgroups: *mut c_void,
    end_encoding: *mut c_void,
    commit: *mut c_void,
    wait_until_completed: *mut c_void,
    contents: *mut c_void,
    retain: *mut c_void,
    release: *mut c_void,
    localized_description: *mut c_void,
    utf8_string: *mut c_void,
}

// SAFETY: Selectors contains only opaque selector pointers obtained from
// sel_registerName, which are constant after registration and trivially Send+Sync.
unsafe impl Send for Selectors {}
unsafe impl Sync for Selectors {}

fn selectors() -> &'static Selectors {
    use std::sync::OnceLock;
    static SEL: OnceLock<Selectors> = OnceLock::new();
    SEL.get_or_init(|| unsafe {
        Selectors {
            new_buffer_with_bytes: sel("newBufferWithBytes:length:options:"),
            new_library_with_source: sel("newLibraryWithSource:options:error:"),
            new_function_with_name: sel("newFunctionWithName:"),
            new_compute_pipeline_state: sel("newComputePipelineStateWithFunction:error:"),
            new_command_queue: sel("newCommandQueue"),
            command_buffer: sel("commandBuffer"),
            compute_command_encoder: sel("computeCommandEncoder"),
            set_compute_pipeline_state: sel("setComputePipelineState:"),
            set_buffer_offset_at_index: sel("setBuffer:offset:atIndex:"),
            dispatch_threadgroups: sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            end_encoding: sel("endEncoding"),
            commit: sel("commit"),
            wait_until_completed: sel("waitUntilCompleted"),
            contents: sel("contents"),
            retain: sel("retain"),
            release: sel("release"),
            localized_description: sel("localizedDescription"),
            utf8_string: sel("UTF8String"),
        }
    })
}

unsafe fn sel(name: &str) -> *mut c_void {
    unsafe { sel_registerName(CString::new(name).unwrap().as_ptr()) }
}

// ═══════════════════════════════════════════════════════════════════
// objc_msgSend typed wrappers
// ═══════════════════════════════════════════════════════════════════

unsafe fn msg_id(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void {
    unsafe { msg_send_id_id(receiver, sel) }
}

unsafe fn msg_id_id(
    receiver: *mut c_void,
    sel: *mut c_void,
    arg: *mut c_void,
) -> *mut c_void {
    unsafe { msg_send_id_id_id(receiver, sel, arg) }
}

unsafe fn msg_id_id_id(
    receiver: *mut c_void,
    sel: *mut c_void,
    arg1: *mut c_void,
    arg2: *mut c_void,
) -> *mut c_void {
    unsafe { msg_send_id_id_id(receiver, sel, arg1, arg2) }
}

unsafe fn msg_void_id(receiver: *mut c_void, sel: *mut c_void, arg: *mut c_void) {
    unsafe { msg_send_void_id(receiver, sel, arg) }
}

unsafe fn msg_void(receiver: *mut c_void, sel: *mut c_void) {
    unsafe { msg_send_void(receiver, sel) }
}

unsafe fn msg_new_buffer(
    receiver: *mut c_void,
    sel: *mut c_void,
    bytes: *const c_void,
    length: u64,
    options: u64,
) -> *mut c_void {
    unsafe { msg_send_id_buf(receiver, sel, bytes, length, options) }
}

unsafe fn msg_new_library(
    receiver: *mut c_void,
    sel: *mut c_void,
    source: *mut c_void,
    opts: *mut c_void,
    error_out: *mut *mut c_void,
) -> *mut c_void {
    unsafe { msg_send_id_lib(receiver, sel, source, opts, error_out) }
}

unsafe fn msg_set_buffer(
    receiver: *mut c_void,
    sel: *mut c_void,
    buffer: *mut c_void,
    offset: u64,
    index: u64,
) {
    unsafe { msg_send_set_buf(receiver, sel, buffer, offset, index) }
}

unsafe fn msg_dispatch(
    receiver: *mut c_void,
    sel: *mut c_void,
    gx: u64,
    gy: u64,
    gz: u64,
    tx: u64,
    ty: u64,
    tz: u64,
) {
    unsafe { msg_send_dispatch(receiver, sel, gx, gy, gz, tx, ty, tz) }
}

// ═══════════════════════════════════════════════════════════════════
// NSString helpers
// ═══════════════════════════════════════════════════════════════════

unsafe fn nsstring(s: &str) -> *mut c_void {
    let cstr = CString::new(s).unwrap();
    let nsstring_class =
        unsafe { objc_getClass(b"NSString\0".as_ptr() as *const _) };
    let init_sel = unsafe { sel("stringWithUTF8String:") };
    let ns = unsafe {
        msg_send_string(nsstring_class, init_sel, cstr.as_ptr() as *const u8)
    };
    // Retain — autorelease pool would free this
    unsafe { msg_id(ns, selectors().retain) };
    ns
}

unsafe fn nsstring_read(ns: *mut c_void) -> String {
    let sels = selectors();
    let utf8 =
        unsafe { msg_id(ns, sels.utf8_string) as *const std::ffi::c_char };
    if utf8.is_null() {
        return "(null)".into();
    }
    unsafe {
        std::ffi::CStr::from_ptr(utf8)
            .to_string_lossy()
            .into_owned()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Internal Metal handles (not exposed to callers)
// ═══════════════════════════════════════════════════════════════════

struct MetalDevice {
    ptr: NonNull<c_void>,
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe {
            msg_void(self.ptr.as_ptr(), selectors().release);
        }
    }
}

struct MetalQueue {
    ptr: NonNull<c_void>,
}

impl Drop for MetalQueue {
    fn drop(&mut self) {
        unsafe {
            msg_void(self.ptr.as_ptr(), selectors().release);
        }
    }
}

/// Drop a Metal pipeline state.
fn drop_pipeline(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            msg_void(raw, selectors().release);
        }
    }
}

/// Drop a Metal buffer.
fn drop_buffer(raw: *mut c_void) {
    if !raw.is_null() {
        unsafe {
            msg_void(raw, selectors().release);
        }
    }
}

/// Get the contents pointer of a Metal buffer.
fn contents_of(raw: *mut c_void) -> *const c_void {
    if raw.is_null() {
        return std::ptr::null();
    }
    unsafe { msg_id(raw, selectors().contents) }
}

// ═══════════════════════════════════════════════════════════════════
// MetalBackend
// ═══════════════════════════════════════════════════════════════════

/// Metal GPU backend for Apple Silicon.
///
/// Holds a reference to the system default `MTLDevice` and a
/// persistent `MTLCommandQueue`. Created via [`MetalBackend::init`].
///
/// # Platform
///
/// Only available on macOS with the `metal` feature enabled.
/// The `MTLCreateSystemDefaultDevice` call returns null on
/// Intel Macs without a Metal-capable GPU (rare) or in CI
/// environments without GPU passthrough.
pub struct MetalBackend {
    device: MetalDevice,
    queue: MetalQueue,
}

impl MetalBackend {
    const STORAGE_MODE_SHARED: u64 = 0;
    /// Buffer slot for the naga-generated sizes buffer (u32 array of element counts).
    const SIZES_BUFFER_SLOT: msl::Slot = 30;
}

impl GpuBackend for MetalBackend {
    fn init() -> Result<Self> {
        let device_ptr = unsafe { MTLCreateSystemDefaultDevice() };
        if device_ptr.is_null() {
            return Err(GpuError::InitFailed(
                "MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU".into(),
            ));
        }

        let sels = selectors();
        let queue_ptr = unsafe { msg_id(device_ptr, sels.new_command_queue) };
        if queue_ptr.is_null() {
            unsafe {
                msg_void(device_ptr, sels.release);
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

    fn compile(&self, entry_point: &str, wgsl_source: &str) -> Result<ComputePipeline> {
        // Step 0: Translate WGSL → MSL via naga
        eprintln!("[borsalino::metal] compile: entry_point={entry_point}");
        let module = wgsl::parse_str(wgsl_source).map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: e.emit_to_string(wgsl_source),
        })?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator.validate(&module).map_err(|e| GpuError::CompileFailed {
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

        let (msl_source, _) = msl::write_string(
            &module,
            &info,
            &msl_opts,
            &msl::PipelineOptions::default(),
        )
        .map_err(|e| GpuError::CompileFailed {
            entry: entry_point.into(),
            message: format!("MSL emission failed: {e}"),
        })?;

        eprintln!("[borsalino::metal] MSL generated ({} bytes):\n{msl_source}", msl_source.len());

        let sels = selectors();
        let dev = self.device.ptr.as_ptr();

        unsafe {
            // Step 1: MTLLibrary from source
            eprintln!("[borsalino::metal] creating MTLLibrary...");
            let ns_src = nsstring(&msl_source);
            eprintln!("[borsalino::metal] nsstring created");
            let mut err: *mut c_void = std::ptr::null_mut();
            let library = msg_new_library(
                dev,
                sels.new_library_with_source,
                ns_src,
                std::ptr::null_mut(),
                &mut err,
            );
            msg_void(ns_src, sels.release);
            eprintln!("[borsalino::metal] library created, library={library:p}");

            if library.is_null() {
                let msg = if !err.is_null() {
                    let desc = msg_id(err, sels.localized_description);
                    let s = nsstring_read(desc);
                    msg_void(err, sels.release);
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
            eprintln!("[borsalino::metal] looking up function '{entry_point}'...");
            let ns_entry = nsstring(entry_point);
            eprintln!("[borsalino::metal] entry nsstring created");
            let func = msg_id_id(library, sels.new_function_with_name, ns_entry);
            eprintln!("[borsalino::metal] func={func:p}");
            msg_void(ns_entry, sels.release);

            if func.is_null() {
                msg_void(library, sels.release);
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: format!("function '{entry_point}' not found in compiled library"),
                });
            }

            // Step 3: MTLComputePipelineState
            eprintln!("[borsalino::metal] creating compute pipeline state...");
            let mut perr: *mut c_void = std::ptr::null_mut();
            let pipeline = msg_id_id_id(
                dev,
                sels.new_compute_pipeline_state,
                func,
                &mut perr as *mut _ as *mut c_void,
            );

            if pipeline.is_null() {
                let msg = if !perr.is_null() {
                    let desc = msg_id(perr, sels.localized_description);
                    let s = nsstring_read(desc);
                    msg_void(perr, sels.release);
                    s
                } else {
                    "unknown pipeline error".into()
                };
                msg_void(func, sels.release);
                msg_void(library, sels.release);
                return Err(GpuError::PipelineFailed {
                    entry: entry_point.into(),
                    message: msg,
                });
            }

            // Release intermediates
            msg_void(func, sels.release);
            msg_void(library, sels.release);

            Ok(ComputePipeline {
                raw: pipeline,
                drop_fn: drop_pipeline,
            })
        }
    }

    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = data.len() * element_size;
        let sels = selectors();

        let buf = unsafe {
            msg_new_buffer(
                self.device.ptr.as_ptr(),
                sels.new_buffer_with_bytes,
                data.as_ptr() as *const c_void,
                byte_len as u64,
                Self::STORAGE_MODE_SHARED,
            )
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

    fn create_buffer_uninit<T: bytemuck::Pod>(&self, len: usize) -> Result<GpuBuffer> {
        let element_size = std::mem::size_of::<T>();
        let byte_len = len * element_size;
        let sels = selectors();

        let buf = unsafe {
            msg_new_buffer(
                self.device.ptr.as_ptr(),
                sels.new_buffer_with_bytes,
                std::ptr::null(),
                byte_len as u64,
                Self::STORAGE_MODE_SHARED,
            )
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
        let sels = selectors();
        unsafe {
            let cmd = msg_id(self.queue.ptr.as_ptr(), sels.command_buffer);
            if cmd.is_null() {
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLCommandBuffer".into(),
                });
            }

            let encoder = msg_id(cmd, sels.compute_command_encoder);
            if encoder.is_null() {
                msg_void(cmd, sels.release);
                return Err(GpuError::DispatchFailed {
                    message: "failed to create MTLComputeCommandEncoder".into(),
                });
            }

            // Set pipeline
            msg_void_id(encoder, sels.set_compute_pipeline_state, pipeline.raw);

            // Bind sizes buffer (naga runtime array element counts)
            let sizes: Vec<u32> = buffers.iter().map(|b| b.len as u32).collect();
            let sizes_buf = msg_new_buffer(
                self.device.ptr.as_ptr(),
                sels.new_buffer_with_bytes,
                sizes.as_ptr() as *const c_void,
                (sizes.len() * 4) as u64,
                Self::STORAGE_MODE_SHARED,
            );
            msg_set_buffer(
                encoder,
                sels.set_buffer_offset_at_index,
                sizes_buf,
                0,
                Self::SIZES_BUFFER_SLOT as u64,
            );

            // Bind buffers
            for (i, buf) in buffers.iter().enumerate() {
                msg_set_buffer(
                    encoder,
                    sels.set_buffer_offset_at_index,
                    buf.raw,
                    0,
                    i as u64,
                );
            }

            // Dispatch
            msg_dispatch(
                encoder,
                sels.dispatch_threadgroups,
                workgroups.0 as u64,
                workgroups.1 as u64,
                workgroups.2 as u64,
                threads_per_group.0 as u64,
                threads_per_group.1 as u64,
                threads_per_group.2 as u64,
            );

            // Finish
            msg_void(encoder, sels.end_encoding);
            msg_void(cmd, sels.commit);
            msg_void(cmd, sels.wait_until_completed);
            msg_void(cmd, sels.release);
            msg_void(sizes_buf, sels.release);
        }

        Ok(())
    }

    fn read_buffer<T: bytemuck::Pod>(&self, buffer: &GpuBuffer) -> Result<Vec<T>> {
        let contents = (buffer.contents_fn)(buffer.raw) as *const T;
        if contents.is_null() {
            return Err(GpuError::BufferReadFailed {
                message: "buffer contents pointer is null".into(),
            });
        }
        let slice = unsafe { std::slice::from_raw_parts(contents, buffer.len) };
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
        let input = backend.create_buffer(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
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
