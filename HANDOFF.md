# Borsalino — Handoff Document

**Date:** 2026-05-19
**Repository:** `/home/elliotthall/working/industrial-algebra/Borsalino`
**Author:** elliotthall (with AI assistance)
**Status:** Metal backend working, Vulkan stub only

---

## 1. What Borsalino Is

A thin GPU compute abstraction for the Industrial Algebra ecosystem. One trait (`GpuBackend`), platform-native backends, zero ceremony. Users write Metal Shading Language kernels, dispatch them synchronously, read results back. No bind groups, no pipeline layouts, no descriptor sets, no async runtime.

Born from frustration with `wgpu`'s ceremony and validation from Quanta's approach (raw FFI works, but we don't need five backends or physics metaphors).

## 2. Current State

| Component | Status |
|-----------|--------|
| `GpuBackend` trait | Complete |
| Error types (`GpuError`, `Result<T>`) | Complete (12 variants) |
| Metal backend (`metal.rs`) | Complete — compiles clean, tests included |
| Vulkan backend | Stub only — `VulkanStub` struct, always returns `NoBackend` |
| `ComputePipeline` / `GpuBuffer` handles | Opaque pointer + drop function pattern |
| crate.io readiness | Not published |
| CI | None yet |

**Builds:** `cargo check --features metal` and `cargo check` both pass with zero warnings and zero clippy errors.

## 3. File Map

```
Borsalino/
├── Cargo.toml          # edition 2024, AGPL-3.0, deps: thiserror + bytemuck
├── src/
│   ├── lib.rs          # Trait + handle types + stub backends (341 lines)
│   ├── error.rs        # GpuError enum (thiserror), Result<T> alias (104 lines)
│   ├── metal.rs        # MetalBackend + raw objc_msgSend FFI (650 lines)
│   └── main.rs         # SAXPY smoke test (60 lines)
```

## 4. Architecture Decisions

### 4.1 Opaque handles with stored drop functions

`ComputePipeline` and `GpuBuffer` don't carry backend-specific types. They hold a `*mut c_void` and a `fn(*mut c_void)` drop function. Each backend stores its native handles behind opaque pointers and provides the drop logic. Thread-safe: both types are `Send + Sync`.

```rust
pub struct GpuBuffer {
    pub(crate) raw: *mut c_void,
    pub(crate) len: usize,
    pub(crate) element_size: usize,
    pub(crate) drop_fn: fn(*mut c_void),
    pub(crate) contents_fn: fn(*mut c_void) -> *const c_void,
}
```

This avoids circular dependencies between `lib.rs` and backend modules — the handle types don't reference backend types at all.

### 4.2 Raw objc_msgSend FFI

The Metal backend calls the Objective-C runtime directly. Three `extern "C"` functions, 18 cached selectors in a `OnceLock<Selectors>`, typed `std::mem::transmute` wrappers for each message signature. Zero crate dependencies beyond `std`.

Key selectors cached:
- `newBufferWithBytes:length:options:` (buffer allocation, storage mode shared)
- `newLibraryWithSource:options:error:` (MSL compilation)
- `newFunctionWithName:` (kernel function lookup)
- `newComputePipelineStateWithFunction:error:` (pipeline creation)
- `commandBuffer` / `computeCommandEncoder` (encoder lifecycle)
- `setComputePipelineState:` / `setBuffer:offset:atIndex:` (binding)
- `dispatchThreadgroups:threadsPerThreadgroup:` (dispatch)
- `endEncoding` / `commit` / `waitUntilCompleted` (sync)
- `contents` (buffer readback)
- `retain` / `release` (memory management)

### 4.3 NSString handling

`nsstring(s: &str)` creates an NSString from a Rust string, explicitly retains it (autorelease pools may drain). `nsstring_read(ns)` reads UTF-8 back. Both are `unsafe`.

### 4.4 Memory management

All Metal objects (`MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLComputePipelineState`) are wrapped in structs with `Drop` impls that call `objc_msgSend(release)`. `MetalBackend` itself does NOT implement Drop — its fields handle their own cleanup independently.

### 4.5 Dispatch: synchronous by design

Every `dispatch()` call creates a command buffer, creates an encoder, sets pipeline, binds buffers, dispatches, ends encoding, commits, and calls `waitUntilCompleted`. The caller blocks until the GPU is done. This is the right tradeoff for compute-heavy Industrial Algebra workloads where you want results in registers, not callback chains.

## 5. Coding Conventions (from Schubert)

- **Copyright header:** `// Copyright (C) 2026 Industrial Algebra` + `// SPDX-License-Identifier: AGPL-3.0-only`
- **Error style:** `thiserror::Error` derive, structured variants with context fields
- **Result alias:** `pub type Result<T> = std::result::Result<T, GpuError>;`
- **Lints:** `#![warn(missing_docs)]`, `#![warn(clippy::all)]`
- **Feature gates:** `#[cfg(all(feature = "metal", target_os = "macos"))]`
- **Doc style:** Module-level doc with architecture overview, per-item docs on all public types
- **Edition:** 2024, MSRV 1.85
- **License:** AGPL-3.0-only

## 6. Test Strategy

Three tests in `metal.rs` (gated behind `#[cfg(test)]`):

1. **`device_init`** — confirms `MTLCreateSystemDefaultDevice` returns non-null (or fails gracefully in CI)
2. **`add_one_kernel`** — compiles MSL, dispatches on [1,2,3,4], asserts [2,3,4,5]
3. **`vector_scale_1024`** — 1024 elements, 4 workgroups of 256, scale by 2.5, bitwise compare

Tests skip gracefully when no Metal GPU is present (CI, headless Linux). On macOS with Apple Silicon, run with:

```sh
cargo test --features metal
```

## 7. Usage Example

```rust
use borsalino::GpuBackend;

let gpu = borsalino::init()?;

let msl = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void add_one(device const float* input  [[buffer(0)]],
                        device float*       output [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
        output[id] = input[id] + 1.0;
    }
"#;

let pipeline = gpu.compile("add_one", msl)?;
let input = gpu.create_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
let output = gpu.create_buffer_uninit::<f32>(4)?;
gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1))?;
let result: Vec<f32> = gpu.read_buffer(&output)?;
assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
```

## 8. Immediate Next Steps

### Must do
1. **Test on real Apple Silicon.** Push to a Mac, run `cargo test --features metal`. Confirm `add_one_kernel` and `vector_scale_1024` both pass.
2. **Verify the `retain`/`release` lifecycle.** Run under Miri if possible, or Instruments (Metal leak detector). The explicit `retain` on NSString followed by manual `release` in MetalBuffer/MetalPipeline drops is correct in principle but warrants stress testing.

### Should do
3. **Add `bytemuck::Pod` derive to applicable Industrial Algebra types.** The trait requires `Pod` for buffer elements. Amari's multivector types should derive it.
4. **Add a `#[test]` that exercises a realistic Industrial Algebra kernel** — e.g., geometric product of two 32-element multivectors.
5. **Benchmark dispatch overhead** vs raw Metal FFI. The encoder+commit+wait pattern per dispatch has a cost. For repeated dispatches, caching the command buffer/encoder might help.

### Could do
6. **Vulkan backend.** Implement `GpuBackend` for a `VulkanBackend` using `vulkano` (already in Cargo.toml as optional dep). Key functions: `vkCreateInstance`, `vkEnumeratePhysicalDevices`, `vkCreateDevice`, MSL→SPIR-V via `glslangValidator` subprocess or naga, `vkCreateComputePipelines`, `vkCmdDispatch`, buffer mapping.
7. **Timestamps.** Add `gpu.timestamp() -> Result<u64>` for profiling. Metal has `MTLCommandBuffer.gpuEndTime`, Vulkan has timestamp queries.
8. **Multiple command queues.** `dispatch_async()` returning a `Pulse` handle that can be waited on later.

## 9. Design Notes for Future Work

### Why not a generic backend init?

`init() -> Result<impl GpuBackend>` failed because the fallback path couldn't infer `T`. Current approach: three cfg-gated `init()` functions returning concrete types (`MetalBackend`, `VulkanStub`, `NoBackendStub`). If adding more backends, add more cfg-gated init functions.

### Why raw FFI instead of metal-rs?

`metal-rs` would have added ~30 transitive dependencies and an `objc` / `block` runtime dependency. The FFI surface for compute-only Metal is genuinely ~15 functions. The tradeoff is that we maintain our own FFI bindings, but they don't change often (Metal ABI is stable).

### The contents_fn field

`GpuBuffer.contents_fn` exists for `read_buffer`: on Metal, it's `msg_id(buffer, contents)`. On Vulkan, it'll be `vkMapMemory`. The field has `#[allow(dead_code)]` because in the `lib.rs` definition it appears unused (the Metal backend reads it through dynamic dispatch).

### SPIR-V translation strategy

When implementing Vulkan, the cleanest path for MSL→SPIR-V is:
1. Call `glslangValidator` as a subprocess (requires Vulkan SDK)
2. Or bundle `naga` (already in wgpu's ecosystem, could be used standalone)
3. Or write a minimal MSL parser → SPIR-V emitter (ambitious, probably not worth it)

Option 1 is simplest and matches the raw-FFI philosophy: one `std::process::Command`, feed MSL on stdin, get SPIR-V on stdout.

## 10. Relationship to Other Industrial Algebra Crates

- **Amari** (`amari-enumerative`, `amari-core`): Will use Borsalino for GPU-accelerated multivector operations. The `bytemuck::Pod` requirement means Amari's scalar types need `Pod` derives.
- **Schubert** (`schubert`): No direct GPU needs, but the coding conventions are the template.
- **Minoru** (`Minoru`): Could use Borsalino for batch ordinal arithmetic checks (phase 3 of the Minoru spec).

## 11. Cargo.toml Features

```toml
[features]
default = []
metal = []                                              # macOS only, raw FFI
vulkan = ["dep:vulkano", "dep:vulkano-shaders"]         # Linux/Windows, not yet implemented
```

No `default = ["metal"]` — callers must opt in. On macOS, build with `--features metal`. On Linux/Windows, the `vulkan` feature will activate the Vulkan stub (and eventually the real backend).
