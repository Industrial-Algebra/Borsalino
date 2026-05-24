# Borsalino — Handoff Document

**Date:** 2026-05-19
**Repository:** `/home/elliotthall/working/industrial-algebra/Borsalino`
**Status:** Metal + Vulkan backends complete, verification Phase 2 partial

---

## 1. What Borsalino Is

A thin GPU compute abstraction for the Industrial Algebra ecosystem. One trait (`GpuBackend`), two backends (Metal + Vulkan), zero ceremony. Users write WGSL compute kernels, dispatch them synchronously, read results back. No bind groups, no pipeline layouts, no descriptor sets, no async runtime.

## 2. Current State

| Component | Status |
|---|---|
| `GpuBackend` trait | Complete |
| Error types (`GpuError`, `Result<T>`) | Complete (10 variants) |
| Metal backend (`metal.rs`) | Complete — WGSL→MSL via naga, raw objc_msgSend FFI |
| Vulkan backend (`vulkan.rs`) | Complete — WGSL→SPIR-V via naga, raw ash FFI |
| `ComputePipeline` / `GpuBuffer` handles | Opaque pointer + drop function pattern |
| WGSL shader language | Standard `@group(0) @binding(N)` syntax |
| karpal-verify GPU obligation bundles | Phase 2: Property types + bundles for 3 kernels |
| Examples | `hello_compute` (add_one), `saxpy` (1024 elements) |
| CI | None yet |
| crate.io readiness | Not published |

**Builds:** All feature combinations pass with zero warnings:
- `cargo check` / `cargo check --features metal` / `cargo check --features vulkan` / `cargo check --features verify`
- `cargo clippy` passes on all
- `cargo test --features "vulkan,verify"` — 9/9 tests pass

## 3. File Map

```
Borsalino/
├── Cargo.toml              # edition 2024, AGPL-3.0, deps: naga, thiserror, bytemuck
├── README.md               # Full documentation with quick start
├── HANDOFF.md              # This file
├── examples/
│   ├── hello_compute.rs    # Simplest kernel: add_one on 4 elements
│   └── saxpy.rs            # SAXPY: fused multiply-add on 1024 elements
├── docs/
│   ├── verification-integration.md   # Karpal verification design
│   └── plans/
│       ├── 2026-05-19-vulkan-backend-design.md
│       └── 2026-05-19-vulkan-backend-plan.md
└── src/
    ├── lib.rs              # Trait + handle types + stub backend (300+ lines)
    ├── error.rs            # GpuError enum (thiserror), Result<T> alias
    ├── metal.rs            # MetalBackend + raw objc_msgSend FFI (700+ lines)
    ├── vulkan.rs           # VulkanBackend + ash FFI (1000+ lines)
    ├── verify.rs           # karpal-verify GPU obligation bundles (200 lines)
    └── main.rs             # SAXPY smoke test
```

## 4. Architecture Decisions

### 4.1 Opaque handles with stored drop functions

`ComputePipeline` and `GpuBuffer` don't carry backend-specific types. They hold a `*mut c_void` and a `fn(*mut c_void)` drop function. Each backend wraps its native handles behind opaque pointers and provides the drop logic.

### 4.2 WGSL shader language via naga

Kernels are authored in WGSL with `@group(0) @binding(N)` buffer declarations. Naga translates to each backend's native format:
- Metal: WGSL → MSL → Metal compiler
- Vulkan: WGSL → SPIR-V → vkCreateComputePipelines

### 4.3 Raw FFI

Both backends use raw FFI with no wrapper crates:
- Metal: `objc_msgSend` extern declarations, 18 cached selectors
- Vulkan: `ash` (generated Vulkan bindings), no safety wrappers

### 4.4 Synchronous dispatch

Every `dispatch()` call blocks until the GPU completes. No callbacks, no fences exposed to the caller.

### 4.5 Vulkan resources

Pre-allocated at init:
- 1× VkPipelineLayout (8 storage buffer bindings)
- 1× VkDescriptorSetLayout
- 1× VkDescriptorSet (pre-allocated, updated per dispatch)
- 1× VkCommandPool (RESET_COMMAND_BUFFER_BIT)
- 1× VkDescriptorPool

Per dispatch: allocate cmd buffer → begin → bind pipeline → update descriptor set → bind set → dispatch → barrier → end → submit → fence wait → free cmd buffer.

## 5. Coding Conventions

- **Copyright header:** `// Copyright (C) 2026 Industrial Algebra` + `// SPDX-License-Identifier: AGPL-3.0-only`
- **Error style:** `thiserror::Error` derive, structured variants
- **Lints:** `#![warn(missing_docs)]`, `#![warn(clippy::all)]`
- **Edition:** 2024, MSRV 1.85
- **License:** AGPL-3.0-only
- **Git flow:** feature branch → PR to develop → develop → release PR to main

## 6. Test Strategy

### Vulkan tests (5 tests, all pass on real hardware)

1. `device_init` — finds a Vulkan compute device
2. `add_one_kernel` — WGSL kernel, dispatch, readback
3. `vector_scale_1024` — 1024 elements, 4 workgroups
4. `compile_error` — invalid WGSL → CompileFailed
5. `roundtrip_empty` — zero-init buffer survives roundtrip

### Metal tests (3 tests, macOS only — untested on Apple Silicon)

1. `device_init` — confirms MTLDevice or skips
2. `add_one_kernel` — compile, dispatch, readback
3. `vector_scale_1024` — 1024 elements, scale by 2.5

### Verify tests (4 tests)

1-3. Bundle structure validation for add_one, scale, saxpy
4. Cross-backend export (SMT, Lean, Kani)

## 7. Quick Start

```rust
use borsalino::GpuBackend;

let wgsl = r#"
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(256)
    fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        output[i] = input[i] + 1.0;
    }
"#;

let gpu = borsalino::init()?;
let pipeline = gpu.compile("add_one", wgsl)?;
let input = gpu.create_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
let output = gpu.create_buffer_uninit::<f32>(4)?;
gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1))?;
let result: Vec<f32> = gpu.read_buffer(&output)?;
assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
```

## 8. Next Steps

### Must do
1. **Test Metal backend on Apple Silicon.** Run `cargo test --features metal` on M3 Mac.
2. **Kani harnesses.** Buffer roundtrip, alignment, workgroup divisibility.
3. **Miri integration.** Buffer lifecycle safety (no UAF, no double-free).

### Should do
4. **Real Industrial Algebra kernel.** Geometric product of 32-element multivectors.
5. **Benchmark dispatch overhead.** Compare per-dispatch encoder cost vs cached encoder.
6. **CI workflow.** GitHub Actions with Vulkan tests on self-hosted runner.

### Could do
7. **`dispatch_verified()` with `Proven<>` gates.** Phase 3 of verification.
8. **amari-flynn statistical verification.** Kernel determinism, memory leak detection.
9. **Vulkan timestamp queries.** `gpu.timestamp() -> Result<u64>` for profiling.
10. **crate.io publication.**
