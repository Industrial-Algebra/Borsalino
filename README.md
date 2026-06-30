# Borsalino

[![crates.io](https://img.shields.io/crates/v/borsalino)](https://crates.io/crates/borsalino)
[![docs.rs](https://img.shields.io/docsrs/borsalino)](https://docs.rs/borsalino)
[![CI](https://github.com/Industrial-Algebra/Borsalino/actions/workflows/ci.yml/badge.svg)](https://github.com/Industrial-Algebra/Borsalino/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

Thin GPU compute abstraction for the Industrial Algebra ecosystem.

> One trait, two backends, zero ceremony.

Write WGSL compute kernels. Dispatch them synchronously on Metal or Vulkan.
Read results back. No bind groups, no pipeline layouts, no descriptor sets,
no async runtime.

## Quick Start

```rust
use borsalino::GpuBackend;

// WGSL compute kernel
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

Run with:

```sh
cargo run --features metal --example hello_compute    # macOS
cargo run --features vulkan --example hello_compute   # Linux / Windows
```

## Backends

| Backend | Platform | Feature | Status |
|---|---|---|---|
| Metal | macOS (Apple Silicon) | `metal` | ✅ Active — raw `objc_msgSend` FFI |
| Vulkan | Linux, Windows | `vulkan` | ✅ Active — raw `ash` FFI |
| Stub | Any | (none) | Returns `NoBackend` — safe fallback |

## Features

| Feature | What it enables |
|---|---|
| `metal` | Metal backend (macOS only) |
| `vulkan` | Vulkan backend via ash (Linux / Windows) |
| `verify` | karpal-verify 0.5 GPU obligation bundles (SMT, Lean, Kani export) |

## Architecture

```
GpuBackend trait (7 methods)
    │
    ├── MetalBackend     (metal.rs)
    │   ├── naga WGSL → MSL translation
    │   └── objc_msgSend FFI (19 selectors, 0 Metal crate deps)
    │
    └── VulkanBackend    (vulkan.rs)
        ├── naga WGSL → SPIR-V translation
        └── ash FFI (Vulkan 1.3)
```

Opaque handle types (`ComputePipeline`, `GpuBuffer`) carry raw pointers and
backend-specific drop functions — no coupling between `lib.rs` and backend
modules.

## Shader Language

Kernels are authored in **WGSL** (WebGPU Shading Language). Borsalino
translates to each backend's native format via [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga):

- Metal: WGSL → MSL → Metal compiler
- Vulkan: WGSL → SPIR-V → vkCreateComputePipelines

Buffer bindings use `@group(0) @binding(N)` in WGSL, mapped to dispatch
buffer position: `buffers[0]` → `@binding(0)`, `buffers[1]` → `@binding(1)`.

## Memory Strategy

Borsalino auto-detects your hardware and picks the optimal memory layout:

| GPU type | Detection | Memory | Behaviour |
|---|---|---|---|
| Apple Silicon M-series | Unified | Host-visible, coherent | Zero-copy between CPU and GPU |
| AMD integrated (APU) | Unified | Host-visible, coherent | Zero-copy |
| NVIDIA Grace Blackwell (GB10) | Unified | Host-visible, coherent | Zero-copy |
| NVIDIA RTX / AMD RDNA / Intel Arc | Discrete (auto) | Device-local VRAM + staging | Automatic PCIe transfers |

For explicit control:

```rust
use borsalino::MemoryStrategy;

let gpu = borsalino::init()?;                                      // auto-detect
let gpu = borsalino::init_device_local()?;                         // force VRAM
let gpu = VulkanBackend::init_with_strategy(MemoryStrategy::Unified)?; // force unified
```

## Batched Dispatch

Chain multiple dispatches into a single command buffer with
[`dispatch_many`](GpuBackend::dispatch_many):

```rust
use borsalino::{DispatchSpec, GpuBackend};

gpu.dispatch_many(&[
    DispatchSpec { pipeline: &p1, buffers: &[&buf_a, &buf_b],
                   workgroups: (4, 1, 1), threads_per_group: (256, 1, 1) },
    DispatchSpec { pipeline: &p2, buffers: &[&buf_b, &buf_c],
                   workgroups: (4, 1, 1), threads_per_group: (256, 1, 1) },
])?;
```

Batching amortises command-buffer allocation overhead. On RTX 5080,
256 dispatches per buffer drops per-dispatch latency from 37 us to
**0.5 us** (75x faster). On GB10: 46 us to **1.0 us** (46x faster).
Peak throughput: **577 GFLOPS** (RTX 5080, 1M elements batched).

## Persistent Buffers

For iterative workloads (ML training, physics simulation), buffers can
live on the GPU across dispatches without CPU readback:

```rust
let weights = gpu.create_device_buffer(&model_weights)?;
let output = gpu.create_device_buffer_uninit::<f32>(N)?;

// Dispatch many times — no CPU round-trip
for _ in 0..1000 {
    gpu.dispatch(&pipeline, &[&weights, &output], (wgs, 1, 1))?;
}

// Read once at the end
let result = gpu.read_buffer(&output)?;
```

On unified memory, `create_device_buffer` is identical to `create_buffer`
(zero copy). On discrete GPUs, it allocates VRAM and uses one-shot staging
only on final readback.

See [BENCHMARKS.md](./BENCHMARKS.md) for full cross-platform performance data.

## Verification

GPU safety properties are encoded as karpal-verify 0.5 obligation bundles
(feature `verify`, fetches from crates.io):

```rust
use borsalino::verify::{add_one_obligations, IsBufferAlignedTo16, Property};

let bundle = add_one_obligations();
assert!(bundle.obligations().iter().any(|o| o.property == IsBufferAlignedTo16::NAME));
```

Bundles export to SMT-LIB2, Lean 4, and Kani verification backends.

## Examples

| Example | Description | Run with |
|---|---|---|
| `hello_compute` | add_one kernel on 4 elements | `cargo run --example hello_compute --features vulkan` |
| `saxpy` | SAXPY (a·x + y) on 1024 elements | `cargo run --example saxpy --features vulkan` |
| `bench` | Cross-platform GPU benchmarks | `cargo run --example bench --features vulkan --release` |
| `dispatch_profile` | Per-component dispatch cost profiling | `cargo run --example dispatch_profile --features vulkan --release` |
| `tiled_matmul` | 2D tiled matrix multiply with shared memory | `cargo run --example tiled_matmul --features vulkan --release` |
| `ia_geometric_product` | IA kernel: 32-blade geometric product (5D GA) | `cargo run --example ia_geometric_product --features vulkan --release` |

## Shader Caching

[`compile_cached`](GpuBackend::compile_cached) skips naga translation on repeat
calls by caching compiled SPIR-V (Vulkan) or MSL (Metal) to disk
(`~/.cache/borsalino/`):

```rust
let pipeline = gpu.compile_cached("add_one", wgsl)?;
// Second call loads from disk — zero naga overhead
let pipeline2 = gpu.compile_cached("add_one", wgsl)?;
```

## Verified Dispatch

[`dispatch_verified`](GpuBackend::dispatch_verified) gates dispatches behind a
runtime workgroup divisibility proof:

```rust
let config = DispatchConfig { total_threads: 1024, threads_per_group: 256 };
let proof = config.verify()?;
gpu.dispatch_verified(&pipeline, &buffers, (4, 1, 1), (256, 1, 1), &proof)?;
```

## Verification (Miri + Kani)

Buffer lifecycle safety is verified under Miri (undefined behaviour detection)
and Kani (bounded model checking):

```sh
cargo +nightly miri test --features vulkan buffer_lifecycle
cargo kani --harness buffer_alignment_boundary
```

CI runs both on label-gated PRs (`run-miri`, `run-kani`).

## Async Dispatch

Non-blocking GPU execution via [`dispatch_async`](GpuBackend::dispatch_async):

```rust
let p1 = gpu.dispatch_async(&pipe_a, &[&buf_x], (64, 1, 1))?;
let p2 = gpu.dispatch_async(&pipe_b, &[&buf_y], (64, 1, 1))?;
// Both running concurrently on GPU. Do CPU work here...
p1.wait();
p2.wait();
let result = gpu.read_buffer(&buf_out)?;
```

[`Pulse`] handles are `Send + Sync`. Drop performs implicit join
(blocks until the GPU dispatch completes).

## Profiling

Measure GPU-side execution time with [`timestamp`](GpuBackend::timestamp):

```rust
let t0 = gpu.timestamp()?;
gpu.dispatch(&pipeline, &buffers, (wgs, 1, 1))?;
let gpu_ns = gpu.timestamp()? - t0;
```



## 2D / 3D Dispatch

Borsalino supports multi-dimensional workgroup grids for tile-based
algorithms (matrix multiply, convolution, attention):

```rust
// 2D workgroup grid: 64×64 workgroups, each 16×16 threads
gpu.dispatch_ex(
    &pipeline, &buffers,
    (64, 64, 1),      // workgroups in (x, y, z)
    (16, 16, 1),       // threads per workgroup
)?;
```

Combine with WGSL shared memory (`var<workgroup>`) and barriers
(`workgroupBarrier()`) for tiled algorithms. See
`examples/tiled_matmul.rs` for a complete 2D tiled matrix multiply
(278 GFLOPS on AMD iGPU, 1024×1024, ~1 TFLOPS on NVIDIA RTX).

## Testing

```sh
# Vulkan backend (Linux / Windows)
cargo test --features vulkan

# Verification obligations
cargo test --features verify

# Both
cargo test --features "verify,vulkan"

# Metal backend (macOS only)
cargo test --features metal
```

## License

Apache-2.0. Copyright (C) 2026 Industrial Algebra.

Contributors must sign the [CLA](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md).
