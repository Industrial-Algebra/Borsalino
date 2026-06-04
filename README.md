# Borsalino

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

AGPL-3.0-only OR Commercial. Copyright (C) 2026 Industrial Algebra.

Dual-licensed: use under AGPL v3 for open-source projects, or obtain a
commercial license for proprietary use. See [LICENSE](./LICENSE) and
[LICENSE-COMMERCIAL](./LICENSE-COMMERCIAL).
