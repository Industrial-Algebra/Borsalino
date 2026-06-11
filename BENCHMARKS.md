# Borsalino Benchmarks

Cross-platform GPU compute benchmarks. Run with:

```sh
cargo run --example bench --features metal --release    # macOS
cargo run --example bench --features vulkan --release   # Linux / Windows
```

## SAXPY Throughput (fused multiply-add, 16M elements)

Higher is better.

| Platform | GPU | Memory | GFLOPS |
|---|---|---|---|
| **NVIDIA RTX 5080** | Blackwell (16 GB) | VRAM (staging) | **66.8** |
| **Apple M3 Pro** | Apple Silicon | Unified | **30.4** |
| **NVIDIA GB10** | Grace Blackwell | NVLink-C2C unified | **29.5** |
| AMD Radeon (mobile) | Integrated | Unified | 16.7 |

## Dispatch Latency

Single workgroup × 256 threads, lower is better.

| Platform | Latency |
|---|---|
| **NVIDIA RTX 5080** | **31 µs** |
| **NVIDIA GB10** | **46 µs** |
| **AMD Radeon (mobile)** | **45 µs** |
| Apple M3 Pro | 142 µs |

## Pipeline Compilation

WGSL → native shader compilation, lower is better.

| Platform | Time |
|---|---|
| NVIDIA RTX 5080 | 0.026 ms |
| NVIDIA GB10 | 0.028 ms |
| Apple M3 Pro | 0.056 ms |
| AMD Radeon (mobile) | 0.33 ms |

## Buffer I/O

### Create (host → GPU, GB/s)

| Size | M3 Pro | GB10 | RTX 5080 |
|---|---|---|---|
| 1K f32 | 2.3 | 0.01 | 0.01 |
| 16K f32 | 6.7 | 0.16 | 0.27 |
| 256K f32 | 16.2 | 1.13 | 1.83 |
| 1M f32 | 17.2 | 1.97 | 2.62 |

### Read (GPU → host, GB/s)

| Size | M3 Pro | GB10 | RTX 5080 |
|---|---|---|---|
| 1K f32 | 50.7 | 0.09 | 0.14 |
| 16K f32 | 43.6 | 1.43 | 2.00 |
| 256K f32 | 79.5 | 8.15 | 5.53 |
| 1M f32 | 83.5 | 7.26 | 5.14 |

## Batched Dispatch (`dispatch_many`)

256 dispatches per command buffer, SAXPY kernel.

| Size | M3 Pro | GB10 | RTX 5080 |
|---|---|---|---|
| 16K el | 18.5 GFLOPS | 96.7 GFLOPS | 106.6 GFLOPS |
| 256K el | 47.6 GFLOPS | 211.3 GFLOPS | 293.0 GFLOPS |
| 1M el | 41.5 GFLOPS | **395.2 GFLOPS** | **476.9 GFLOPS** |
| 16M el | 33.9 GFLOPS | 59.6 GFLOPS | 70.5 GFLOPS |

Per-dispatch overhead at 256 batched:

| Platform | Single | Batched ×256 | Reduction |
|---|---|---|---|
| GB10 | 46 µs | **0.4 µs** | 115× |
| RTX 5080 | 31 µs | **0.5 µs** | 62× |
| M3 Pro | 142 µs | **1.9 µs** | 75× |

## Tiled Matrix Multiply (2D workgroup, shared memory)

1024×1024 matrix, 16×16 tile size, 64×64 workgroups.

| Platform | GFLOPS |
|---|---|
| **NVIDIA RTX 5080** | **1,120** |
| **NVIDIA GB10** | **1,097** |
| **AMD Radeon (iGPU)** | **278** |
| Apple M3 Pro | 184 |

The tiled kernel demonstrates Borsalino's compute-bound ceiling — approaching
1+ TFLOPS on discrete hardware. Compare with element-wise SAXPY at 17-67 GFLOPS
— the 2D shared-memory pattern delivers 6-40× higher throughput.

## GPU Timestamp Resolution

| Platform | Resolution |
|---|---|
| Apple M3 Pro | ~1.0 ns |
| NVIDIA GB10 | ~140 µs (coarse) |
| NVIDIA RTX 5080 | ~33 µs |
| AMD Radeon (mobile) | ~50 µs |

## Test Hardware

| Platform | CPU | GPU | RAM | OS |
|---|---|---|---|---|
| AMD laptop | AMD Ryzen | Integrated Radeon | 16 GB DDR5 | Ubuntu 25.10 |
| GB10 DGX | Grace ARM64 | Blackwell (NVLink-C2C) | 128 GB LPDDR5X | Ubuntu 24.04 |
| RTX 5080 | AMD Ryzen 9 | RTX 5080 16 GB | 64 GB DDR5 | Ubuntu 24.04 |
| M3 Pro | Apple M3 Pro | Integrated (18-core) | 36 GB unified | macOS 15 |

*Benchmarks run 2026-06-03 (v0.1.0) / 2026-06-11 (v0.2.0). Results are single-run warm averages; production workloads may vary.*
