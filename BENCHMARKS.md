# Borsalino Benchmarks

Cross-platform GPU compute benchmarks. Run with:

```sh
cargo run --example bench --features metal --release    # macOS
cargo run --example bench --features vulkan --release   # Linux / Windows
```

## SAXPY Throughput (fused multiply-add, GFLOPS)

Higher is better. 16M elements, 256 threads per workgroup.

| Platform | GPU | Memory | GFLOPS |
|---|---|---|---|
| **NVIDIA RTX 5080** | Blackwell (16 GB) | VRAM (staging) | **96.1** |
| **NVIDIA GB10** | Grace Blackwell | NVLink-C2C unified | **39.9** |
| **Apple M3 Pro** | Apple Silicon | Unified | **29.9** |
| AMD Radeon (mobile) | Integrated | Unified | 16.7 |

## Dispatch Latency

Single workgroup × 256 threads, lower is better.

| Platform | Latency |
|---|---|
| **AMD Radeon (mobile)** | **45 µs** |
| **NVIDIA GB10** | **48 µs** |
| **Apple M3 Pro** | **136 µs** |
| NVIDIA RTX 5080 | 29 µs (best) / 84 µs (worst run) |

## Pipeline Compilation

WGSL → native shader compilation, lower is better.

| Platform | Time |
|---|---|
| NVIDIA GB10 | 0.028 ms |
| NVIDIA RTX 5080 | 0.032 ms |
| Apple M3 Pro | 0.07 ms |
| AMD Radeon (mobile) | 0.33 ms |

## Buffer I/O

### Create (host → GPU, GB/s)

| Size | GB10 | RTX 5080 | AMD Radeon |
|---|---|---|---|
| 1K f32 | 0.04 | 0.01 | 0.19 |
| 16K f32 | 0.67 | 0.24 | 2.10 |
| 256K f32 | 5.91 | 1.02 | 7.31 |
| 1M f32 | 11.19 | 3.30 | 8.21 |

### Read (GPU → host, GB/s)

| Size | GB10 | RTX 5080 | AMD Radeon |
|---|---|---|---|
| 1K f32 | instant | 0.15 | instant |
| 16K f32 | instant | 2.01 | instant |
| 256K f32 | instant | 7.67 | instant |
| 1M f32 | instant | 5.36 | instant |

*"instant" = < 0.1 µs — unified memory platforms read via pointer dereference (zero copy).*

## Batched Dispatch (`dispatch_many`)

256 dispatches per command buffer, SAXPY kernel. Higher GFLOPS is better.

| Size | GB10 (single) | GB10 (×256 batch) | RTX 5080 (single) | RTX 5080 (×256 batch) |
|---|---|---|---|---|
| 1K el | 0.07 | 6.1 GFLOPS (87×) | 0.12 | 6.9 GFLOPS (58×) |
| 16K el | 1.1 | 60.5 GFLOPS (55×) | 1.8 | 112 GFLOPS (62×) |
| 256K el | 10.9 | 141.7 GFLOPS (13×) | 27.7 | **408 GFLOPS** (15×) |
| 1M el | 27.2 | 208 GFLOPS (7.6×) | 101.7 | **577 GFLOPS** (5.7×) |
| 16M el | 49.2 | 61.6 GFLOPS (1.3×) | 92.2 | 75.3 GFLOPS (0.8×) |

Per-dispatch overhead at 256 batched:

| Platform | Single | Batched ×256 | Reduction |
|---|---|---|---|
| GB10 | 46 µs | **1.0 µs** | 46× |
| RTX 5080 | 37 µs | **0.5 µs** | 75× |

At small element counts, batching eliminates command-buffer alloc/free
dominance. At 16M elements, kernel compute dominates and batching provides
marginal benefit.

## Test Hardware

| Platform | CPU | GPU | RAM | OS |
|---|---|---|---|---|
| AMD laptop | AMD Ryzen | Integrated Radeon | 16 GB DDR5 | Ubuntu 25.10 |
| GB10 DGX | Grace ARM64 | Blackwell (NVLink-C2C) | 128 GB LPDDR5X | Ubuntu 24.04 |
| RTX 5080 | AMD Ryzen 9 | RTX 5080 16 GB | 64 GB DDR5 | Ubuntu 24.04 |
| M3 Pro | Apple M3 Pro | Integrated (18-core) | 36 GB unified | macOS 15 |

*Benchmarks run 2026-06-03. Results are single-run warm averages; production workloads may vary.*
