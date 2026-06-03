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

## Observations

- **Unified memory wins on buffer I/O** — Apple Silicon, AMD APU, and GB10 have zero-copy reads. RTX 5080 requires a PCIe staging transfer for each read.
- **Discrete GPUs win on compute throughput** — VRAM-local buffers avoid PCIe during dispatch. RTX 5080 delivers 3× the SAXPY throughput of GB10.
- **Dispatch overhead is consistently low** — 29-136 µs across all platforms. The synchronous command-buffer model adds negligible latency.
- **Pipeline compilation is dominated by naga** — shader translation time overwhelms the Vulkan/Metal driver compilation cost.

## Test Hardware

| Platform | CPU | GPU | RAM | OS |
|---|---|---|---|---|
| AMD laptop | AMD Ryzen | Integrated Radeon | 16 GB DDR5 | Ubuntu 25.10 |
| GB10 DGX | Grace ARM64 | Blackwell (NVLink-C2C) | 128 GB LPDDR5X | Ubuntu 24.04 |
| RTX 5080 | AMD Ryzen 9 | RTX 5080 16 GB | 64 GB DDR5 | Ubuntu 24.04 |
| M3 Pro | Apple M3 Pro | Integrated (18-core) | 36 GB unified | macOS 15 |

*Benchmarks run 2026-06-03. Results are single-run warm averages; production workloads may vary.*
