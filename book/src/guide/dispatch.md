# Batched & Async Dispatch

## dispatch_many

Chain multiple dispatches into a single command buffer:

```rust
use borsalino::{DispatchSpec, GpuBackend};

gpu.dispatch_many(&[
    DispatchSpec { pipeline: &p1, buffers: &[&buf_a, &buf_b],
                   workgroups: (4, 1, 1), threads_per_group: (256, 1, 1) },
    DispatchSpec { pipeline: &p2, buffers: &[&buf_b, &buf_c],
                   workgroups: (4, 1, 1), threads_per_group: (256, 1, 1) },
])?;
```

Batching amortises command-buffer allocation: on RTX 5080, 256 dispatches per
buffer drops per-dispatch latency from 37 µs to **0.5 µs** (75×); on GB10,
46 µs to **1.0 µs** (46×). Peak throughput **577 GFLOPS** (RTX 5080, 1M
elements batched).

## dispatch_async

Returns a [`Pulse`] immediately; the dispatch runs concurrently with CPU work:

```rust
let pulse = gpu.dispatch_async(&pipeline, &[&input, &output], (1, 1, 1))?;
pulse.wait();   // or let Drop join implicitly
```

Pulses are GC-tracked like every other dispatch, and may outlive the backend
(shared ownership — see [Epoch Tracking](./../concepts/epoch.md)).
