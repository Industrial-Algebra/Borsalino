# Introduction

**Borsalino** is a thin GPU compute abstraction for the Industrial Algebra
ecosystem.

> One trait, two backends, zero ceremony.

Write WGSL compute kernels. Dispatch them synchronously on Metal or Vulkan.
Read results back. No bind groups, no pipeline layouts, no descriptor sets,
no async runtime.

## Key Features

- **WGSL-first** — kernels are authored once in WGSL; naga translates to MSL
  (Metal) or SPIR-V (Vulkan).
- **Raw FFI backends** — `objc_msgSend` on macOS (19 selectors, zero Metal
  crate deps), `ash` on Linux/Windows. No `wgpu`, by design.
- **GC-safe dispatch tracking** — an atomic epoch tracker counts in-flight
  dispatches; `prove_quiescent()` certifies a moving GC may compact.
- **Verification built in** — determinism checking, a numerical-reference
  protocol, and Kani harnesses for bounded model checking.
- **Batched dispatch** — 75× per-dispatch latency reduction on RTX 5080 via
  `dispatch_many`; async dispatch via `dispatch_async` + `Pulse`.

## Relation to the ecosystem

Borsalino is the **compute** face of the IA GPU stack. The planned direction
(roadmap): the device/buffer/epoch substrate moves into
[Zunesha](https://github.com/Industrial-Algebra/Zunesha), with Borsalino as
its compute consumer and Goldenweek (graphics) beside it on the same device —
enabling zero-copy compute→render interop. **This release still owns its
device and buffers internally; the Zunesha migration is upcoming work**, not
shipped in v0.7.
