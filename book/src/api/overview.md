# API Overview

Full reference on [docs.rs/borsalino](https://docs.rs/borsalino). The
load-bearing surface:

| Type / method | Role |
|---|---|
| [`GpuBackend`] | The one trait: compile, create_buffer, dispatch, read. |
| [`ComputePipeline`] | Opaque compiled-kernel handle. |
| [`GpuBuffer`] | Opaque buffer handle; `read_buffer` returns typed data. |
| `dispatch` / `dispatch_many` / `dispatch_async` | Sync, batched, and async dispatch. |
| [`DispatchSpec`] | One entry in a `dispatch_many` batch. |
| [`Pulse`] | Async dispatch handle; `wait()` joins, Drop joins implicitly. |
| [`GpuEpochTracker`] / `QuiescenceProof` | GC-safety counting and certification. |
| [`NumericalReference`] | Kernel metadata for the numerical-check driver. |
| `prove_quiescent()` | Epoch-backed quiescence certificate (past instant). |
