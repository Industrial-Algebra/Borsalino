# hello_compute

The canonical smoke test — the Quick Start kernel as a runnable example:

```sh
cargo run --features metal --example hello_compute    # macOS
cargo run --features vulkan --example hello_compute   # Linux / Windows
```

Creates a backend, compiles `add_one`, dispatches over four elements, reads
back `[2.0, 3.0, 4.0, 5.0]`, and exits non-zero on mismatch. If this passes
on a machine, the whole stack (naga translation, backend FFI, memory
strategy, readback) works there.
