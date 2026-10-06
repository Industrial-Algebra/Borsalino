# Determinism Check

`examples/determinism_check.rs` — run a kernel N times, compare outputs:

```sh
cargo run --features vulkan,verify --example determinism_check
```

Reports whether repeated runs agree byte-for-byte, quantifies disagreement
when they don't, and exits non-zero on failure. The empirical complement to
the numerical protocol: determinism can only be *observed*, not proven, so
the harness makes observation cheap and loud.
