# Numerical Verification

`examples/numerical_verification.rs` — the exact-match protocol end to end:

```sh
cargo run --features vulkan,verify --example numerical_verification
```

The example defines a kernel's `NumericalReference` — inputs (f32 per
binding, in binding order), `output_len()`, `workgroups()` — then hands it to
the driver, which allocates the output buffer, chains inputs, dispatches on
real hardware, and compares the read-back output against the reference.
Exit-code hard: green means the kernel executed *and* matched.

Run it on hardware with the `--ignored` selection to pick up the recorded
end-to-end hardware trials.
