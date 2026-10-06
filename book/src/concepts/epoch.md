# Epoch Tracking & Pulses

Every dispatch increments an atomic counter; completion decrements it. The
tracker is what makes Borsalino safe to embed in a moving-GC host:

```rust
if gpu.is_quiescent() {
    // No dispatch touches host memory: compact away.
}
```

`prove_quiescent()` returns a `QuiescenceProof` — a certificate that all
dispatches were *observed complete* at a past instant. It is not a promise
about the future: between taking the proof and acting on it, another thread
may begin a dispatch (TOCTOU window). Closing the window is the embedder's
discipline (single dispatcher, or a shared lock around dispatch-and-compact).

## Pulses

`dispatch_async` returns a `Pulse` — a join handle for a GPU dispatch:

```rust
let pulse = gpu.dispatch_async(&pipeline, &[&input, &output], (1, 1, 1))?;
// ... overlap CPU work ...
pulse.wait();   // blocks until the dispatch completes
```

A pulse may outlive the backend that created it (shared tracker + shared
device ownership), and dropping it implicitly waits — no dispatch is ever
abandoned mid-flight.
