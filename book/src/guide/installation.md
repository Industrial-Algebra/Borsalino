# Installation & Feature Flags

```toml
[dependencies]
borsalino = { version = "0.7", features = ["vulkan"] }
```

| Feature | What it enables |
|---|---|
| `metal` | Metal backend (macOS only) |
| `vulkan` | Vulkan backend via ash (Linux / Windows) |
| `verify` | karpal-verify obligation bundles + numerical correctness + determinism verification |

Features are additive: the crate builds with none (stub path), and each
backend layers on without touching the core types.

## CI notes

- GPU tests are serialized (`#[serial]`) — parallel GPU tests on real hardware
  contend for driver-internal locks.
- The dedicated hardware jobs set `BORSALINO_REQUIRE_METAL` /
  device-required env so a green job means kernels actually executed, not
  silently skipped.
