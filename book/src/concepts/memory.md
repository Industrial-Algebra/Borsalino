# Memory Strategy

Borsalino auto-detects the hardware and picks the optimal layout:

| GPU type | Detection | Memory | Behaviour |
|---|---|---|---|
| Apple Silicon M-series | Unified | Host-visible, coherent | Zero-copy between CPU and GPU |
| AMD integrated (APU) | Unified | Host-visible, coherent | Zero-copy |
| NVIDIA Grace Blackwell (GB10) | Unified | Host-visible, coherent | Zero-copy |
| NVIDIA RTX / AMD RDNA / Intel Arc | Discrete (auto) | Device-local VRAM + staging | Automatic PCIe transfers |

Explicit control:

```rust
use borsalino::MemoryStrategy;

let gpu = borsalino::init()?;                                          // auto
let gpu = borsalino::init_device_local()?;                             // VRAM
let gpu = VulkanBackend::init_with_strategy(MemoryStrategy::Unified)?; // unified
```

Persistent buffers keep data on the GPU across dispatches without CPU
readback — the working set for iterative workloads (ML training, physics
simulation).
