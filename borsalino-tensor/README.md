# borsalino-tensor

Typed tensor op-set contract for the Borsalino GPU substrate.

T0 scope: the **contract only** — the L0–L2 operation inventory as typed
Rust (dtype policy, shape/stride/layout rules, determinism classes, error
taxonomy) plus the Katakuri layout-manifest interop. No kernels, no
compute, no CPU reference. Those are slices T1 (CPU reference) and T2
(WGSL kernels) of
`docs/plans/2026-10-16-borsalino-tensor-extension-handover.md`.

## Boundaries

- **f32 is the contract.** f16/bf16 appear only as explicit `Cast` types
  with documented roundtrip semantics; nothing narrows silently.
- **Not in scope** (handover §3): autodiff, training, optimizers, model
  specs (Katakuri's half), f64 exactness (Amari's half), graphics.
- **The manifest is data.** `zunesha::Buffer` binding is the consumer's
  concern; this crate never names a buffer type.

## Usage

```rust,ignore
use borsalino_tensor::op::TensorOp;
use borsalino_tensor::ops::l2::MatMul;

assert_eq!(<MatMul as TensorOp>::NAME, "matmul");
```

License: Apache-2.0.
