# borsalino-tensor

Typed tensor op-set contract for the Borsalino GPU substrate.

T0 scope: the **contract** — the L0–L2 operation inventory as typed Rust
(dtype policy, shape/stride/layout rules, determinism classes, error
taxonomy) plus the Katakuri layout-manifest interop.

T1 scope: the **CPU reference** for that inventory — every op executes,
bit-deterministically, on borrowed slices into a caller-provided output
buffer. It is first-class, not a fallback (handover §4.2): it is Ijima's
GPU-less execution path *and* the equivalence oracle T2's WGSL kernels are
measured against. See
`docs/plans/2026-10-16-borsalino-tensor-extension-handover.md`.

## Determinism and precision

Every reduction sums **strictly left-to-right** over the row-major element
order, single-threaded; GEMM's `k`-chain is ascending. No threads, no
atomics, no order-dependent fast paths — same inputs, same bits, any run,
any platform. Accumulation is **f32 throughout**: the oracle has to live in
the arithmetic domain of the kernels it judges (WGSL has no f64), and f64
exactness belongs to Amari.

NaN and ±inf **inputs** are rejected with a typed error before any element
is computed. Outputs are not policed: `f32 -> f16` overflows where its
`CastContract` says it does, and `broadcast_div` by zero keeps the
IEEE-754 result.

## Boundaries

- **f32 is the contract.** f16/bf16 appear only as explicit `Cast` types
  with documented roundtrip semantics; nothing narrows silently.
- **Not in scope** (handover §3): autodiff, training, optimizers, model
  specs (Katakuri's half), f64 exactness (Amari's half), graphics.
- **The manifest is data.** `zunesha::Buffer` binding is the consumer's
  concern; this crate never names a buffer type.

## Usage

```rust,ignore
use borsalino_tensor::exec::{Executable, View};
use borsalino_tensor::layout::{Shape, TensorLayout};
use borsalino_tensor::ops::l2::MatMul;
use borsalino_tensor::dtype::Dtype;

let lhs = TensorLayout::row_major(Dtype::F32, Shape::new(&[1, 2]).unwrap()).unwrap();
let rhs = TensorLayout::row_major(Dtype::F32, Shape::new(&[2, 1]).unwrap()).unwrap();
let mut out = [0.0f32; 1];
MatMul::execute(
    (View::new(&lhs, &[3.0, 4.0]), View::new(&rhs, &[5.0, 6.0])),
    &mut out,
)
.unwrap();
assert_eq!(out[0], 39.0);
```

License: Apache-2.0.
