# Verification

Borsalino's verification stack (behind the `verify` feature):

- **Determinism** — `determinism.rs` runs a kernel repeatedly and compares
  outputs empirically; disagreement is quantified, not just detected.
- **Numerical reference protocol** — `numerical_check.rs`: a kernel's
  `NumericalReference` describes its inputs (f32-native, per binding, in
  order), output length, and workgroup shape. The driver allocates the output
  buffer, chains the inputs, dispatches, and reads **the output buffer** —
  and the harness (a fake backend) asserts the reference's bindings and
  workgroups are exactly what the kernel received.
- **Kani harnesses** — bounded model checking of the structural contracts.
- **Hardware gates** — GPU tests run on real hardware (RTX 5080 / Apple
  Silicon runners) with mutation-tested verdicts: a sign-flipped kernel must
  FAIL verification, proving the tests can see.

## The five-silences lesson

v0.7's numerical verification was rebuilt after an audit found the old driver
never allocated the output buffer (drivers read the last input as "output"),
uploaded binaries as raw bytes to `array<f32>` kernels, never supplied the
sign table, and ran zero `--ignored` tests while `|| true` swallowed exit
codes. The rebuild's principle: **a green light must mean kernels executed
and were checked** — reference shape explicit, buffers explicit, gates
exit-code-hard.
