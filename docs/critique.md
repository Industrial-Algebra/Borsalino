# Critique of the **Borsalino** Project

> **v0.2.0 update (2026-06-03):** Since this critique was written (v0.1.0 pre-release),
> benchmarks have been published (4 platforms), documentation has been
> expanded (README, BENCHMARKS, CHANGELOG, ROADMAP), a dual commercial license
> model is in place, and CI covers format/clippy/test/docs/publish. The AGPL
> license is intentional by design, not a gap. See CHANGELOG.md for full history.

## TL;DR — Quick takeaways
- **Purpose** - A minimal, synchronous GPU compute abstraction that lets you write WGSL kernels and run them on either Metal (macOS) or Vulkan (Linux/Windows).  It aims for *zero-ceremony* - just a few function calls.
- **License** - **AGPL-3.0-only** (no commercial-license exception).  Shipping a binary that includes Borsalino forces the whole product to be AGPL-licensed.
- **Maturity** - 0.1.0, very early.  The crate compiles, has a modest API surface, and includes a `verify` feature that pulls in `karpal-verify`/`karpal-proof` for GPU-safety checks, but many parts are still experimental.
- **Target audience** - Researchers or hobbyists needing a *thin* cross-platform GPU compute layer and who are comfortable with the AGPL copyleft.

---

## Strengths (Why it could be useful)
| Area | Details |
|------|---------|
| **Very small surface** | Only four public types (`ComputePipeline`, `GpuBuffer`, `GpuError`, `Result`) and a single trait (`GpuBackend`).  Minimal boilerplate for dispatching compute kernels. |
| **Cross-platform backends** | Metal on macOS (via `objc`) and Vulkan on Linux/Windows (via `ash`).  The same WGSL source works on both, thanks to `naga` for translation. |
| **Synchronous, blocking API** | Simpler mental model - `dispatch` returns only when the GPU is finished.  Good for quick scripts or prototyping where async isn't needed. |
| **Safety wrapper** | The public API is safe; all `unsafe` is confined to the backend modules.  Buffer bounds are checked, shader compilation errors surface as `GpuError`. |
| **Verification feature** | Optional `verify` feature brings in `karpal-verify`/`karpal-proof`, offering formal GPU-safety checks for those who need stronger guarantees. |
| **No heavy abstractions** | No bind-group layout gymnastics, no descriptor-set management - you just pass buffers in order.  This matches the "zero-ceremony" promise. |
| **Rust-first design** | Uses `bytemuck` for POD data, `naga` for shader translation, and follows idiomatic error handling (`Result`). |

---

## Weaknesses / Red Flags (What limits its adoption)
| Issue | Impact |
|-------|--------|
| **AGPL-3.0-only license** | The copyleft nature blocks commercial use unless you release your entire product under the same license.  Most companies prefer permissive licenses (MIT/Apache). |
| **Very early stage (0.1.0)** | API is still stabilising; breaking changes are likely.  The crate has limited documentation and few examples. |
| **Synchronous-only** | No async or non-blocking dispatch.  For high-throughput workloads this can become a bottleneck. |
| **Thin feature set** | Only compute pipelines; no support for graphics, ray-tracing, or compute-shader pipelines with multiple dispatches. |
| **Backend complexity** | The Metal backend uses raw `objc` FFI; the Vulkan backend uses `ash`.  Users must have the appropriate SDKs installed (Xcode for Metal, Vulkan SDK for Linux/Windows). |
| **Sparse documentation** | The README gives a quick start, but module-level docs are minimal.  Users need to read source to understand lifetime rules, buffer alignment, and error handling. |
| **Limited testing** | The repository contains only a few unit tests.  No stress tests, no CI for all three platforms (macOS, Linux, Windows). |
| **No benchmark data** | No published performance numbers; it's unclear how the abstraction overhead compares to raw Metal/Vulkan usage. |
| **Verification optional** | The `verify` feature is powerful but optional; without it you lose the formal safety guarantees that the `karpal` ecosystem provides. |
| **No async or multi-GPU support** | The design assumes a single device and blocks until completion; scaling to multi-GPU or multi-threaded dispatches would require substantial changes. |

---

## Who would actually benefit?
- **Academic or hobbyist developers** experimenting with GPU compute kernels and wanting a uniform API across Metal and Vulkan.
- **Prototype engineers** who need a quick "write-once-run-anywhere" WGSL compute pipeline without dealing with bind-group boilerplate.
- **Rust-first teams** that already use `naga` and `bytemuck` and are comfortable with low-level GPU FFI.

*Not a good fit* for:
- Production-grade services that need a permissive license and guaranteed API stability.
- Applications requiring asynchronous pipelines, multi-GPU scaling, or advanced graphics features.
- Teams that lack the required platform SDKs (Xcode, Vulkan SDK) or cannot ship AGPL-licensed code.

---

## Recommendations for Improvement (If you control the project)
1. **Offer a dual-license** - Provide a commercial-friendly MIT/Apache option alongside AGPL to broaden adoption.
2. **Stabilise the API** - Move to a 1.0 release or at least a clear deprecation policy; publish a changelog.
3. **Expand documentation** - Add a "Getting Started" guide with a full end-to-end example (buffer creation → compile → dispatch → read).  Document safety guarantees, alignment requirements, and error codes.
4. **Add async support** - Provide a non-blocking `dispatch_async` variant that returns a future, enabling pipelines for high-throughput workloads.
5. **Publish benchmarks** - Show latency and throughput for typical workloads (e.g., vector addition) on Metal vs. Vulkan.
6. **Increase test coverage** - Add integration tests for both backends, stress tests for large buffers, and CI that runs on macOS, Linux, and Windows.
7. **Make verification default** - Consider enabling `karpal-verify` by default (or at least warn users when it's disabled) to promote safer GPU code.
8. **Provide a higher-level abstraction** - Optional helper for bind-group layout generation or automatic buffer alignment could attract users who want a bit more convenience without losing the "thin" philosophy.

---

*Prepared by the coding-agent on 2026-06-03.*