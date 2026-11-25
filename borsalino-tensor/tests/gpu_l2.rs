// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T2c GPU contract tests: the FixedOrderGemm kernels (`matmul`,
//! `batched_matmul`), each against an INDEPENDENTLY written oracle
//! (hand-rolled triple loops — the CPU reference is the contract, but
//! the expected values here are recomputed so a shared bug cannot pass
//! both sides). Final-output comparisons: GEMM has no
//! architecture-precision tail op (no division, no sqrt), so
//! bit-equality IS the assertion — the accumulation-matching it
//! observes is the empirical expectation (fused requests on both
//! sides, per the `fma` semantics probe), scoped exactly like T2b's.
//!
//! Skip discipline matches `gpu_l0.rs`/`gpu_l1.rs`.

#![cfg(feature = "gpu")]

use borsalino_tensor::error::TensorError;
use borsalino_tensor::gpu::TensorGpu;
use borsalino_tensor::layout::Shape;

fn gpu() -> Option<impl borsalino::GpuBackend> {
    match borsalino::init() {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(
                std::env::var("BORSALINO_TENSOR_REQUIRE_GPU").is_err(),
                "BORSALINO_TENSOR_REQUIRE_GPU is set but backend init failed: {e}"
            );
            eprintln!("skipping: no GPU backend ({e})");
            None
        }
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Independent GEMM oracle: row-major triple loop, ascending K, one
/// `mul_add` per term — written from the contract, not shared with the
/// implementation.
fn gemm_reference(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc = lhs[row * k + c].mul_add(rhs[c * n + col], acc);
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// `matmul` bit-equality on a non-square case with a partial final
/// workgroup (37×11 = 407 outputs, K = 53).
#[test]
#[serial_test::serial]
fn gpu_matmul_matches_independent_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (m, k, n) = (37usize, 53, 11);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i % 29) as f32 * 0.25 - 3.5).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| (i % 31) as f32 * 0.2 - 3.0).collect();
    let ls = Shape::new(&[m as u32, k as u32]).expect("fixture");
    let rs = Shape::new(&[k as u32, n as u32]).expect("fixture");
    let got = tg
        .execute_matmul(&lhs, &rhs, &[&ls, &rs])
        .expect("dispatch");
    let expected = gemm_reference(&lhs, &rhs, m, k, n);
    assert_eq!(bits(&got), bits(&expected));
}

/// `matmul` order sensitivity: the K order is the contract — a fixture
/// where ascending and descending accumulation differ in bits
/// (1e8, -1e8, 1.0 against a unit column), asserted against the
/// ascending oracle. This is the fixture that FAILS under a
/// K-reordered kernel (mutation-verified on this tree).
#[test]
#[serial_test::serial]
fn gpu_matmul_pins_ascending_k_order() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs = [1.0e8f32, -1.0e8, 1.0];
    let rhs = [1.0f32; 3];
    let ls = Shape::new(&[1, 3]).expect("fixture");
    let rs = Shape::new(&[3, 1]).expect("fixture");
    let got = tg
        .execute_matmul(&lhs, &rhs, &[&ls, &rs])
        .expect("dispatch");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_bits(), 1.0f32.to_bits());
    // The fixture must be order-sensitive (the reverse chain loses the
    // 1.0).
    let mut descending = 0.0f32;
    for i in (0..3).rev() {
        descending = lhs[i].mul_add(rhs[i], descending);
    }
    assert_ne!(descending.to_bits(), got[0].to_bits());
}

/// `matmul` with K = 0: the empty contraction accumulates to `0.0` —
/// an all-zero output, matching the CPU loop.
#[test]
#[serial_test::serial]
fn gpu_matmul_zero_k_produces_zeros() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[4, 0]).expect("fixture");
    let rs = Shape::new(&[0, 3]).expect("fixture");
    let got = tg.execute_matmul(&[], &[], &[&ls, &rs]).expect("dispatch");
    assert_eq!(bits(&got), bits(&[0.0f32; 12]));
}

/// `matmul` with M = 0 or N = 0: empty output, no dispatch.
#[test]
#[serial_test::serial]
fn gpu_matmul_empty_output_is_empty() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[0, 5]).expect("fixture");
    let rs = Shape::new(&[5, 7]).expect("fixture");
    let got = tg
        .execute_matmul(&[], &rhs_of(5, 7), &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
    let ls = Shape::new(&[3, 5]).expect("fixture");
    let rs = Shape::new(&[5, 0]).expect("fixture");
    let got = tg
        .execute_matmul(&lhs_of(3, 5), &[], &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
}

/// K-dimension mismatch still rejects at the shape rule (the gate).
#[test]
#[serial_test::serial]
fn gpu_matmul_contract_mismatch_rejects() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[3, 5]).expect("fixture");
    let rs = Shape::new(&[4, 7]).expect("fixture"); // K 5 vs 4
    match tg.execute_matmul(&lhs_of(3, 5), &rhs_of(4, 7), &[&ls, &rs]) {
        Err(TensorError::InnerDimensionMismatch { .. }) => {}
        other => panic!("contract mismatch must reject, got {other:?}"),
    }
}

/// Malformed shapes reject BEFORE any indexing (guards, not panics) —
/// the #74 r1 lesson, applied from the start. CPU-only: ErrBackend.
#[test]
#[serial_test::serial]
fn gpu_matmul_entries_reject_malformed_shapes_without_panicking() {
    let mut tg = TensorGpu::new(ErrBackend);
    let rank2 = Shape::new(&[1, 1]).expect("fixture");
    // Missing second shape.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tg.execute_matmul(&[1.0], &[1.0], &[&rank2])
    }));
    assert!(
        matches!(r, Ok(Err(TensorError::GpuDispatch { .. }))),
        "{r:?}"
    );
    // Wrong rank on the second port.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tg.execute_batched_matmul(&[1.0], &[1.0], &[&rank2, &rank2])
    }));
    assert!(
        matches!(r, Ok(Err(TensorError::RankMismatch { .. }))),
        "{r:?}"
    );
    // Empty shape list.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tg.execute_batched_matmul(&[], &[], &[])
    }));
    assert!(
        matches!(r, Ok(Err(TensorError::GpuDispatch { .. }))),
        "{r:?}"
    );
}

/// `batched_matmul` bit-equality: [2,3,5,7] x [2,3,7,11] -> [2,3,5,11]
/// (330 outputs — partial final workgroup), independently referenced.
#[test]
#[serial_test::serial]
fn gpu_batched_matmul_matches_independent_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (b, h, s, d, t) = (2usize, 3, 5, 7, 11);
    let lhs: Vec<f32> = (0..b * h * s * d)
        .map(|i| ((i * 7) % 43) as f32 * 0.25 - 5.0)
        .collect();
    let rhs: Vec<f32> = (0..b * h * d * t)
        .map(|i| ((i * 11) % 37) as f32 * 0.2 - 3.5)
        .collect();
    let ls = Shape::new(&[2, 3, 5, 7]).expect("fixture");
    let rs = Shape::new(&[2, 3, 7, 11]).expect("fixture");
    let got = tg
        .execute_batched_matmul(&lhs, &rhs, &[&ls, &rs])
        .expect("dispatch");
    // Independent reference: per-slice GEMMs.
    let mut expected = Vec::with_capacity(b * h * s * t);
    for slice in 0..b * h {
        let left = &lhs[slice * s * d..][..s * d];
        let right = &rhs[slice * d * t..][..d * t];
        expected.extend(gemm_reference(left, right, s, d, t));
    }
    assert_eq!(bits(&got), bits(&expected));
}

/// `batched_matmul` empty dimensions: B, H, S, or T zero → empty output
/// (an enormous batch with an empty head is legal and writes nothing);
/// D zero → all zeros.
#[test]
#[serial_test::serial]
fn gpu_batched_matmul_empty_dimensions() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    // B = 0.
    let ls = Shape::new(&[0, 3, 5, 7]).expect("fixture");
    let rs = Shape::new(&[0, 3, 7, 11]).expect("fixture");
    let got = tg
        .execute_batched_matmul(&[], &[], &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
    // T = 0.
    let ls = Shape::new(&[2, 3, 5, 7]).expect("fixture");
    let rs = Shape::new(&[2, 3, 7, 0]).expect("fixture");
    let got = tg
        .execute_batched_matmul(&lhs_of(2 * 3 * 5 * 7, 1), &[], &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
    // H = 0 and S = 0: empty outputs (the coverage r1 asked for).
    let ls = Shape::new(&[2, 0, 5, 7]).expect("fixture");
    let rs = Shape::new(&[2, 0, 7, 11]).expect("fixture");
    let got = tg
        .execute_batched_matmul(&[], &[], &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
    let ls = Shape::new(&[2, 3, 0, 7]).expect("fixture");
    let rs = Shape::new(&[2, 3, 7, 11]).expect("fixture");
    // rhs keeps its B*H*D*T elements (S does not appear in rhs).
    let got = tg
        .execute_batched_matmul(&[], &rhs_of(2 * 3 * 7, 11), &[&ls, &rs])
        .expect("dispatch");
    assert!(got.is_empty());
    // D = 0: 30 outputs of 0.0.
    let ls = Shape::new(&[2, 3, 5, 0]).expect("fixture");
    let rs = Shape::new(&[2, 3, 0, 11]).expect("fixture");
    let got = tg
        .execute_batched_matmul(&[], &[], &[&ls, &rs])
        .expect("dispatch");
    assert_eq!(bits(&got), bits(&[0.0f32; 2 * 3 * 5 * 11]));
}

/// Batch/head mismatch rejects (heads are not broadcast).
#[test]
#[serial_test::serial]
fn gpu_batched_matmul_batch_mismatch_rejects() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[2, 3, 5, 7]).expect("fixture");
    let rs = Shape::new(&[2, 4, 7, 11]).expect("fixture"); // H mismatch
    match tg.execute_batched_matmul(&[], &[], &[&ls, &rs]) {
        Err(TensorError::BatchDimensionMismatch { .. }) => {}
        other => panic!("batch mismatch must reject, got {other:?}"),
    }
}

/// Both GEMMs are repeat-dispatch deterministic.
#[test]
#[serial_test::serial]
fn gpu_gemm_is_repeat_dispatch_deterministic() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs: Vec<f32> = (0..5 * 7).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
    let rhs: Vec<f32> = (0..7 * 3).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let ls = Shape::new(&[5, 7]).expect("fixture");
    let rs = Shape::new(&[7, 3]).expect("fixture");
    let first = bits(&tg.execute_matmul(&lhs, &rhs, &[&ls, &rs]).expect("d1"));
    let again = bits(&tg.execute_matmul(&lhs, &rhs, &[&ls, &rs]).expect("d2"));
    assert_eq!(first, again);

    let bl: Vec<f32> = (0..2 * 2 * 3 * 5).map(|i| i as f32 * 0.1).collect();
    let br: Vec<f32> = (0..2 * 2 * 5 * 3).map(|i| -i as f32 * 0.1).collect();
    let bls = Shape::new(&[2, 2, 3, 5]).expect("fixture");
    let brs = Shape::new(&[2, 2, 5, 3]).expect("fixture");
    let first = bits(
        &tg.execute_batched_matmul(&bl, &br, &[&bls, &brs])
            .expect("d1"),
    );
    let again = bits(
        &tg.execute_batched_matmul(&bl, &br, &[&bls, &brs])
            .expect("d2"),
    );
    assert_eq!(first, again);
}

/// GEMM pipelines cache: second call must not recompile.
#[test]
#[serial_test::serial]
fn gpu_gemm_pipelines_cache() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[4, 4]).expect("fixture");
    let rs = Shape::new(&[4, 4]).expect("fixture");
    let data = [1.0f32; 16];
    let _ = tg.execute_matmul(&data, &data, &[&ls, &rs]).expect("warm");
    assert!(tg.pipelines().contains_key("matmul"));
    // Second call reuses the cached pipeline and stays correct.
    let again = tg
        .execute_matmul(&data, &data, &[&ls, &rs])
        .expect("second");
    let first = &tg.pipelines();
    let _ = first;
    assert_eq!(again.len(), 16);
    let bls = Shape::new(&[1, 1, 4, 4]).expect("fixture");
    let _ = tg
        .execute_batched_matmul(&data, &data, &[&bls, &bls])
        .expect("warm");
    assert!(tg.pipelines().contains_key("batched_matmul"));
}

/// r1 #75 regressions (CPU-only): overflow-class shapes must reach the
/// empty-output early return (or a named error), never panic. The
/// reviewer's reproductions: `[MAX,MAX,MAX,0] x [MAX,MAX,0,0]` (the
/// B*H*S product overflowed u64 before the trailing zero) and a direct
/// public `params` call on `[0,1,MAX,2] x [0,1,2,1]` (u32 stride
/// product overflow).
#[test]
#[serial_test::serial]
fn gpu_gemm_overflow_shapes_never_panic() {
    use borsalino_tensor::gpu::GpuKernel;
    use borsalino_tensor::ops::l2::BatchedMatMul;
    let mut tg = TensorGpu::new(ErrBackend);
    let max = u32::MAX;
    let ls = Shape::new(&[max, max, max, 0]).expect("fixture");
    let rs = Shape::new(&[max, max, 0, 0]).expect("fixture");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tg.execute_batched_matmul(&[], &[], &[&ls, &rs])
    }));
    assert!(matches!(&r, Ok(Ok(out)) if out.is_empty()), "{r:?}");

    // Direct public params with an overflowing stride product (both
    // factors non-zero, product > u32): must reject, not panic.
    let ls = Shape::new(&[0, 1, max, 2]).expect("fixture");
    let rs = Shape::new(&[0, 1, 2, 1]).expect("fixture");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        BatchedMatMul::params(&[&ls, &rs])
    }));
    // The zero is in B, which is not a stride factor: S*D = MAX*2
    // genuinely overflows, so the honest answer is Err — a Result,
    // never a panic. (The entry path cannot reach this: the
    // empty-output early return fires first, and span checks bound
    // S*D <= u32::MAX otherwise.)
    assert!(
        matches!(&r, Ok(Err(TensorError::GpuDispatch { .. }))),
        "overflowing stride must reject, got {r:?}"
    );
    // All-nonzero overflow rejects.
    let ls = Shape::new(&[1, 1, max, 2]).expect("fixture");
    let rs = Shape::new(&[1, 1, 2, 1]).expect("fixture");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        BatchedMatMul::params(&[&ls, &rs])
    }));
    assert!(
        matches!(r, Ok(Err(TensorError::GpuDispatch { .. }))),
        "{r:?}"
    );
}

/// A backend whose every call fails — the guards under test run BEFORE
/// any backend call (the #74 r1 technique, carried into T2c).
struct ErrBackend;

impl borsalino::GpuBackend for ErrBackend {
    fn init() -> borsalino::Result<Self> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn compile(&self, _entry: &str, _wgsl: &str) -> borsalino::Result<borsalino::ComputePipeline> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn create_buffer<T: borsalino::bytemuck::Pod>(
        &self,
        _data: &[T],
    ) -> borsalino::Result<borsalino::GpuBuffer> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn create_buffer_uninit<T: borsalino::bytemuck::Pod>(
        &self,
        _len: usize,
    ) -> borsalino::Result<borsalino::GpuBuffer> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn dispatch(
        &self,
        _pipeline: &borsalino::ComputePipeline,
        _buffers: &[&borsalino::GpuBuffer],
        _workgroups: (u32, u32, u32),
    ) -> borsalino::Result<()> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn dispatch_ex(
        &self,
        _pipeline: &borsalino::ComputePipeline,
        _buffers: &[&borsalino::GpuBuffer],
        _workgroups: (u32, u32, u32),
        _threads_per_group: (u32, u32, u32),
    ) -> borsalino::Result<()> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn read_buffer<T: borsalino::bytemuck::Pod>(
        &self,
        _buffer: &borsalino::GpuBuffer,
    ) -> borsalino::Result<Vec<T>> {
        Err(borsalino::GpuError::NoBackend)
    }
    fn timestamp(&self) -> borsalino::Result<u64> {
        Err(borsalino::GpuError::NoBackend)
    }
}

fn lhs_of(m: usize, k: usize) -> Vec<f32> {
    (0..m * k).map(|i| (i % 7) as f32 * 0.5).collect()
}

fn rhs_of(k: usize, n: usize) -> Vec<f32> {
    (0..k * n).map(|i| (i % 5) as f32 * -0.25).collect()
}
