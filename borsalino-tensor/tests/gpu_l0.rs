// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T2a GPU gates — every kernel runs against the CPU oracle with
//! BIT-IDENTICAL comparison (all T2a ops are
//! [`DeterminismClass::BitExact`]), plus a repeat-dispatch determinism
//! sweep through Borsalino's own driver.
//!
//! Hardware discipline: the backend initialises per platform (vulkan off
//! Linux, metal off macOS — the dev-dependency carries both features);
//! tests skip politely when no device appears, and fail hard under
//! `BORSALINO_TENSOR_REQUIRE_GPU=1` (the CI GPU jobs set it).

#![cfg(feature = "gpu")]

mod common;

use borsalino::GpuBackend;
use borsalino_tensor::error::TensorError;
use borsalino_tensor::gpu::{GpuKernel, TensorGpu};
use borsalino_tensor::layout::Shape;
use borsalino_tensor::ops::l0::{Add, Mul};
use borsalino_tensor::ops::l2::Transpose2D;
use borsalino_tensor::ops::l3::Relu;

/// Platform-resolved backend via Borsalino's own `init` (metal on macOS,
/// vulkan elsewhere — the concrete type is opaque here, which is the
/// point: the executor is backend-generic).
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

/// Adversarial f32 classes for BitExact ops, NORMAL RANGE ONLY: ±0.0,
/// ±MIN_POSITIVE (the normal/subnormal boundary), the finite extremes,
/// mixed magnitudes. Subnormals are excluded from the oracle-equality
/// fixtures — they hit the documented FTZ/DAZ flush (pinned separately
/// below).
fn adversarial_normal(n: usize) -> Vec<f32> {
    let classes = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1e-20,
        -3.75,
        0.5,
    ];
    let mut out = Vec::with_capacity(n);
    let mut i = 0u32;
    while out.len() < n {
        let v = classes[(i as usize) % classes.len()];
        out.push(v + (i / 64) as f32 * 1e-3); // drift within normal range
        i += 1;
    }
    out
}

#[test]
#[serial_test::serial]
fn gpu_relu_matches_cpu_oracle_bit_exactly() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 1027; // non-power-of-2: partial last workgroup
    let input = adversarial_normal(n);
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let got = tg
        .execute_unary::<Relu>(&input, n, &[&shape])
        .expect("dispatch");
    // CPU oracle, contract form.
    let expected: Vec<f32> = input
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.0 })
        .collect();
    assert_eq!(bits(&got), bits(&expected));
}

#[test]
#[serial_test::serial]
fn gpu_add_matches_cpu_oracle_bit_exactly() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 1027;
    let lhs = adversarial_normal(n);
    // rhs is GENUINELY class-shifted by one position (classes[(i+1)%11]
    // at index i — NOT a slice of a longer vector, which yields the same
    // values at the same indices and pairs MAX with MAX → inf).
    let rhs: Vec<f32> = (0..n)
        .map(|i| {
            let classes = [
                1.0f32,
                -1.0,
                0.0,
                -0.0,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::MAX,
                f32::MIN,
                1e-20,
                -3.75,
                0.5,
            ];
            classes[(i + 1) % 11] + (i / 64) as f32 * 1e-3
        })
        .collect();
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let got = tg
        .execute_binary::<Add>(&lhs, &rhs, n, &[&shape, &shape])
        .expect("dispatch");
    // The fixture proves itself: every expected sum is finite and
    // normal-or-zero (overflow and subnormal sums are outside this gate
    // — the flush seam has its own test).
    let expected: Vec<f32> = lhs
        .iter()
        .zip(&rhs)
        .map(|(a, b)| {
            let s = a + b;
            assert!(
                s.is_finite() && (s.is_normal() || s == 0.0),
                "fixture produced {s} from {a} + {b}"
            );
            s
        })
        .collect();
    assert_eq!(bits(&got), bits(&expected));
}

#[test]
#[serial_test::serial]
fn gpu_mul_matches_cpu_oracle_bit_exactly() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 999;
    let lhs = adversarial_normal(n);
    // rhs is exactly ±1.0: identity / sign-flip. Products are then
    // provably normal-or-zero for the normal lhs classes — no overflow,
    // no subnormal results (that seam is pinned by the flush-rule test).
    let rhs: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let got = tg
        .execute_binary::<Mul>(&lhs, &rhs, n, &[&shape, &shape])
        .expect("dispatch");
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a * b).collect();
    assert_eq!(bits(&got), bits(&expected));
}

#[test]
#[serial_test::serial]
fn gpu_transpose_matches_cpu_oracle_bit_exactly() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (rows, cols) = (37u32, 53u32); // non-square, non-power-of-2
    let input = adversarial_normal((rows * cols) as usize);
    let in_shape = Shape::new(&[rows, cols]).expect("fixture");
    let got = tg
        .execute_unary::<Transpose2D>(&input, (rows * cols) as usize, &[&in_shape])
        .expect("dispatch");
    // The CPU reference's exact permutation (widened arithmetic).
    let mut expected = vec![0.0f32; (rows * cols) as usize];
    for r in 0..rows {
        for c in 0..cols {
            expected[(c * rows + r) as usize] = input[(r * cols + c) as usize];
        }
    }
    assert_eq!(bits(&got), bits(&expected));
}

/// The FTZ/DAZ flush rule, pinned as CONTRACT (see the gpu module docs):
/// subnormal inputs and results flush to a SIGN-PRESERVING zero, and
/// comparisons read subnormals as zero. These are not oracle-equality
/// failures — they are the documented seam, asserted exactly.
#[test]
#[serial_test::serial]
fn gpu_subnormal_flush_rule_is_sign_preserving_and_daz() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let sub = f32::from_bits(1); // smallest positive subnormal
    let neg_sub = -sub;
    let shape = Shape::new(&[3]).expect("fixture");

    // relu: DAZ — the comparison reads subnormals as zero, so a
    // POSITIVE subnormal yields +0.0 (not the value, not -0.0).
    let got = tg
        .execute_unary::<Relu>(&[sub, neg_sub, 1.0], 3, &[&shape])
        .expect("dispatch");
    assert_eq!(got[0].to_bits(), 0, "positive subnormal: DAZ → +0.0");
    assert_eq!(got[1].to_bits(), 0, "negative subnormal → +0.0");
    assert_eq!(got[2].to_bits(), 1.0f32.to_bits());

    // add: FTZ — subnormal + subnormal flushes to sign-preserving zero.
    let got = tg
        .execute_binary::<Add>(
            &[sub, neg_sub, sub],
            &[sub, neg_sub, sub],
            3,
            &[&shape, &shape],
        )
        .expect("dispatch");
    assert_eq!(got[0].to_bits(), 0, "sub + sub → +0.0");
    assert_eq!(
        got[1].to_bits(),
        (-0.0f32).to_bits(),
        "(-sub) + (-sub) → -0.0 (sign preserved)"
    );

    // add, RESULT class (FTZ): two NORMAL operands whose exact sum is
    // subnormal — 1.5×min_normal + (-min_normal) = 0.5×min_normal. The
    // inputs survive; only the result flushes.
    let min_n = f32::MIN_POSITIVE; // smallest normal
    let a = 1.5f32 * min_n; // normal
    let b = -min_n; // normal
    let exact = a + b; // subnormal on exact hardware
    assert!(exact.is_subnormal(), "fixture: exact sum must be subnormal");
    let got = tg
        .execute_binary::<Add>(&[a, -a, sub], &[b, -b, sub], 3, &[&shape, &shape])
        .expect("dispatch");
    assert_eq!(
        got[0].to_bits(),
        0,
        "subnormal SUM of normals → +0.0 (result FTZ)"
    );
    assert_eq!(
        got[1].to_bits(),
        (-0.0f32).to_bits(),
        "negative subnormal sum → -0.0 (sign preserved through the flush)"
    );

    // mul, RESULT class (FTZ): normal × normal with a subnormal product.
    let two_shape = Shape::new(&[2]).expect("fixture");
    let tiny = 1e-20f32; // normal
    let got = tg
        .execute_binary::<Mul>(&[tiny, tiny], &[1.0, 1.0], 2, &[&two_shape, &two_shape])
        .expect("dispatch");
    // tiny×1.0 = tiny (normal): passes through.
    let got2 = tg
        .execute_binary::<Mul>(&[tiny, tiny], &[tiny, tiny], 2, &[&two_shape, &two_shape])
        .expect("dispatch");
    assert_eq!(got[0].to_bits(), tiny.to_bits());
    assert_eq!(got2[0].to_bits(), 0, "subnormal product → +0.0");
}

/// Gather end-to-end: oracle equality vs an independently written
/// row-copy reference, plus the contract rejection (out-of-range index
/// → IndexOutOfRange BEFORE dispatch — mirroring the CPU gate, not the
/// kernel's defensive clamp).
#[test]
#[serial_test::serial]
fn gpu_gather_matches_cpu_oracle_and_rejects_bad_indices() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (vocab, dim) = (37u32, 11u32);
    let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.25 - 40.0).collect();
    let indices: Vec<u32> = vec![0, 36, 5, 0, 19, 36, 36, 5];
    let table_shape = Shape::new(&[vocab, dim]).expect("fixture");
    let idx_shape = Shape::new(&[indices.len() as u32]).expect("fixture");
    let out_len = indices.len() * dim as usize;

    let got = tg
        .execute_gather(&table, &indices, out_len, &[&table_shape, &idx_shape])
        .expect("dispatch");
    // Independent reference: out[n*d + c] = table[idx[n]*d + c].
    let expected: Vec<f32> = indices
        .iter()
        .flat_map(|&idx| {
            table[(idx * dim) as usize..((idx + 1) * dim) as usize]
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(bits(&got), bits(&expected));

    // Out-of-range rejects BEFORE dispatch (the CPU contract, host-side).
    let mut bad = vec![0u32; indices.len()];
    bad[indices.len() - 1] = vocab;
    match tg.execute_gather(
        &table,
        &bad,
        indices.len() * dim as usize,
        &[&table_shape, &idx_shape],
    ) {
        Err(TensorError::IndexOutOfRange {
            op, index, rows, ..
        }) => {
            assert_eq!(op, "gather");
            assert_eq!(index, vocab);
            assert_eq!(rows, vocab);
        }
        other => panic!("out-of-range index must reject, got {other:?}"),
    }
}

/// The executor's contract gate (review r1 P1): malformed calls reject
/// BEFORE any buffer exists — a short operand against a declared shape,
/// and a wrong output length.
#[test]
#[serial_test::serial]
fn gpu_executor_gates_malformed_calls() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let shape = Shape::new(&[8]).expect("fixture");
    let input = vec![1.0f32; 8];

    // Operand shorter than its declared shape.
    match tg.execute_unary::<Relu>(&input[..5], 5, &[&shape]) {
        Err(TensorError::ByteLenMismatch { .. }) => {}
        other => panic!("short operand must reject, got {other:?}"),
    }
    // Output length disagreeing with the inferred shape.
    match tg.execute_unary::<Relu>(&input, 7, &[&shape]) {
        Err(TensorError::ByteLenMismatch { .. }) => {}
        other => panic!("wrong output length must reject, got {other:?}"),
    }
}

#[test]
#[serial_test::serial]
fn gpu_kernels_are_repeat_dispatch_deterministic() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 1027;
    let input = adversarial_normal(n);
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let first = tg
        .execute_unary::<Relu>(&input, n, &[&shape])
        .expect("dispatch");
    for _ in 0..8 {
        let again = tg
            .execute_unary::<Relu>(&input, n, &[&shape])
            .expect("dispatch");
        assert_eq!(bits(&again), bits(&first), "repeat dispatch diverged");
    }
}

/// The pipeline cache actually caches: two executions compile once.
/// (Not a performance assertion — a correctness one: the cached pipeline
/// must be the SAME handle semantics, observable via repeated results.)
#[test]
#[serial_test::serial]
fn gpu_pipeline_cache_reuses_compiled_kernels() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 64;
    let input = adversarial_normal(n);
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let a = tg.execute_unary::<Relu>(&input, n, &[&shape]).expect("d1");
    let b = tg.execute_unary::<Relu>(&input, n, &[&shape]).expect("d2");
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(tg.pipelines().len(), 1, "one op, one cached pipeline");
}
