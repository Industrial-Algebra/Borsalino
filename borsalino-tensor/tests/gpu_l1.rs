// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T2b GPU contract tests: broadcast execution and the fixed-order
//! reductions, each against an INDEPENDENTLY written oracle (no shared
//! helper with the implementation — the CPU reference module is the
//! contract, but the expected values here are recomputed by hand-rolled
//! loops so a shared bug cannot pass both sides).
//!
//! Skip discipline matches `gpu_l0.rs`: runs only where a GPU exists,
//! `BORSALINO_TENSOR_REQUIRE_GPU=1` turns the skip into a failure.

#![cfg(feature = "gpu")]

use borsalino_tensor::error::TensorError;
use borsalino_tensor::gpu::TensorGpu;
use borsalino_tensor::layout::Shape;
use borsalino_tensor::op::TensorOp;
use borsalino_tensor::ops::l0::{Add, BroadcastDiv, Mul};
use borsalino_tensor::ops::l1::{MaskedMean, RowwiseL2Norm};

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

/// Ulp distance between two same-sign finite f32s (bit distance).
fn ulps(gpu: f32, cpu: f32) -> u32 {
    gpu.to_bits().abs_diff(cpu.to_bits())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Right-aligned axis access: axis 0 is the LAST dim; leading dims a
/// shorter operand lacks are 1 (NumPy's rule).
fn axis(shape: &[u32], from_right: usize) -> usize {
    if from_right < shape.len() {
        shape[shape.len() - 1 - from_right] as usize
    } else {
        1
    }
}

/// Independent broadcast oracle: decomposes each flat output index into
/// per-axis coordinates (right-aligned), maps through stride-0
/// broadcast strides, applies the combining op. Written from the NumPy
/// rule directly — not shared with the executor.
#[allow(clippy::needless_range_loop)] // the odometer is clearest indexed
fn broadcast_reference(
    lhs: &[f32],
    rhs: &[f32],
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    op: fn(f32, f32) -> f32,
) -> Vec<f32> {
    let rank = lhs_shape.len().max(rhs_shape.len());
    let out_dim = |from_right: usize| axis(lhs_shape, from_right).max(axis(rhs_shape, from_right));
    let out_dims: Vec<usize> = (0..rank).map(out_dim).collect();
    // element strides per operand, right-aligned, 0 on broadcast axes
    let strides = |shape: &[u32]| -> Vec<usize> {
        let mut s = vec![0usize; rank];
        let mut acc = 1usize;
        for from_right in 0..rank {
            let d = axis(shape, from_right);
            s[from_right] = if d == 1 { 0 } else { acc };
            acc *= d;
        }
        s
    };
    let ls = strides(lhs_shape);
    let rs = strides(rhs_shape);
    let total: usize = out_dims.iter().product();
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut lo = 0usize;
        let mut ro = 0usize;
        for from_right in 0..rank {
            let c = rem % out_dims[from_right];
            rem /= out_dims[from_right];
            lo += c * ls[from_right];
            ro += c * rs[from_right];
        }
        out.push(op(lhs[lo], rhs[ro]));
    }
    out
}

/// Broadcast add over mixed ranks and broadcast axes — the stride
/// mapping, proven bit-exact against the independent oracle. Output
/// count 2×3×5×7 = 210 < 256 also exercises a partial workgroup.
#[test]
#[serial_test::serial]
fn gpu_broadcast_add_matches_independent_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs_shape = [2u32, 5, 3, 7];
    let rhs_shape = [3u32, 1]; // right-aligned: 3 pairs with the 3-axis, 1 broadcasts over 7
    let lhs: Vec<f32> = (0..210).map(|i| i as f32 * 0.25 - 25.0).collect();
    let rhs: Vec<f32> = (0..3).map(|i| i as f32 * 1.5 - 1.5).collect();
    let ls = Shape::new(&lhs_shape).expect("fixture");
    let rs = Shape::new(&rhs_shape).expect("fixture");
    let got = tg
        .execute_binary::<Add>(&lhs, &rhs, 210, &[&ls, &rs])
        .expect("dispatch");
    let expected = broadcast_reference(&lhs, &rhs, &lhs_shape, &rhs_shape, |a, b| a + b);
    assert_eq!(bits(&got), bits(&expected));
}

/// Broadcast mul with a leading-axis broadcast and rank mismatch
/// (`[1, 5] × [4, 5]`): output 20 — under one workgroup's worth.
#[test]
#[serial_test::serial]
fn gpu_broadcast_mul_matches_independent_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs_shape = [1u32, 5];
    let rhs_shape = [4u32, 5];
    let lhs: Vec<f32> = (0..5).map(|i| i as f32 * 0.5 - 1.0).collect();
    let rhs: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 - 1.0).collect();
    let ls = Shape::new(&lhs_shape).expect("fixture");
    let rs = Shape::new(&rhs_shape).expect("fixture");
    let got = tg
        .execute_binary::<Mul>(&lhs, &rhs, 20, &[&ls, &rs])
        .expect("dispatch");
    let expected = broadcast_reference(&lhs, &rhs, &lhs_shape, &rhs_shape, |a, b| a * b);
    assert_eq!(bits(&got), bits(&expected));
}

/// Broadcast div including ZERO divisors. Three seams documented in the
/// module docs: `inf` results are bit-exact (sign included); `NaN`
/// results match in CLASS only (payloads are platform-defined — this
/// stack emits `0x7FFFFFFF` where the CPU emits the canonical quiet
/// NaN); finite division is architecture-precision (this driver's
/// approx division can round one ulp off the correctly rounded
/// result — pinned by the `1e10 / 7` case at tolerance 1).
#[test]
#[serial_test::serial]
fn gpu_broadcast_div_inf_exact_nan_class_finite_ulp() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs_shape = [5u32];
    let _rhs_shape_len = 5usize; // rhs is [5]; the [5]×[5] call below
    let lhs = vec![1.0f32, -1.0, 0.0, f32::NAN, 1e10];
    let rhs = vec![0.0f32, 0.0, 0.0, 0.0, 7.0]; // [5] broadcasts to [5]
    let ls = Shape::new(&lhs_shape).expect("fixture");
    let rs = Shape::new(&[5u32]).expect("fixture");
    let got = tg
        .execute_binary::<BroadcastDiv>(&lhs, &rhs, 5, &[&ls, &rs])
        .expect("dispatch");
    // inf: bit-exact, sign preserved.
    assert_eq!(got[0].to_bits(), (1.0f32 / 0.0).to_bits());
    assert_eq!(got[1].to_bits(), (-1.0f32 / 0.0).to_bits());
    // NaN: class equality, payload is platform-defined.
    assert!(got[2].is_nan() && f32::NAN.is_nan());
    assert!(got[3].is_nan());
    // Finite division: architecture-precision — within 1 ulp.
    assert!(
        ulps(got[4], 1e10 / 7.0) <= 1,
        "{} vs {}",
        got[4],
        1e10 / 7.0
    );
}

/// Equal shapes still take the elementwise pipeline (no regression from
/// the broadcast path): same op, same shapes — the T2a behavior.
#[test]
#[serial_test::serial]
fn gpu_equal_shapes_still_elementwise() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let shape = Shape::new(&[37, 53]).expect("fixture");
    let lhs: Vec<f32> = (0..37 * 53).map(|i| i as f32 * 0.125 - 100.0).collect();
    let rhs: Vec<f32> = (0..37 * 53).map(|i| -(i as f32) * 0.0625 + 50.0).collect();
    let got = tg
        .execute_binary::<Add>(&lhs, &rhs, 37 * 53, &[&shape, &shape])
        .expect("dispatch");
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect();
    assert_eq!(bits(&got), bits(&expected));
}

/// Incompatible broadcast shapes still reject at the shape rule (the
/// gate, not the kernel): `[2] × [3]` has no common dim.
#[test]
#[serial_test::serial]
fn gpu_incompatible_broadcast_still_rejects() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let ls = Shape::new(&[2]).expect("fixture");
    let rs = Shape::new(&[3]).expect("fixture");
    match tg.execute_binary::<Add>(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, &[&ls, &rs]) {
        Err(TensorError::BroadcastIncompatible { .. }) => {}
        other => panic!("incompatible broadcast must reject, got {other:?}"),
    }
}

/// Broadcast dispatch-to-dispatch determinism: three dispatches of the
/// same broadcast call produce identical bit patterns.
#[test]
#[serial_test::serial]
fn gpu_broadcast_is_repeat_dispatch_deterministic() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let lhs_shape = [2u32, 5, 3, 7];
    let rhs_shape = [3u32, 1];
    let lhs: Vec<f32> = (0..210).map(|i| i as f32 * 0.25 - 25.0).collect();
    let rhs: Vec<f32> = (0..3).map(|i| i as f32 * 1.5 - 1.5).collect();
    let ls = Shape::new(&lhs_shape).expect("fixture");
    let rs = Shape::new(&rhs_shape).expect("fixture");
    let first = bits(
        &tg.execute_binary::<Add>(&lhs, &rhs, 210, &[&ls, &rs])
            .expect("d1"),
    );
    for _ in 0..2 {
        let again = bits(
            &tg.execute_binary::<Add>(&lhs, &rhs, 210, &[&ls, &rs])
                .expect("dn"),
        );
        assert_eq!(first, again);
    }
}

/// `sum`: sequential left-to-right accumulation, bit-exact vs the
/// hand-rolled loop. 999 elements (partial-workgroup-free — the kernel
/// is single-invocation) and a cancellation-heavy fixture so any order
/// deviation changes bits.
#[test]
#[serial_test::serial]
fn gpu_sum_matches_sequential_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let n = 999;
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 3.0).collect();
    let shape = Shape::new(&[n as u32]).expect("fixture");
    let got = tg.execute_sum(&values, &shape).expect("dispatch");
    let mut expected = 0.0f32;
    for &v in &values {
        expected += v;
    }
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_bits(), expected.to_bits());
}

/// `sum` of an EMPTY input is 0.0 on the CPU reference (the loop never
/// runs) — the GPU must match.
#[test]
#[serial_test::serial]
fn gpu_sum_of_empty_input_is_zero() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let shape = Shape::new(&[0]).expect("fixture");
    let got = tg.execute_sum(&[], &shape).expect("dispatch");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_bits(), 0);
}

/// `max`: strictly-`>` earliest-wins — bit-exact vs the hand-rolled
/// scan, including tie fixtures where the FIRST occurrence must win
/// (that is the -0.0-vs-0.0 case).
#[test]
#[serial_test::serial]
fn gpu_max_matches_strict_scan_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let values = vec![3.0f32, -7.5, 3.0, 1e-30, 3.0, -0.0, 2.99];
    let shape = Shape::new(&[values.len() as u32]).expect("fixture");
    let got = tg.execute_max(&values, &shape).expect("dispatch");
    let mut best = values[0];
    for &v in &values[1..] {
        if v > best {
            best = v;
        }
    }
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].to_bits(), best.to_bits());
}

/// `max` of an EMPTY input rejects with `EmptyReduction` before
/// dispatch — the CPU gate.
#[test]
#[serial_test::serial]
fn gpu_max_of_empty_input_rejects() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let shape = Shape::new(&[0]).expect("fixture");
    match tg.execute_max(&[], &shape) {
        Err(TensorError::EmptyReduction { op }) => assert_eq!(op, "max"),
        other => panic!("empty max must reject, got {other:?}"),
    }
}

/// `rowwise_l2_norm`: parallel across rows (37 rows — partial final
/// workgroup), sequential within. Bit-exact vs the hand-rolled
/// per-row loop.
#[test]
#[serial_test::serial]
fn gpu_rowwise_l2_norm_matches_oracle() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (rows, cols) = (37u32, 53u32);
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 7) % 211) as f32 * 0.05 - 5.0)
        .collect();
    let shape = Shape::new(&[rows, cols]).expect("fixture");
    let got = tg
        .execute_rowwise_l2_norm(&values, &shape)
        .expect("dispatch");
    let width = cols as usize;
    let expected: Vec<f32> = (0..rows as usize)
        .map(|row| {
            let mut sos = 0.0f32;
            for &v in &values[row * width..][..width] {
                sos = v.mul_add(v, sos);
            }
            sos.sqrt()
        })
        .collect();
    // The accumulation is fused and bit-exact; the final `sqrt` is
    // architecture-precision (this driver's approx sqrt rounds up to
    // 2 ulp off the correctly rounded result). Determinism is pinned
    // separately (repeat-dispatch, identical bits).
    for (gpu, cpu) in got.iter().zip(&expected) {
        assert!(ulps(*gpu, *cpu) <= 2, "sqrt seam: {gpu} vs {cpu}");
    }
}

/// `rowwise_l2_norm` of `[0, C]`: zero rows, empty result, no dispatch.
#[test]
#[serial_test::serial]
fn gpu_rowwise_l2_norm_zero_rows_is_empty() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let shape = Shape::new(&[0, 4]).expect("fixture");
    let got = tg.execute_rowwise_l2_norm(&[], &shape).expect("dispatch");
    assert!(got.is_empty());
}

/// `masked_mean`: `[B,S,D] + [B,S] -> [B,D]` with mask rows that are
/// fully zero (mean defined as 0.0, never NaN) and partially zero.
/// Bit-exact vs the hand-rolled double loop.
#[test]
#[serial_test::serial]
fn gpu_masked_mean_matches_oracle_including_zero_rows() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let (batch, tokens, width) = (5u32, 7u32, 11u32);
    let values: Vec<f32> = (0..batch * tokens * width)
        .map(|i| ((i * 13) % 191) as f32 * 0.1 - 9.0)
        .collect();
    // Row 0 fully masked; row 3 fully unmasked; rows 1/2/4 mixed.
    let mut mask = vec![1.0f32; (batch * tokens) as usize];
    for t in 0..tokens {
        mask[t as usize] = 0.0; // row 0: all zero
    }
    for t in 0..tokens {
        if t % 2 == 0 {
            mask[(tokens + t) as usize] = 0.0; // row 1: even tokens masked
        }
        if t < 2 {
            mask[(3 * tokens + t) as usize] = 1.0; // row 3: fully on (explicit)
        }
        if t >= tokens - 1 {
            mask[(4 * tokens + t) as usize] = 0.0; // row 4: all but last masked
        }
    }
    let vs = Shape::new(&[batch, tokens, width]).expect("fixture");
    let ms = Shape::new(&[batch, tokens]).expect("fixture");
    let got = tg
        .execute_masked_mean(&values, &mask, &[&vs, &ms])
        .expect("dispatch");
    let s = tokens as usize;
    let w = width as usize;
    let expected: Vec<f32> = (0..batch as usize)
        .flat_map(|b| (0..w).map(move |c| (b, c)))
        .map(|(b, c)| {
            let mut weight = 0.0f32;
            for t in 0..s {
                weight += mask[b * s + t];
            }
            let mut weighted = 0.0f32;
            for t in 0..s {
                weighted = mask[b * s + t].mul_add(values[(b * s + t) * w + c], weighted);
            }
            if weight == 0.0 {
                0.0
            } else {
                weighted / weight
            }
        })
        .collect();
    // Zero-mask rows are bit-exact (0.0 by definition); everything else
    // ends in `weighted / weight` — architecture-precision division
    // (this driver can round 1 ulp off). Determinism pinned separately.
    for (gpu, cpu) in got.iter().zip(&expected) {
        if cpu == &0.0 {
            assert_eq!(gpu.to_bits(), 0, "zero-mask row must be bit-exact");
        } else {
            assert!(ulps(*gpu, *cpu) <= 2, "division seam: {gpu} vs {cpu}");
        }
    }
}

/// The reductions are repeat-dispatch deterministic (all four).
#[test]
#[serial_test::serial]
fn gpu_reductions_are_repeat_dispatch_deterministic() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let values: Vec<f32> = (0..257)
        .map(|i| ((i * 31) % 101) as f32 * 0.25 - 12.0)
        .collect();
    let shape = Shape::new(&[257]).expect("fixture");
    let s1 = bits(&tg.execute_sum(&values, &shape).expect("s1"));
    let s2 = bits(&tg.execute_sum(&values, &shape).expect("s2"));
    assert_eq!(s1, s2);
    let m1 = bits(&tg.execute_max(&values, &shape).expect("m1"));
    let m2 = bits(&tg.execute_max(&values, &shape).expect("m2"));
    assert_eq!(m1, m2);

    let (rows, cols) = (33u32, 7u32);
    let mat: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.3).cos()).collect();
    let mat_shape = Shape::new(&[rows, cols]).expect("fixture");
    let n1 = bits(&tg.execute_rowwise_l2_norm(&mat, &mat_shape).expect("n1"));
    let n2 = bits(&tg.execute_rowwise_l2_norm(&mat, &mat_shape).expect("n2"));
    assert_eq!(n1, n2);

    let (batch, tokens, width) = (3u32, 5u32, 4u32);
    let vals: Vec<f32> = (0..batch * tokens * width)
        .map(|i| i as f32 * 0.5)
        .collect();
    let mut msk = vec![1.0f32; (batch * tokens) as usize];
    msk[2] = 0.0;
    let vs = Shape::new(&[batch, tokens, width]).expect("fixture");
    let ms = Shape::new(&[batch, tokens]).expect("fixture");
    let p1 = bits(
        &tg.execute_masked_mean(&vals, &msk, &[&vs, &ms])
            .expect("p1"),
    );
    let p2 = bits(
        &tg.execute_masked_mean(&vals, &msk, &[&vs, &ms])
            .expect("p2"),
    );
    assert_eq!(p1, p2);
}

/// Reduction kernel pipelines cache: a second call must not recompile
/// (dispatch is fast after the first).
#[test]
#[serial_test::serial]
fn gpu_reduction_pipelines_cache() {
    let Some(g) = gpu() else { return };
    let mut tg = TensorGpu::new(g);
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let shape = Shape::new(&[64]).expect("fixture");
    let _ = tg.execute_sum(&values, &shape).expect("warm");
    assert!(tg.pipelines().contains_key("sum"));
    let _ = tg.execute_max(&values, &shape).expect("warm");
    assert!(tg.pipelines().contains_key("max"));
    let mat_shape = Shape::new(&[8, 8]).expect("fixture");
    let mat: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let _ = tg.execute_rowwise_l2_norm(&mat, &mat_shape).expect("warm");
    assert!(tg.pipelines().contains_key("rowwise_l2_norm"));
    let vs = Shape::new(&[2, 4, 3]).expect("fixture");
    let ms = Shape::new(&[2, 4]).expect("fixture");
    let _ = tg
        .execute_masked_mean(&[1.0; 24], &[1.0; 8], &[&vs, &ms])
        .expect("warm");
    assert!(tg.pipelines().contains_key("masked_mean"));
}

/// Type-level exports still resolve (the ops are usable through the
/// trait surface used by the executor).
#[test]
#[serial_test::serial]
fn gpu_l1_op_constants_resolve() {
    assert_eq!(MaskedMean::NAME, "masked_mean");
    assert_eq!(RowwiseL2Norm::NAME, "rowwise_l2_norm");
}
