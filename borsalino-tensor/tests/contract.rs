// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! T0 typed op-set contract tests (RED-first).
//!
//! The contract *is* the deliverable at T0: no kernel exists, so every
//! assertion here pins a type, a constant, or a rejected shape. The
//! inventory assertions mirror handover §3 one op at a time; the shape
//! assertions pin the rules a T1/T2 implementation must satisfy.

use borsalino_tensor::dtype::{
    BF16, CastContract, Dtype, DtypeMarker, F16, F32, Overflow, Roundtrip, SubnormalPolicy,
};
use borsalino_tensor::error::TensorError;
use borsalino_tensor::layout::{Shape, Strides, TensorLayout};
use borsalino_tensor::op::{DeterminismClass, OpLevel, PortRole, PortSpec, ShapeRule, TensorOp};
use borsalino_tensor::ops::l0::{Add, BroadcastDiv, Cast, Copy, Gather, Mul};
use borsalino_tensor::ops::l1::{MaskedMean, Max, RowwiseL2Norm, Sum};
use borsalino_tensor::ops::l2::{BatchedMatMul, MatMul};

/// 2^`exp` as an f64 — exact for every exponent these fixtures use.
fn pow2(exp: i32) -> f64 {
    f64::from_bits(((1023 + exp) as u64) << 52)
}

fn header<O: TensorOp>() -> (&'static str, OpLevel, usize) {
    (O::NAME, O::LEVEL, O::INPUTS.len())
}

#[test]
fn l0_inventory_matches_the_handover() {
    // Handover §3 L0: copy, cast, add, mul, broadcast_div, gather.
    let inventory = [
        header::<Copy>(),
        header::<Cast<F32, F16>>(),
        header::<Add>(),
        header::<Mul>(),
        header::<BroadcastDiv>(),
        header::<Gather>(),
    ];

    assert_eq!(
        inventory,
        [
            ("copy", OpLevel::L0, 1),
            ("cast", OpLevel::L0, 1),
            ("add", OpLevel::L0, 2),
            ("mul", OpLevel::L0, 2),
            ("broadcast_div", OpLevel::L0, 2),
            ("gather", OpLevel::L0, 2),
        ]
    );
}

#[test]
fn l1_inventory_matches_the_handover() {
    // Handover §3 L1: sum, attention-masked mean, L2 norm, max.
    let inventory = [
        header::<Sum>(),
        header::<MaskedMean>(),
        header::<RowwiseL2Norm>(),
        header::<Max>(),
    ];

    assert_eq!(
        inventory,
        [
            ("sum", OpLevel::L1, 1),
            ("masked_mean", OpLevel::L1, 2),
            ("rowwise_l2_norm", OpLevel::L1, 1),
            ("max", OpLevel::L1, 1),
        ]
    );
}

#[test]
fn l2_inventory_matches_the_handover() {
    // Handover §3 L2: matmul (2D), batched matmul [B,H,S,D].
    let inventory = [header::<MatMul>(), header::<BatchedMatMul>()];

    assert_eq!(
        inventory,
        [
            ("matmul", OpLevel::L2, 2),
            ("batched_matmul", OpLevel::L2, 2)
        ]
    );
}

#[test]
fn determinism_classes_are_per_level() {
    // Elementwise/cast work is bit-exact by construction.
    for class in [
        <Copy as TensorOp>::DETERMINISM,
        <Cast<F32, F16> as TensorOp>::DETERMINISM,
        <Add as TensorOp>::DETERMINISM,
        <Mul as TensorOp>::DETERMINISM,
        <BroadcastDiv as TensorOp>::DETERMINISM,
        <Gather as TensorOp>::DETERMINISM,
    ] {
        assert_eq!(class, DeterminismClass::BitExact);
    }

    // Reductions are determinism-sensitive (handover §3 L1).
    for class in [
        <Sum as TensorOp>::DETERMINISM,
        <MaskedMean as TensorOp>::DETERMINISM,
        <RowwiseL2Norm as TensorOp>::DETERMINISM,
        <Max as TensorOp>::DETERMINISM,
    ] {
        assert_eq!(class, DeterminismClass::FixedOrderReduction);
    }

    // GEMM carries its own fixed tiling order.
    for class in [
        <MatMul as TensorOp>::DETERMINISM,
        <BatchedMatMul as TensorOp>::DETERMINISM,
    ] {
        assert_eq!(class, DeterminismClass::FixedOrderGemm);
    }
}

#[test]
fn f32_is_the_only_operand_dtype() {
    // Every operand port of every non-cast op is f32 (handover §4.3).
    let port_sets: [&'static [PortSpec]; 11] = [
        <Copy as TensorOp>::INPUTS,
        <Add as TensorOp>::INPUTS,
        <Mul as TensorOp>::INPUTS,
        <BroadcastDiv as TensorOp>::INPUTS,
        <Gather as TensorOp>::INPUTS,
        <Sum as TensorOp>::INPUTS,
        <MaskedMean as TensorOp>::INPUTS,
        <RowwiseL2Norm as TensorOp>::INPUTS,
        <Max as TensorOp>::INPUTS,
        <MatMul as TensorOp>::INPUTS,
        <BatchedMatMul as TensorOp>::INPUTS,
    ];

    for ports in port_sets {
        for port in ports {
            if port.role == PortRole::Indices {
                assert_eq!(port.dtype, Dtype::U32);
                assert!(port.dtype.is_index());
            } else {
                assert_eq!(port.dtype, Dtype::F32);
            }
        }
    }

    // Outputs of every op are f32 except the narrowing cast.
    assert_eq!(<Cast<F32, F16> as TensorOp>::OUTPUT.dtype, Dtype::F16);
    assert_eq!(<Cast<F16, F32> as TensorOp>::OUTPUT.dtype, Dtype::F32);
    assert_eq!(<Cast<F32, BF16> as TensorOp>::OUTPUT.dtype, Dtype::BF16);
}

#[test]
fn gather_takes_u32_indices_on_its_second_port() {
    let ports = <Gather as TensorOp>::INPUTS;

    assert_eq!(ports[0], PortSpec::f32());
    assert_eq!(ports[1], PortSpec::indices());
    assert_eq!(ports[1].dtype, Dtype::U32);
    assert_eq!(ports[1].role, PortRole::Indices);
    assert_eq!(<Gather as TensorOp>::OUTPUT, PortSpec::f32());
}

#[test]
fn cast_is_the_only_explicit_dtype_conversion() {
    // Typed markers carry the dtype through the op type itself.
    assert_eq!(<Cast<F32, F16> as TensorOp>::INPUTS[0].dtype, Dtype::F32);
    assert_eq!(<Cast<F32, F16> as TensorOp>::OUTPUT.dtype, Dtype::F16);
    assert_eq!(<Cast<F16, F32> as TensorOp>::INPUTS[0].dtype, Dtype::F16);
    assert_eq!(<Cast<F16, F32> as TensorOp>::OUTPUT.dtype, Dtype::F32);
}

#[test]
fn narrowing_casts_document_roundtrip_semantics() {
    // f32 -> f16: round-to-nearest-even, overflow above the midpoint to
    // 2^16, and every f32 subnormal far below f16's minimum subnormal.
    let down = CastContract::for_dtypes(Dtype::F32, Dtype::F16).unwrap();
    assert_eq!(down.from, Dtype::F32);
    assert_eq!(down.to, Dtype::F16);
    assert_eq!(down.roundtrip, Roundtrip::HalfUlpInexact);
    assert_eq!(down.overflow, Overflow::ToInfinity);
    assert_eq!(down.subnormals, SubnormalPolicy::FlushesToZero);

    // f32 -> bf16: bf16 shares f32's exponent range but NOT f32's maximum
    // finite value, so it does overflow; f32 subnormals survive as bf16
    // subnormals down to bf16's floor.
    let down = CastContract::for_dtypes(Dtype::F32, Dtype::BF16).unwrap();
    assert_eq!(down.roundtrip, Roundtrip::HalfUlpInexact);
    assert_eq!(down.overflow, Overflow::ToInfinity);
    assert_eq!(down.subnormals, SubnormalPolicy::RoundsToTargetSubnormals);

    // Widening is exact in both directions.
    for (from, to) in [(Dtype::F16, Dtype::F32), (Dtype::BF16, Dtype::F32)] {
        let up = CastContract::for_dtypes(from, to).unwrap();
        assert_eq!(up.roundtrip, Roundtrip::Exact);
        assert_eq!(up.overflow, Overflow::CannotOverflow);
        assert_eq!(up.subnormals, SubnormalPolicy::Preserved);
    }

    // The typed op exposes the same contract.
    assert_eq!(
        Cast::<F32, F16>::contract().unwrap().roundtrip,
        Roundtrip::HalfUlpInexact
    );
    assert_eq!(
        Cast::<F16, F32>::contract().unwrap().roundtrip,
        Roundtrip::Exact
    );
}

#[test]
fn f16_narrowing_overflows_at_the_midpoint_not_at_the_maximum() {
    let c = CastContract::for_dtypes(Dtype::F32, Dtype::F16).unwrap();

    // Reviewer repro (PR #65 round 1): f16's largest finite value is 65504
    // and the next binade starts at 2^16, so rounding overflows to infinity
    // only at or above their midpoint, 65520. Values in between round DOWN.
    assert_eq!(c.max_finite, 65504.0);
    assert_eq!(c.overflow_threshold, 65520.0);
    assert_eq!(c.overflow_threshold, (c.max_finite + pow2(16)) / 2.0);

    // 65505 is above the maximum finite value but below the midpoint, so it
    // rounds to 65504 — it does not become infinity.
    let just_past_max = 65505.0;
    assert!(just_past_max > c.max_finite);
    assert!(just_past_max < c.overflow_threshold);

    // f32::MAX is far past the midpoint, so it does overflow.
    assert!(f64::from(f32::MAX) > c.overflow_threshold);
}

#[test]
fn f16_flushes_every_f32_subnormal_because_the_floor_is_far_above_them() {
    let c = CastContract::for_dtypes(Dtype::F32, Dtype::F16).unwrap();

    // f16's minimum subnormal is 2^-24; half of it (2^-25) is the flush
    // threshold. The largest f32 subnormal is around 2^-126.
    assert_eq!(c.min_subnormal, pow2(-24));
    let largest_f32_subnormal = (pow2(23) - 1.0) * pow2(-149);
    assert!(largest_f32_subnormal < c.min_subnormal / 2.0);
}

#[test]
fn bf16_narrowing_overflows_because_its_maximum_finite_is_smaller() {
    let c = CastContract::for_dtypes(Dtype::F32, Dtype::BF16).unwrap();

    // Reviewer repro (PR #65 round 1): bf16 keeps f32's 8-bit exponent but
    // not its 8-bit significand, so its largest finite value is
    // (2 - 2^-7) * 2^127 = 3.3895e38 < f32::MAX = 3.4028e38. Narrowing CAN
    // overflow, and f32::MAX is past the midpoint to 2^128.
    let max_finite = 255.0 * pow2(120);
    assert_eq!(c.max_finite, max_finite);
    assert!(c.max_finite < f64::from(f32::MAX));

    assert_eq!(c.overflow_threshold, (max_finite + pow2(128)) / 2.0);
    assert!(f64::from(f32::MAX) >= c.overflow_threshold);
}

#[test]
fn bf16_represents_f32_subnormals_down_to_its_own_floor() {
    let c = CastContract::for_dtypes(Dtype::F32, Dtype::BF16).unwrap();

    // bf16's minimum subnormal is 2^-133, sixteen binades above f32's
    // deepest, so f32 subnormals become bf16 subnormals rather than zero.
    assert_eq!(c.min_subnormal, pow2(-133));

    // Reviewer repro: f32 bits 0x0000_c000 is 49152 * 2^-149 = 0.75 * 2^-133.
    // Three quarters of bf16's minimum subnormal rounds to ONE ulp, not zero.
    let value = f64::from(f32::from_bits(0x0000_c000));
    assert_eq!(value, 0.75 * pow2(-133));
    assert!(value > c.min_subnormal / 2.0);

    // The genuine flush case sits below half of the minimum subnormal.
    assert!(pow2(-135) < c.min_subnormal / 2.0);
}

#[test]
fn widening_casts_are_exact_and_cannot_overflow() {
    for from in [Dtype::F16, Dtype::BF16] {
        let up = CastContract::for_dtypes(from, Dtype::F32).unwrap();
        assert_eq!(up.overflow, Overflow::CannotOverflow);
        assert_eq!(up.max_finite, f64::from(f32::MAX));
        assert_eq!(up.min_subnormal, pow2(-149));

        // The claim is numeric, not a convention: the largest finite source
        // value sits strictly below the target's overflow threshold.
        let source = from.float_bounds().unwrap();
        assert!(source.max_finite < up.overflow_threshold);
        // And every source subnormal is representable in f32.
        assert!(source.min_subnormal >= up.min_subnormal);
    }
}

#[test]
fn dtype_bounds_are_the_ieee_754_values() {
    let f32_bounds = Dtype::F32.float_bounds().unwrap();
    assert_eq!(f32_bounds.max_finite, f64::from(f32::MAX));
    assert_eq!(f32_bounds.min_subnormal, pow2(-149));
    // The midpoint between f32::MAX and 2^128: nothing representable in f32
    // reaches it, which is what makes an f16/bf16 widening exact.
    assert_eq!(
        f32_bounds.overflow_threshold,
        (f64::from(f32::MAX) + pow2(128)) / 2.0
    );

    let f16_bounds = Dtype::F16.float_bounds().unwrap();
    assert_eq!(f16_bounds.max_finite, 65504.0);
    assert_eq!(f16_bounds.overflow_threshold, 65520.0);
    assert_eq!(f16_bounds.min_subnormal, pow2(-24));

    let bf16_bounds = Dtype::BF16.float_bounds().unwrap();
    let bf16_max = 255.0 * pow2(120);
    assert_eq!(bf16_bounds.max_finite, bf16_max);
    assert_eq!(bf16_bounds.overflow_threshold, (bf16_max + pow2(128)) / 2.0);
    assert_eq!(bf16_bounds.min_subnormal, pow2(-133));

    // bf16's maximum finite value is strictly inside f32's: widening cannot
    // overflow, narrowing can.
    assert!(bf16_bounds.max_finite < f32_bounds.max_finite);

    // Index dtypes have no floating bounds.
    assert!(Dtype::U32.float_bounds().is_none());
}

#[test]
fn the_cast_contract_and_the_input_gate_agree_across_every_marker_pair() {
    fn layout_of<M: DtypeMarker>() -> TensorLayout {
        TensorLayout::row_major(M::DTYPE, Shape::new(&[4]).unwrap()).unwrap()
    }

    fn assert_agreement<From: DtypeMarker, To: DtypeMarker>() {
        let contract = Cast::<From, To>::contract().is_ok();
        let gate = <Cast<From, To> as TensorOp>::check_inputs(&[layout_of::<From>()]).is_ok();
        assert_eq!(
            contract,
            gate,
            "cast {} -> {} disagrees: contract ok = {contract}, input gate ok = {gate}",
            From::DTYPE,
            To::DTYPE
        );
    }

    // Exhaustive over the marker set (3 x 3). Indices have no marker, so no
    // `Cast<U32, _>` can even be named; the taxonomy test pins that refusal.
    assert_agreement::<F32, F32>();
    assert_agreement::<F32, F16>();
    assert_agreement::<F32, BF16>();
    assert_agreement::<F16, F32>();
    assert_agreement::<F16, F16>();
    assert_agreement::<F16, BF16>();
    assert_agreement::<BF16, F32>();
    assert_agreement::<BF16, F16>();
    assert_agreement::<BF16, BF16>();

    // Agreement is not vacuous: exactly the four real conversions pass the
    // gate, and the identity / mixed-narrowing pairs fail it.
    assert!(<Cast<F32, F16> as TensorOp>::check_inputs(&[layout_of::<F32>()]).is_ok());
    assert!(<Cast<F16, F32> as TensorOp>::check_inputs(&[layout_of::<F16>()]).is_ok());
    assert!(<Cast<F32, BF16> as TensorOp>::check_inputs(&[layout_of::<F32>()]).is_ok());
    assert!(<Cast<BF16, F32> as TensorOp>::check_inputs(&[layout_of::<BF16>()]).is_ok());
    assert!(<Cast<F32, F32> as TensorOp>::check_inputs(&[layout_of::<F32>()]).is_err());
    assert!(<Cast<F16, F16> as TensorOp>::check_inputs(&[layout_of::<F16>()]).is_err());
    assert!(<Cast<BF16, BF16> as TensorOp>::check_inputs(&[layout_of::<BF16>()]).is_err());
    assert!(<Cast<F16, BF16> as TensorOp>::check_inputs(&[layout_of::<F16>()]).is_err());
    assert!(<Cast<BF16, F16> as TensorOp>::check_inputs(&[layout_of::<BF16>()]).is_err());
}

#[test]
fn identity_and_non_float_casts_are_not_defined() {
    // f32 -> f32 is `copy`, never a cast.
    assert!(matches!(
        CastContract::for_dtypes(Dtype::F32, Dtype::F32),
        Err(TensorError::CastNotDefined { .. })
    ));
    // f16 -> bf16 is two-step: widen to f32, then narrow explicitly.
    assert!(matches!(
        CastContract::for_dtypes(Dtype::F16, Dtype::BF16),
        Err(TensorError::CastNotDefined { .. })
    ));
    // Indices are not a cast target.
    assert!(matches!(
        CastContract::for_dtypes(Dtype::F32, Dtype::U32),
        Err(TensorError::CastNotDefined { .. })
    ));
}

#[test]
fn dtype_inventory_excludes_f64() {
    // Handover §3 NOT-scope: f64 exactness is Amari's layer.
    let names: Vec<&'static str> = Dtype::ALL.iter().map(|d| d.name()).collect();
    assert_eq!(names, vec!["f32", "f16", "bf16", "u32"]);

    assert_eq!(Dtype::F32.size_bytes(), 4);
    assert_eq!(Dtype::F16.size_bytes(), 2);
    assert_eq!(Dtype::BF16.size_bytes(), 2);
    assert_eq!(Dtype::U32.size_bytes(), 4);

    assert!(Dtype::F32.is_contract());
    assert!(!Dtype::F16.is_contract());
    assert!(!Dtype::BF16.is_contract());
    assert!(!Dtype::U32.is_contract());
}

#[test]
fn matmul_infers_the_projection_shape() {
    let lhs = Shape::new(&[4, 8]).unwrap();
    let rhs = Shape::new(&[8, 16]).unwrap();

    let out = <MatMul as TensorOp>::infer_shape(&[&lhs, &rhs]).unwrap();
    assert_eq!(out, Shape::new(&[4, 16]).unwrap());
    assert!(
        <MatMul as TensorOp>::SHAPE_RULE
            .check(<MatMul as TensorOp>::NAME, &[&lhs, &rhs], &out)
            .is_ok()
    );

    // A declared output that disagrees with the rule is rejected.
    assert!(matches!(
        <MatMul as TensorOp>::SHAPE_RULE.check(
            "matmul",
            &[&lhs, &rhs],
            &Shape::new(&[4, 15]).unwrap()
        ),
        Err(TensorError::OutputShapeMismatch { .. })
    ));
}

#[test]
fn matmul_rejects_disagreeing_inner_dimensions() {
    let lhs = Shape::new(&[4, 8]).unwrap();
    let rhs = Shape::new(&[9, 16]).unwrap();

    assert!(matches!(
        <MatMul as TensorOp>::infer_shape(&[&lhs, &rhs]),
        Err(TensorError::InnerDimensionMismatch { lhs: 8, rhs: 9, .. })
    ));
}

#[test]
fn batched_matmul_carries_four_dimensions() {
    // [B,H,S,D] x [B,H,D,T] -> [B,H,S,T] (attention projections).
    let q = Shape::new(&[2, 4, 16, 64]).unwrap();
    let k = Shape::new(&[2, 4, 64, 16]).unwrap();

    let out = <BatchedMatMul as TensorOp>::infer_shape(&[&q, &k]).unwrap();
    assert_eq!(out, Shape::new(&[2, 4, 16, 16]).unwrap());
}

#[test]
fn batched_matmul_rejects_batch_and_rank_mismatches() {
    let a = Shape::new(&[2, 4, 16, 64]).unwrap();
    let bad_batch = Shape::new(&[3, 4, 64, 16]).unwrap();
    let bad_head = Shape::new(&[2, 5, 64, 16]).unwrap();
    let rank_three = Shape::new(&[4, 64, 16]).unwrap();

    assert!(matches!(
        <BatchedMatMul as TensorOp>::infer_shape(&[&a, &bad_batch]),
        Err(TensorError::BatchDimensionMismatch { .. })
    ));
    assert!(matches!(
        <BatchedMatMul as TensorOp>::infer_shape(&[&a, &bad_head]),
        Err(TensorError::BatchDimensionMismatch { .. })
    ));
    assert!(matches!(
        <BatchedMatMul as TensorOp>::infer_shape(&[&a, &rank_three]),
        Err(TensorError::RankMismatch { .. })
    ));
}

#[test]
fn empty_batch_and_head_dimensions_are_legal() {
    // Handover §5: empty batch/head dims are an adversarial shape class,
    // not an invalidity — they must infer, not error.
    let a = Shape::new(&[0, 4, 16, 64]).unwrap();
    let b = Shape::new(&[0, 4, 64, 16]).unwrap();
    let out = <BatchedMatMul as TensorOp>::infer_shape(&[&a, &b]).unwrap();
    assert_eq!(out, Shape::new(&[0, 4, 16, 16]).unwrap());
    assert_eq!(out.element_count(), 0);
}

#[test]
fn elementwise_ops_broadcast_like_numpy() {
    let row = Shape::new(&[1, 4]).unwrap();
    let col = Shape::new(&[3, 1]).unwrap();
    let out = <Add as TensorOp>::infer_shape(&[&row, &col]).unwrap();
    assert_eq!(out, Shape::new(&[3, 4]).unwrap());

    // Different ranks align on the trailing dimension.
    let a = Shape::new(&[3, 1, 4]).unwrap();
    let b = Shape::new(&[4]).unwrap();
    let out = <Mul as TensorOp>::infer_shape(&[&a, &b]).unwrap();
    assert_eq!(out, Shape::new(&[3, 1, 4]).unwrap());

    // Scalar broadcast.
    let s = Shape::new(&[]).unwrap();
    let out = <BroadcastDiv as TensorOp>::infer_shape(&[&row, &s]).unwrap();
    assert_eq!(out, row);
}

#[test]
fn elementwise_ops_reject_incompatible_broadcasts() {
    let a = Shape::new(&[2, 4]).unwrap();
    let b = Shape::new(&[3, 4]).unwrap();

    assert!(matches!(
        <Add as TensorOp>::infer_shape(&[&a, &b]),
        Err(TensorError::BroadcastIncompatible { .. })
    ));
    assert!(matches!(
        <BroadcastDiv as TensorOp>::infer_shape(&[&a, &b]),
        Err(TensorError::BroadcastIncompatible { .. })
    ));
}

#[test]
fn same_shape_ops_do_not_broadcast() {
    // `add`/`mul` are declared broadcast-capable; `copy` is identity.
    assert_eq!(<Copy as TensorOp>::SHAPE_RULE, ShapeRule::Identity);
    assert_eq!(<Add as TensorOp>::SHAPE_RULE, ShapeRule::Broadcast);
    assert_eq!(<Mul as TensorOp>::SHAPE_RULE, ShapeRule::Broadcast);
    assert_eq!(<BroadcastDiv as TensorOp>::SHAPE_RULE, ShapeRule::Broadcast);
}

#[test]
fn gather_looks_up_table_rows() {
    // table [V,D] + indices [N] -> [N,D] (embedding lookup).
    let table = Shape::new(&[10, 4]).unwrap();
    let indices = Shape::new(&[3]).unwrap();
    let out = <Gather as TensorOp>::infer_shape(&[&table, &indices]).unwrap();
    assert_eq!(out, Shape::new(&[3, 4]).unwrap());

    // A 1-D table is not the embedding shape.
    let flat = Shape::new(&[10]).unwrap();
    assert!(matches!(
        <Gather as TensorOp>::infer_shape(&[&flat, &indices]),
        Err(TensorError::RankMismatch { .. })
    ));
    // Scalar indices are not accepted at T0.
    let scalar = Shape::new(&[]).unwrap();
    assert!(matches!(
        <Gather as TensorOp>::infer_shape(&[&table, &scalar]),
        Err(TensorError::RankMismatch { .. })
    ));
}

#[test]
fn reductions_infer_their_contracted_outputs() {
    let scalar = Shape::new(&[]).unwrap();

    // Full reductions collapse to a rank-0 scalar.
    let tokens = Shape::new(&[2, 7, 4]).unwrap();
    assert_eq!(<Sum as TensorOp>::infer_shape(&[&tokens]).unwrap(), scalar);
    assert_eq!(<Max as TensorOp>::infer_shape(&[&tokens]).unwrap(), scalar);

    // Rowwise L2 norm: [R,D] -> [R].
    let rows = Shape::new(&[3, 5]).unwrap();
    assert_eq!(
        <RowwiseL2Norm as TensorOp>::infer_shape(&[&rows]).unwrap(),
        Shape::new(&[3]).unwrap()
    );

    // Attention-masked mean pools the token axis: [B,S,D] + [B,S] -> [B,D].
    let values = Shape::new(&[2, 7, 4]).unwrap();
    let mask = Shape::new(&[2, 7]).unwrap();
    assert_eq!(
        <MaskedMean as TensorOp>::infer_shape(&[&values, &mask]).unwrap(),
        Shape::new(&[2, 4]).unwrap()
    );

    // Mask rank must be the token axis only.
    assert!(matches!(
        <MaskedMean as TensorOp>::infer_shape(&[&values, &Shape::new(&[7]).unwrap()]),
        Err(TensorError::RankMismatch { .. })
    ));
    // Mask batch must agree with the values batch.
    assert!(matches!(
        <MaskedMean as TensorOp>::infer_shape(&[&values, &Shape::new(&[3, 7]).unwrap()]),
        Err(TensorError::BatchDimensionMismatch { .. })
    ));
}

#[test]
fn check_inputs_enforces_arity_dtype_and_shape_rule() {
    let a = TensorLayout::row_major(Dtype::F32, Shape::new(&[2, 3]).unwrap()).unwrap();
    let b = TensorLayout::row_major(Dtype::F32, Shape::new(&[2, 3]).unwrap()).unwrap();
    let wide = TensorLayout::row_major(Dtype::F32, Shape::new(&[2, 4]).unwrap()).unwrap();
    let half = TensorLayout::row_major(Dtype::F16, Shape::new(&[2, 3]).unwrap()).unwrap();

    assert!(<Add as TensorOp>::check_inputs(&[a.clone(), b.clone()]).is_ok());

    // Arity.
    assert!(matches!(
        <Add as TensorOp>::check_inputs(std::slice::from_ref(&a)),
        Err(TensorError::ArityMismatch {
            expected: 2,
            got: 1,
            ..
        })
    ));
    // Dtype policy: f16 never enters f32 arithmetic silently.
    assert!(matches!(
        <Add as TensorOp>::check_inputs(&[a.clone(), half.clone()]),
        Err(TensorError::DtypeMismatch { port: 1, .. })
    ));
    // Shape rule: [2,3] + [2,4] does not broadcast.
    assert!(matches!(
        <Add as TensorOp>::check_inputs(&[a.clone(), wide]),
        Err(TensorError::BroadcastIncompatible { .. })
    ));
    // Gather indices must be U32.
    let table = TensorLayout::row_major(Dtype::F32, Shape::new(&[10, 4]).unwrap()).unwrap();
    let idx = TensorLayout::row_major(Dtype::F32, Shape::new(&[3]).unwrap()).unwrap();
    assert!(matches!(
        <Gather as TensorOp>::check_inputs(&[table, idx]),
        Err(TensorError::DtypeMismatch {
            port: 1,
            expected: Dtype::U32,
            ..
        })
    ));
}

#[test]
fn layouts_are_row_major_contiguous_only() {
    let shape = Shape::new(&[2, 3]).unwrap();

    // Canonical row-major strides for [2,3] are [3,1].
    let layout = TensorLayout::row_major(Dtype::F32, shape.clone()).unwrap();
    assert_eq!(layout.strides().as_slice(), &[3u64, 1]);
    assert!(layout.is_row_major_contiguous());
    assert_eq!(layout.byte_len(), 24);

    assert!(TensorLayout::new(Dtype::F32, shape.clone(), Strides::new(&[3, 1])).is_ok());
    // Padded / transposed strides are rejected at the contract boundary.
    assert!(matches!(
        TensorLayout::new(Dtype::F32, shape.clone(), Strides::new(&[4, 1])),
        Err(TensorError::NonContiguous { .. })
    ));
    assert!(matches!(
        TensorLayout::new(Dtype::F32, shape, Strides::new(&[1, 2])),
        Err(TensorError::NonContiguous { .. })
    ));
}

#[test]
fn byte_lengths_must_match_shape_and_dtype() {
    let layout = TensorLayout::row_major(Dtype::F32, Shape::new(&[2, 3]).unwrap()).unwrap();

    assert!(layout.check_byte_len(24).is_ok());
    assert!(matches!(
        layout.check_byte_len(20),
        Err(TensorError::ByteLenMismatch {
            expected: 24,
            got: 20,
            ..
        })
    ));

    let half = TensorLayout::row_major(Dtype::F16, Shape::new(&[2, 3]).unwrap()).unwrap();
    assert_eq!(half.byte_len(), 12);
}

#[test]
fn byte_footprints_that_overflow_u64_are_rejected_not_saturated() {
    // Reviewer repro (PR #65 round 1): [u32::MAX, u32::MAX] f32 holds
    // 18_446_744_065_119_617_025 elements (which fits in u64) but would need
    // 73_786_976_260_478_468_100 bytes (which does not). Saturating to
    // u64::MAX made that layout accept a byte length it cannot represent.
    let shape = Shape::new(&[u32::MAX, u32::MAX]).unwrap();
    assert_eq!(shape.element_count(), 18_446_744_065_119_617_025);
    assert_eq!(
        u128::from(shape.element_count()) * 4,
        73_786_976_260_478_468_100
    );
    assert!(shape.element_count().checked_mul(4).is_none());

    assert!(matches!(
        TensorLayout::row_major(Dtype::F32, shape.clone()),
        Err(TensorError::ByteLenOverflow {
            dtype: Dtype::F32,
            ..
        })
    ));
    // Same shape in a 2-byte dtype: still past u64::MAX / 2.
    assert!(matches!(
        TensorLayout::row_major(Dtype::F16, shape),
        Err(TensorError::ByteLenOverflow {
            dtype: Dtype::F16,
            ..
        })
    ));
}

#[test]
fn a_zero_dimension_is_detected_before_the_product_overflows() {
    // Reviewer repro (PR #65 round 1): a left-to-right fold overflows on
    // [MAX, MAX, MAX, 0] before it reaches the zero, while
    // [0, MAX, MAX, MAX] short-circuits. Both shapes hold zero elements, so
    // both must construct.
    let trailing = Shape::new(&[u32::MAX, u32::MAX, u32::MAX, 0]).unwrap();
    let leading = Shape::new(&[0, u32::MAX, u32::MAX, u32::MAX]).unwrap();
    assert_eq!(trailing.element_count(), 0);
    assert_eq!(leading.element_count(), 0);

    // A zero-element layout has a zero-byte footprint, so it is bindable.
    let layout = TensorLayout::row_major(Dtype::F32, trailing).unwrap();
    assert_eq!(layout.byte_len(), 0);

    // Without a zero dimension the product genuinely overflows.
    assert!(matches!(
        Shape::new(&[u32::MAX, u32::MAX, u32::MAX]),
        Err(TensorError::ShapeOverflow { .. })
    ));
}

#[test]
fn shape_element_counts_are_overflow_checked() {
    let huge = [u32::MAX, u32::MAX, u32::MAX];
    assert!(matches!(
        Shape::new(&huge),
        Err(TensorError::ShapeOverflow { .. })
    ));

    // A zero dimension is legal and yields a zero count.
    let empty = Shape::new(&[0, 4]).unwrap();
    assert_eq!(empty.element_count(), 0);
    assert!(!empty.is_scalar());
    assert!(Shape::new(&[]).unwrap().is_scalar());
}

#[test]
fn error_taxonomy_is_exhaustive() {
    // One constructed value per variant: the match below stops compiling the
    // moment a variant is added or renamed, which is the point of the test.
    fn slug(err: &TensorError) -> &'static str {
        match err {
            TensorError::UnalignedOffset { .. } => "unaligned_offset",
            TensorError::ShapeOverflow { .. } => "shape_overflow",
            TensorError::ByteLenOverflow { .. } => "byte_len_overflow",
            TensorError::NonContiguous { .. } => "non_contiguous",
            TensorError::ArityMismatch { .. } => "arity_mismatch",
            TensorError::DtypeMismatch { .. } => "dtype_mismatch",
            TensorError::RankMismatch { .. } => "rank_mismatch",
            TensorError::InnerDimensionMismatch { .. } => "inner_dimension_mismatch",
            TensorError::BatchDimensionMismatch { .. } => "batch_dimension_mismatch",
            TensorError::BroadcastIncompatible { .. } => "broadcast_incompatible",
            TensorError::OutputShapeMismatch { .. } => "output_shape_mismatch",
            TensorError::AxisOutOfRange { .. } => "axis_out_of_range",
            TensorError::CastNotDefined { .. } => "cast_not_defined",
            TensorError::ByteLenMismatch { .. } => "byte_len_mismatch",
            TensorError::OutOfBounds { .. } => "out_of_bounds",
            TensorError::OverlappingEntries { .. } => "overlapping_entries",
            TensorError::ManifestArityMismatch { .. } => "manifest_arity_mismatch",
            TensorError::DeterminismDtypeMismatch { .. } => "determinism_dtype_mismatch",
            TensorError::NonContractDtype { .. } => "non_contract_dtype",
            TensorError::NonFiniteInput { .. } => "non_finite_input",
            TensorError::IndexOutOfRange { .. } => "index_out_of_range",
            TensorError::EmptyReduction { .. } => "empty_reduction",
        }
    }

    let cases = [
        TensorError::ShapeOverflow { shape: vec![1] },
        TensorError::ByteLenOverflow {
            shape: vec![2],
            dtype: Dtype::F32,
        },
        TensorError::NonContiguous {
            shape: vec![1],
            strides: vec![1],
        },
        TensorError::ArityMismatch {
            op: "add",
            expected: 2,
            got: 1,
        },
        TensorError::DtypeMismatch {
            op: "add",
            port: 0,
            expected: Dtype::F32,
            got: Dtype::F16,
        },
        TensorError::RankMismatch {
            op: "matmul",
            port: 0,
            expected: "rank 2",
            got: 3,
        },
        TensorError::InnerDimensionMismatch {
            op: "matmul",
            lhs: 8,
            rhs: 9,
        },
        TensorError::BatchDimensionMismatch {
            op: "batched_matmul",
            lhs: vec![2],
            rhs: vec![3],
        },
        TensorError::BroadcastIncompatible {
            op: "add",
            lhs: vec![2],
            rhs: vec![3],
        },
        TensorError::OutputShapeMismatch {
            op: "copy",
            expected: vec![1],
            got: vec![2],
        },
        TensorError::AxisOutOfRange {
            op: "sum",
            axis: 3,
            rank: 2,
        },
        TensorError::CastNotDefined {
            from: Dtype::F32,
            to: Dtype::F32,
        },
        TensorError::ByteLenMismatch {
            name: "w".to_string(),
            shape: vec![2],
            dtype: Dtype::F32,
            expected: 8,
            got: 4,
        },
        TensorError::OutOfBounds {
            name: "w".to_string(),
            offset: 8,
            end: 16,
            available: 12,
        },
        TensorError::OverlappingEntries {
            first: "a.weight".to_string(),
            second: "b.weight".to_string(),
            offset: 0,
            end: 4,
        },
        TensorError::ManifestArityMismatch {
            op: "matmul",
            expected: 2,
            got: 1,
        },
        TensorError::DeterminismDtypeMismatch {
            pinned: Dtype::F32,
            manifest: Dtype::F16,
        },
        TensorError::NonContractDtype { got: Dtype::F16 },
        // T1 additions: the execution-time rejections. A non-finite operand,
        // a gather index past the table, and a reduction with no identity
        // element are all typed, not panics.
        TensorError::NonFiniteInput {
            op: "sum",
            port: 0,
            index: 3,
        },
        TensorError::IndexOutOfRange {
            op: "gather",
            position: 1,
            index: 9,
            rows: 4,
        },
        TensorError::EmptyReduction { op: "max" },
    ];

    let slugs: Vec<&'static str> = cases.iter().map(slug).collect();
    assert_eq!(
        slugs,
        vec![
            "shape_overflow",
            "byte_len_overflow",
            "non_contiguous",
            "arity_mismatch",
            "dtype_mismatch",
            "rank_mismatch",
            "inner_dimension_mismatch",
            "batch_dimension_mismatch",
            "broadcast_incompatible",
            "output_shape_mismatch",
            "axis_out_of_range",
            "cast_not_defined",
            "byte_len_mismatch",
            "out_of_bounds",
            "overlapping_entries",
            "manifest_arity_mismatch",
            "determinism_dtype_mismatch",
            "non_contract_dtype",
            "non_finite_input",
            "index_out_of_range",
            "empty_reduction",
        ]
    );

    // Errors render with the op name, not a bare "invalid argument".
    // Located by variant, not by index: the taxonomy grows, and a positional
    // assertion would silently start checking a different error.
    let arity = cases
        .iter()
        .find(|error| matches!(error, TensorError::ArityMismatch { .. }))
        .expect("the taxonomy covers ArityMismatch");
    assert!(arity.to_string().contains("add"));
}
