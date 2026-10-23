// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Dtype policy: **f32 is the contract**; f16/bf16 are explicit casts.
//!
//! Handover §4.3 draws the line this module enforces:
//!
//! - f32 is the type every kernel computes in. Weights arrive as f32 from
//!   Katakuri, buffers are bound as f32, and no backend may substitute a
//!   narrower or faster representation (no TF32, no f16 accumulation).
//! - f16 and bf16 exist **only** as the target or source of an explicit
//!   [`Cast`](crate::ops::l0::Cast), with the roundtrip semantics documented
//!   in [`CastContract`]. Nothing narrows silently.
//! - `u32` is the index type ([`Gather`](crate::ops::l0::Gather)'s second
//!   port). It is not a compute dtype and not a cast target.
//! - f64 is not here at all: f64 exactness is Amari's layer, not
//!   Borsalino's (handover §3, NOT-scope).

use core::fmt;

use crate::error::TensorError;

/// An element type the tensor op-set can name.
///
/// The inventory is closed by design: [`Dtype::ALL`] is the contract's
/// complete dtype surface, so a new element type is a deliberate contract
/// change rather than an accident of a dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit IEEE-754 float — the one compute dtype (handover §4.3).
    F32,
    /// IEEE-754 binary16 — a cast source or target only.
    F16,
    /// bfloat16 — f32's exponent range, 8-bit significand. Cast only.
    BF16,
    /// Unsigned 32-bit integer — index ports only, never computed on.
    U32,
}

impl Dtype {
    /// Every dtype the contract knows, in canonical order.
    pub const ALL: &'static [Dtype] = &[Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::U32];

    /// The dtype's lowercase name, as it appears in WGSL and manifests.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::U32 => "u32",
        }
    }

    /// The dtype's size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Dtype::F32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }

    /// Whether this dtype is the compute contract dtype.
    ///
    /// Only [`Dtype::F32`] is: every arithmetic op's data ports declare it,
    /// and any other float must be cast to f32 first.
    #[must_use]
    pub const fn is_contract(self) -> bool {
        matches!(self, Dtype::F32)
    }

    /// Whether this dtype may appear on an index port.
    #[must_use]
    pub const fn is_index(self) -> bool {
        matches!(self, Dtype::U32)
    }

    /// The finite bounds of a floating dtype, as exact `f64` values.
    ///
    /// `None` for [`Dtype::U32`]: an index dtype is not a rounding target,
    /// so "maximum finite" has no meaning for it.
    ///
    /// The bounds are exact (every value is representable in `f64`), because
    /// each is either a value of a narrower float or the midpoint of two of
    /// them.
    #[must_use]
    pub fn float_bounds(self) -> Option<FloatBounds> {
        let bounds = match self {
            Dtype::F32 => FloatBounds {
                max_finite: f64::from(f32::MAX),
                overflow_threshold: (f64::from(f32::MAX) + pow2(128)) / 2.0,
                min_subnormal: pow2(-149),
            },
            Dtype::F16 => FloatBounds {
                max_finite: 65504.0,
                // f16's last binade is [2^15, 2^16): the next representable
                // magnitude after 65504 is 2^16, so rounding overflows from
                // their midpoint onwards.
                overflow_threshold: 65520.0,
                min_subnormal: pow2(-24),
            },
            Dtype::BF16 => {
                // (2 - 2^-7) * 2^127 — bf16 keeps f32's 8-bit exponent but
                // narrows the significand to 8 bits, so its largest finite
                // value is *smaller* than f32's.
                let max_finite = 255.0 * pow2(120);
                FloatBounds {
                    max_finite,
                    overflow_threshold: (max_finite + pow2(128)) / 2.0,
                    min_subnormal: pow2(-133),
                }
            }
            Dtype::U32 => return None,
        };
        Some(bounds)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// 2^`exp` as an exact `f64`.
///
/// Constructed from exponent bits rather than a decimal literal so every
/// bound below is exact by construction; the exponents used stay well inside
/// `f64`'s normal range.
fn pow2(exp: i32) -> f64 {
    f64::from_bits(((1023 + exp) as u64) << 52)
}

/// The finite bounds of a floating dtype (IEEE-754 binaries).
///
/// These are the numbers that make a cast contract checkable: "overflow" and
/// "flushes to zero" are only meaningful claims once the thresholds are
/// pinned, because rounding is *nearest-first* — a value above the maximum
/// finite value still rounds **down** to it as long as it is below the
/// midpoint to the next power of two.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatBounds {
    /// The largest finite value the dtype represents.
    pub max_finite: f64,
    /// At or above this magnitude, rounding to the dtype yields infinity.
    ///
    /// This is the midpoint between [`max_finite`](Self::max_finite) and the
    /// next power of two, not the maximum finite value itself.
    pub overflow_threshold: f64,
    /// The smallest positive subnormal; below half of it, a nonzero source
    /// value rounds to zero.
    pub min_subnormal: f64,
}

/// Reject a dtype that is not the f32 contract dtype.
///
/// v1 manifests are f32-only, so this guard cannot fire today; it exists so
/// that the load boundary has one place to enforce
/// [`Dtype::is_contract`] when a manifest type grows a second dtype.
///
/// # Errors
///
/// [`TensorError::NonContractDtype`] when `dtype` is not f32.
pub fn require_contract(dtype: Dtype) -> Result<(), TensorError> {
    if dtype.is_contract() {
        Ok(())
    } else {
        Err(TensorError::NonContractDtype { got: dtype })
    }
}

/// A dtype that can be named as a compile-time operand marker.
///
/// Markers let an op carry its dtype in its type: `Cast<F32, F16>` *is* a
/// narrowing cast, checked by the type system before any shape is seen.
/// Only the three floating types have markers — indices are never cast.
pub trait DtypeMarker: Copy + fmt::Debug + 'static {
    /// The runtime dtype this marker denotes.
    const DTYPE: Dtype;
}

/// The f32 marker — the contract dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32;

impl DtypeMarker for F32 {
    const DTYPE: Dtype = Dtype::F32;
}

/// The f16 marker — cast source or target only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F16;

impl DtypeMarker for F16 {
    const DTYPE: Dtype = Dtype::F16;
}

/// The bf16 marker — cast source or target only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BF16;

impl DtypeMarker for BF16 {
    const DTYPE: Dtype = Dtype::BF16;
}

/// What a cast does to the bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Roundtrip {
    /// The cast is lossless: every source value is representable.
    Exact,
    /// The cast rounds to nearest-even, so a downcast roundtrip is inexact
    /// by up to half an ulp.
    HalfUlpInexact,
}

/// What a cast does to values beyond the target's range.
///
/// Rounding is nearest-first, so "beyond the range" is not the same as
/// "above the maximum finite value": the overflow point is the midpoint to
/// the next power of two (see [`FloatBounds::overflow_threshold`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// Values at or above the target's overflow threshold become infinity;
    /// values between the maximum finite value and that threshold round back
    /// down to the maximum finite value.
    ToInfinity,
    /// The target's range contains the source's, so no source value can
    /// reach the target's overflow threshold.
    CannotOverflow,
}

/// What a cast does to subnormal source values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubnormalPolicy {
    /// Every source subnormal is exactly representable in the target.
    Preserved,
    /// Source subnormals land on target subnormals, down to the target's
    /// minimum subnormal; below half of it they round to zero.
    ///
    /// This is the `f32 -> bf16` case: bf16's exponent range matches f32's,
    /// so its minimum subnormal (2^-133) is sixteen binades above f32's
    /// deepest subnormal.
    RoundsToTargetSubnormals,
    /// Every source subnormal is below half of the target's minimum
    /// subnormal, so all of them round to zero.
    ///
    /// This is the `f32 -> f16` case: f16's minimum subnormal (2^-24) sits
    /// 102 binades above f32's largest subnormal (just below 2^-126).
    FlushesToZero,
}

/// The documented semantics of one cast pair.
///
/// This is the whole of T0's cast contract: dtype policy is data, so a
/// kernel author reads the semantics from the type instead of guessing at
/// the hardware's rounding mode. The three threshold fields are the target
/// dtype's, so a caller can decide what a specific value will do without
/// implementing rounding: below `max_finite` rounds back to `max_finite`,
/// from `overflow_threshold` onwards it is infinity, and below half of
/// `min_subnormal` it is zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastContract {
    /// The source dtype.
    pub from: Dtype,
    /// The target dtype.
    pub to: Dtype,
    /// Whether the value survives a roundtrip.
    pub roundtrip: Roundtrip,
    /// What happens beyond the target's range.
    pub overflow: Overflow,
    /// What happens to subnormal source values.
    pub subnormals: SubnormalPolicy,
    /// The target's largest finite value.
    pub max_finite: f64,
    /// The target's overflow threshold: at or above it, infinity.
    pub overflow_threshold: f64,
    /// The target's smallest positive subnormal.
    pub min_subnormal: f64,
}

impl CastContract {
    /// The contract for a cast pair.
    ///
    /// Defined pairs (handover §4.3), all round-to-nearest-even:
    ///
    /// - `f32 -> f16`: half-ulp inexact; values from 65520 (the midpoint
    ///   between 65504 and 2^16) become infinity, values in between round
    ///   back to 65504; every f32 subnormal is far below half of f16's
    ///   minimum subnormal, so all of them become zero.
    /// - `f32 -> bf16`: half-ulp inexact; bf16 has f32's exponent range but
    ///   not f32's maximum finite value — `(2 - 2^-7) * 2^127 < f32::MAX` —
    ///   so this narrowing **can** overflow, from the midpoint to 2^128.
    ///   f32 subnormals survive as bf16 subnormals down to 2^-133.
    /// - `f16 -> f32`, `bf16 -> f32`: exact, no overflow, subnormals
    ///   preserved (both fit f32's exponent range with room to spare).
    ///
    /// # Errors
    ///
    /// [`TensorError::CastNotDefined`] for any other pair — including the
    /// identity cast (that is `copy`) and `f16 <-> bf16` (two explicit steps
    /// through f32), and for index dtypes.
    pub fn for_dtypes(from: Dtype, to: Dtype) -> Result<Self, TensorError> {
        use Dtype::{BF16 as B, F16 as H, F32 as S};
        let (roundtrip, overflow, subnormals) = match (from, to) {
            (S, H) => (
                Roundtrip::HalfUlpInexact,
                Overflow::ToInfinity,
                SubnormalPolicy::FlushesToZero,
            ),
            (S, B) => (
                Roundtrip::HalfUlpInexact,
                Overflow::ToInfinity,
                SubnormalPolicy::RoundsToTargetSubnormals,
            ),
            (H, S) | (B, S) => (
                Roundtrip::Exact,
                Overflow::CannotOverflow,
                SubnormalPolicy::Preserved,
            ),
            _ => return Err(TensorError::CastNotDefined { from, to }),
        };
        // Every defined pair has a float target, so the bounds exist.
        let bounds = to
            .float_bounds()
            .expect("defined cast targets are floating dtypes");
        Ok(Self {
            from,
            to,
            roundtrip,
            overflow,
            subnormals,
            max_finite: bounds.max_finite,
            overflow_threshold: bounds.overflow_threshold,
            min_subnormal: bounds.min_subnormal,
        })
    }
}
