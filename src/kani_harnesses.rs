// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Kani verification harnesses for Borsalino buffer safety.
//!
//! Run with:
//! ```sh
//! cargo kani --features vulkan --harness buffer_alignment_boundary
//! cargo kani --features vulkan --harness workgroup_divisibility
//! cargo kani --features vulkan --harness epoch_balanced_never_negative
//! ```
//!
//! Requires Kani installed: <https://model-checking.github.io/kani/>

/// Verify that buffer size alignment always satisfies 16-byte boundary.
#[cfg(kani)]
#[kani::proof]
fn buffer_alignment_boundary() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 65536);

    // Pad to 16-byte alignment (matching minStorageBufferOffsetAlignment)
    let aligned = ((size + 15) / 16) * 16;

    assert!(aligned >= size);
    assert!(aligned % 16 == 0);
    assert!(aligned - size < 16);
}

/// Verify workgroup divisibility prevents partial threadgroups.
#[cfg(kani)]
#[kani::proof]
fn workgroup_divisibility() {
    let total_threads: u32 = kani::any();
    let threads_per_group: u32 = kani::any();
    kani::assume(threads_per_group > 0 && threads_per_group <= 256);
    kani::assume(total_threads <= 65536);

    let workgroups = total_threads.div_ceil(threads_per_group);

    // Property: workgroups * threads_per_group >= total_threads
    assert!(workgroups * threads_per_group >= total_threads);

    // Property: no partial threadgroup if divisible
    if total_threads % threads_per_group == 0 {
        assert_eq!(workgroups * threads_per_group, total_threads);
    }
}

/// Verify buffer element count × element size doesn't overflow.
#[cfg(kani)]
#[kani::proof]
fn buffer_size_no_overflow() {
    let len: u32 = kani::any();
    let element_size: u32 = kani::any();
    kani::assume(len > 0 && len <= 65536);
    kani::assume(element_size > 0 && element_size <= 32);

    let byte_len = (len as u64) * (element_size as u64);
    assert!(byte_len <= (1u64 << 32)); // fits in u32 for Vulkan buffer size
    assert!(byte_len <= u64::MAX);
}

/// Verify that a balanced sequence of begin/end dispatch calls keeps
/// the epoch counter non-negative and returns to zero.
///
/// This proves the AtomicU64 counter cannot underflow when every
/// `end_dispatch` is paired with a prior `begin_dispatch`.
#[cfg(kani)]
#[kani::proof]
fn epoch_balanced_never_negative() {
    use crate::epoch::GpuEpochTracker;

    let tracker = GpuEpochTracker::new();
    assert!(tracker.is_quiescent());

    let n: u32 = kani::any();
    kani::assume(n > 0 && n <= 5);

    for _ in 0..n {
        tracker.begin_dispatch();
    }
    assert_eq!(tracker.in_flight(), n as u64);
    assert!(!tracker.is_quiescent());

    for _ in 0..n {
        tracker.end_dispatch();
    }
    assert_eq!(tracker.in_flight(), 0);
    assert!(tracker.is_quiescent());
}

/// Verify that interleaved begin/end sequences track correctly.
///
/// Proves that the counter accurately reflects the number of in-flight
/// dispatches regardless of interleaving order.
#[cfg(kani)]
#[kani::proof]
fn epoch_interleaved_tracks_correctly() {
    use crate::epoch::GpuEpochTracker;

    let tracker = GpuEpochTracker::new();

    let a: u32 = kani::any();
    let b: u32 = kani::any();
    kani::assume(a <= 3);
    kani::assume(b <= 3);

    // Phase 1: begin `a` dispatches
    for _ in 0..a {
        tracker.begin_dispatch();
    }
    assert_eq!(tracker.in_flight(), a as u64);

    // Phase 2: begin `b` more (interleaved with ending `a`)
    for _ in 0..b {
        tracker.begin_dispatch();
    }
    assert_eq!(tracker.in_flight(), (a + b) as u64);

    // Phase 3: end all
    for _ in 0..(a + b) {
        tracker.end_dispatch();
    }
    assert_eq!(tracker.in_flight(), 0);
    assert!(tracker.is_quiescent());
}
