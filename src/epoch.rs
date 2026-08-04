// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! GPU dispatch epoch tracking for GC safety.
//!
//! When a WASM runtime (e.g., Baedeker) runs a moving garbage collector,
//! it must know whether any GPU dispatches are in-flight before compacting
//! host memory. A zero-copy GPU buffer backed by GC-managed memory would
//! dangle if the GC moves the underlying allocation.
//!
//! [`GpuEpochTracker`][crate::epoch::GpuEpochTracker] provides an atomic counter
//! that is incremented on dispatch and decremented on completion. The WASM
//! runtime checks
//! [`is_quiescent`][crate::epoch::GpuEpochTracker::is_quiescent] before
//! compaction — if false, it defers compaction until the GPU quiesces.
//!
//! # Performance
//!
//! Each dispatch costs one `AtomicU64::fetch_add` and each completion costs
//! one `AtomicU64::fetch_sub`. A GC check is one `AtomicU64::load`. This is
//! negligible against the microseconds of GPU dispatch overhead already
//! measured (0.5–142 µs per dispatch).

use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks outstanding GPU dispatches for GC safety.
///
/// Embedded in each backend struct. Call [`begin_dispatch`](Self::begin_dispatch)
/// before submitting work to the GPU and [`end_dispatch`](Self::end_dispatch)
/// when the GPU completes.
///
/// # Examples
///
/// ```
/// use borsalino::epoch::GpuEpochTracker;
///
/// let tracker = GpuEpochTracker::new();
/// assert!(tracker.is_quiescent());
///
/// tracker.begin_dispatch();
/// assert!(!tracker.is_quiescent()); // GPU work in-flight
///
/// tracker.end_dispatch();
/// assert!(tracker.is_quiescent()); // safe to GC compact
/// ```
#[derive(Debug)]
pub struct GpuEpochTracker {
    in_flight: AtomicU64,
}

impl Default for GpuEpochTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuEpochTracker {
    /// Create a new epoch tracker with zero in-flight dispatches.
    pub const fn new() -> Self {
        Self {
            in_flight: AtomicU64::new(0),
        }
    }

    /// Record the start of a GPU dispatch.
    ///
    /// Increments the in-flight counter. The counter must be balanced by a
    /// later call to [`end_dispatch`](Self::end_dispatch).
    pub fn begin_dispatch(&self) {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
    }

    /// Record the completion of a GPU dispatch.
    ///
    /// Decrements the in-flight counter. Must be paired with a prior
    /// [`begin_dispatch`](Self::begin_dispatch).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the counter would go negative, which
    /// indicates an unmatched `end_dispatch` (a readback without a dispatch).
    pub fn end_dispatch(&self) {
        let prev = self.in_flight.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(
            prev > 0,
            "end_dispatch called without matching begin_dispatch (counter underflow)"
        );
    }

    /// Number of dispatches that have begun but not yet completed.
    pub fn in_flight(&self) -> u64 {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// True when no GPU operations are outstanding.
    ///
    /// The WASM runtime (Baedeker) calls this before GC compaction.
    /// If `false`, compaction must be deferred until the GPU quiesces.
    pub fn is_quiescent(&self) -> bool {
        self.in_flight() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_quiescent() {
        let tracker = GpuEpochTracker::new();
        assert_eq!(tracker.in_flight(), 0);
        assert!(tracker.is_quiescent());
    }

    #[test]
    fn begin_dispatch_increments_counter() {
        let tracker = GpuEpochTracker::new();
        tracker.begin_dispatch();
        assert_eq!(tracker.in_flight(), 1);
        assert!(!tracker.is_quiescent());
    }

    #[test]
    fn end_dispatch_decrements_counter() {
        let tracker = GpuEpochTracker::new();
        tracker.begin_dispatch();
        tracker.end_dispatch();
        assert_eq!(tracker.in_flight(), 0);
        assert!(tracker.is_quiescent());
    }

    #[test]
    fn multiple_concurrent_dispatches_track_correctly() {
        let tracker = GpuEpochTracker::new();
        tracker.begin_dispatch();
        tracker.begin_dispatch();
        tracker.begin_dispatch();
        assert_eq!(tracker.in_flight(), 3);

        tracker.end_dispatch();
        assert_eq!(tracker.in_flight(), 2);
        assert!(!tracker.is_quiescent());

        tracker.end_dispatch();
        tracker.end_dispatch();
        assert_eq!(tracker.in_flight(), 0);
        assert!(tracker.is_quiescent());
    }

    #[test]
    fn interleaved_begin_end_tracks_correctly() {
        let tracker = GpuEpochTracker::new();

        // dispatch A starts
        tracker.begin_dispatch();
        assert_eq!(tracker.in_flight(), 1);

        // dispatch B starts while A is in-flight
        tracker.begin_dispatch();
        assert_eq!(tracker.in_flight(), 2);

        // dispatch A completes
        tracker.end_dispatch();
        assert_eq!(tracker.in_flight(), 1);

        // dispatch B completes
        tracker.end_dispatch();
        assert!(tracker.is_quiescent());
    }

    #[test]
    fn default_trait_impl_matches_new() {
        let default = GpuEpochTracker::default();
        assert_eq!(default.in_flight(), 0);
        assert!(default.is_quiescent());
    }
}
