// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Statistical determinism verification for GPU kernels.
//!
//! Dispatches the same inputs multiple times and compares outputs
//! bit-for-bit. If all runs produce identical results, the kernel is
//! deterministic within empirical bounds. This catches nondeterministic
//! atomic operations, race conditions, and implementation-defined
//! reductions.
//!
//! Unlike full amari-flynn probabilistic contracts, this is a concrete
//! "dispatch N times, compare" check — simple and effective for catching
//! practical nondeterminism.

use crate::{ComputePipeline, GpuBackend, GpuBuffer, Result};

/// Result of a determinism check.
#[derive(Clone, Debug, PartialEq)]
pub struct DeterminismResult {
    /// Number of dispatch trials performed.
    pub trials: u32,
    /// True if all trials produced bit-identical output.
    pub all_identical: bool,
    /// Fraction of output bytes that differed across trials.
    /// `0.0` means perfect agreement. `1.0` means all bytes differed.
    pub disagreement_fraction: f32,
}

impl DeterminismResult {
    /// Returns `true` if the kernel passed the determinism check
    /// (`all_identical == true`).
    pub fn is_deterministic(&self) -> bool {
        self.all_identical
    }
}

/// Compare multiple output byte vectors for bit-exact agreement.
///
/// Returns a [`DeterminismResult`] summarising whether all outputs are
/// identical and what fraction of bytes disagreed.
///
/// # Examples
///
/// ```
/// use borsalino::determinism::compare_determinism;
///
/// let outputs = vec![
///     vec![0u8, 1, 2, 3],
///     vec![0u8, 1, 2, 3],
/// ];
/// let result = compare_determinism(&outputs);
/// assert!(result.is_deterministic());
/// assert_eq!(result.disagreement_fraction, 0.0);
/// ```
pub fn compare_determinism(outputs: &[Vec<u8>]) -> DeterminismResult {
    if outputs.is_empty() {
        return DeterminismResult {
            trials: 0,
            all_identical: false,
            disagreement_fraction: 1.0,
        };
    }

    if outputs.len() == 1 {
        return DeterminismResult {
            trials: 1,
            all_identical: true,
            disagreement_fraction: 0.0,
        };
    }

    let reference = &outputs[0];
    let len = reference.len();
    let mut all_identical = true;
    let mut total_disagreeing_bytes = 0usize;

    for output in &outputs[1..] {
        if output.len() != len {
            all_identical = false;
            total_disagreeing_bytes += len.max(output.len());
            continue;
        }
        for (ref_byte, out_byte) in reference.iter().zip(output.iter()) {
            if ref_byte != out_byte {
                all_identical = false;
                total_disagreeing_bytes += 1;
            }
        }
    }

    let total_bytes = len * (outputs.len() - 1);
    let disagreement_fraction = if total_bytes == 0 {
        0.0
    } else {
        total_disagreeing_bytes as f32 / total_bytes as f32
    };

    DeterminismResult {
        trials: outputs.len() as u32,
        all_identical,
        disagreement_fraction,
    }
}

/// Dispatch the same inputs N times, compare outputs bit-for-bit.
///
/// For each of `trials` dispatches: creates fresh buffers from the input
/// data, dispatches the pipeline, reads back the output. All outputs are
/// then compared via [`compare_determinism`].
///
/// If all trials match, the kernel is deterministic within empirical
/// bounds. This catches nondeterministic atomic operations, race
/// conditions, and implementation-defined reductions.
///
/// # Arguments
///
/// * `gpu` — Any [`GpuBackend`] implementation.
/// * `pipeline` — The compiled compute pipeline to test.
/// * `input_data` — One `Vec<u8>` per input binding.
/// * `output_len` — Expected output size in bytes.
/// * `trials` — Number of dispatch+readback cycles.
///
/// # Errors
///
/// Returns an error if any individual dispatch or readback fails.
pub fn verify_deterministic(
    gpu: &impl GpuBackend,
    pipeline: &ComputePipeline,
    input_data: &[Vec<u8>],
    output_len: usize,
    trials: u32,
) -> Result<DeterminismResult> {
    let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(trials as usize);

    for _ in 0..trials {
        // Create fresh input buffers each trial
        let input_buffers: Vec<GpuBuffer> = input_data
            .iter()
            .map(|data| gpu.create_buffer(data))
            .collect::<Result<Vec<_>>>()?;

        let output_buffer = gpu.create_buffer_uninit::<u8>(output_len)?;

        let buffer_refs: Vec<&GpuBuffer> = input_buffers
            .iter()
            .chain(std::iter::once(&output_buffer))
            .collect();

        gpu.dispatch(pipeline, &buffer_refs, (1, 1, 1))?;

        let output: Vec<u8> = gpu.read_buffer(&output_buffer)?;
        outputs.push(output);
    }

    Ok(compare_determinism(&outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_outputs_are_deterministic() {
        let outputs = vec![
            vec![0u8, 1, 2, 3, 4, 5],
            vec![0u8, 1, 2, 3, 4, 5],
            vec![0u8, 1, 2, 3, 4, 5],
        ];
        let result = compare_determinism(&outputs);
        assert!(result.is_deterministic());
        assert!(result.all_identical);
        assert_eq!(result.disagreement_fraction, 0.0);
        assert_eq!(result.trials, 3);
    }

    #[test]
    fn differing_outputs_are_not_deterministic() {
        let outputs = vec![
            vec![0u8, 1, 2, 3],
            vec![0u8, 1, 2, 4], // last byte differs
            vec![0u8, 1, 2, 3],
        ];
        let result = compare_determinism(&outputs);
        assert!(!result.is_deterministic());
        assert!(!result.all_identical);
        assert!(result.disagreement_fraction > 0.0);
    }

    #[test]
    fn single_output_is_trivially_deterministic() {
        let outputs = vec![vec![1u8, 2, 3]];
        let result = compare_determinism(&outputs);
        assert!(result.is_deterministic());
        assert_eq!(result.trials, 1);
    }

    #[test]
    fn empty_outputs_are_not_deterministic() {
        let outputs: Vec<Vec<u8>> = vec![];
        let result = compare_determinism(&outputs);
        assert!(!result.is_deterministic());
        assert_eq!(result.trials, 0);
    }

    #[test]
    fn mismatched_lengths_count_as_disagreement() {
        let outputs = vec![
            vec![0u8, 1, 2],
            vec![0u8, 1], // shorter
        ];
        let result = compare_determinism(&outputs);
        assert!(!result.is_deterministic());
        assert!(result.disagreement_fraction > 0.0);
    }

    #[test]
    fn disagreement_fraction_quantifies_difference() {
        // 4 bytes, 1 trial after reference, 1 byte differs
        // disagreement_fraction = 1/4 = 0.25
        let outputs = vec![
            vec![0u8, 0, 0, 0],
            vec![0u8, 0, 0, 1], // 1 of 4 bytes differs
        ];
        let result = compare_determinism(&outputs);
        assert!(!result.is_deterministic());
        assert!((result.disagreement_fraction - 0.25).abs() < 0.001);
    }
}
