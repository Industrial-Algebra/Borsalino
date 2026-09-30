// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! DeepReinforce exact-match numerical correctness protocol for linear GPU kernels.
//!
//! This module implements the protocol described in
//! [Towards a Reliable Kernel Correctness Check in Matrix
//! Multiplication](https://deep-reinforce.com/correctness_check.html).
//!
//! # How It Works
//!
//! Standard tolerance-based checks (`torch.allclose` with `atol`/`rtol`) are
//! unreliable for GPU kernels because floating-point associativity does not
//! hold: `(a+b)+c ≠ a+(b+c)` in FP16/BF16. Different GPU thread orderings
//! produce different accumulation sequences, so two correct kernels can produce
//! different outputs. No universal tolerance works across matrix sizes or
//! precision formats.
//!
//! The exact-match protocol sidesteps this by restricting kernel inputs to
//! **binary `{0, 1}`** values with a zero-biased distribution. This guarantees:
//!
//! 1. All partial sums are non-negative and monotonically non-decreasing
//! 2. Within the FP16 exact-integer range `[0, 2048]`, floating-point
//!    associativity holds exactly
//!
//! The kernel output is then compared against an FP32 CPU reference with
//! **bit-exact equality** at every position where the reference value is at or
//! below the threshold (2048 for FP16). Positions above the threshold are
//! ignored because they have lost exactness.
//!
//! # Applicability
//!
//! This protocol applies to **all linear kernels** (matmul, saxpy, scale,
//! add_one) and bilinear kernels where both operands are binary (geometric
//! product). It does **not** apply to non-linear operations (`log`, `exp`,
//! `tanh`) — those produce irrational outputs that cannot be checked with
//! exact match.
//!
//! # Example
//!
//! ```ignore
//! use borsalino::numerical_check::{self, ExactMatchConfig, GeometricProductReference};
//!
//! let gpu = borsalino::init()?;
//! let pipeline = gpu.compile("gp", borsalino::kernels::GEOMETRIC_PRODUCT)?;
//! let result = numerical_check::verify_numerical(
//!     &gpu,
//!     &pipeline,
//!     &GeometricProductReference { blades: 32 },
//!     &ExactMatchConfig::default(),
//! )?;
//! assert!(result.passed);
//! # Ok::<(), borsalino::GpuError>(())
//! ```

use crate::{ComputePipeline, GpuBackend, GpuBuffer, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for the exact-match numerical correctness protocol.
///
/// # Defaults
///
/// | Field | Default | Rationale |
/// |---|---|---|
/// | `threshold` | 2048.0 | Largest integer exactly representable in FP16 |
/// | `trials` | 16 | Sufficient trials to catch systematic bugs |
/// | `p_zero` | 0.7 | 70% zeros keeps accumulated sums below threshold |
#[derive(Debug, Clone)]
pub struct ExactMatchConfig {
    /// The exact-match ceiling. Positions where the FP32 CPU reference output
    /// exceeds this value are ignored (they have lost floating-point exactness).
    ///
    /// Default: 2048.0 (FP16 exact-integer ceiling).
    pub threshold: f32,

    /// Number of random binary-input trials to run.
    ///
    /// Default: 16. A kernel that passes all trials is very unlikely to be
    /// incorrect, though passing does not constitute a mathematical proof.
    pub trials: u32,

    /// Probability of sampling 0 vs 1 for each input element.
    ///
    /// Default: 0.7 (70% zeros). A higher zero probability keeps accumulated
    /// sums below the threshold for larger matrices. For small kernels, 0.5
    /// (uniform binary) is fine.
    pub p_zero: f32,
}

impl Default for ExactMatchConfig {
    fn default() -> Self {
        Self {
            threshold: 2048.0,
            trials: 16,
            p_zero: 0.7,
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────

/// Result of a numerical correctness check.
#[derive(Debug, Clone)]
pub struct NumericalCheckResult {
    /// Number of trials run.
    pub trials: u32,

    /// Total output positions compared across all trials (positions where
    /// reference ≤ threshold).
    pub positions_checked: usize,

    /// Positions that matched the reference exactly.
    pub positions_exact: usize,

    /// Maximum absolute difference observed at positions below the threshold.
    /// Should be 0.0 for a correct kernel.
    pub max_diff_below_threshold: f32,

    /// Whether the kernel passed the check.
    ///
    /// True only if every position at or below the threshold matched the
    /// reference exactly across all trials.
    pub passed: bool,
}

// ── Reference implementations ──────────────────────────────────────

/// A kernel's CPU reference implementation for the exact-match protocol.
///
/// Implement this for each kernel type. The reference must compute the same
/// mathematical function as the GPU kernel, in FP32 on CPU.
///
/// The reference is also the **kernel metadata** the driver needs: it names
/// every input buffer (in binding order, constant tables included), the
/// output buffer's element count, and the dispatch workgroup count. The
/// driver allocates the output buffer itself and chains it after the inputs
/// — output binding = `generate_inputs().len()`.
///
/// Inputs are **f32 arrays** with binary `{0.0, 1.0}` values (constant
/// tables excepted — the geometric product's sign table is ±1.0/0.0),
/// uploaded as `array<f32>` storage — matching the f32-reading kernels this
/// protocol targets. The reference computes the expected output from those
/// same inputs.
pub trait NumericalReference {
    /// Generate binary inputs for trial `trial_idx`, ready for GPU upload.
    ///
    /// Returns one `Vec<f32>` per **input binding**, in binding order —
    /// including constant tables the kernel reads (e.g. the geometric
    /// product's sign table at binding 0). The driver appends the output
    /// buffer after these.
    ///
    /// Use `rng` for reproducibility — the same seed should produce the same
    /// inputs across runs.
    fn generate_inputs(
        &self,
        trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>>;

    /// Output buffer length in **f32 elements**.
    ///
    /// The driver allocates exactly this many `f32`s as the output buffer
    /// and reads it back after dispatch.
    #[must_use]
    fn output_len(&self) -> usize;

    /// Workgroup count for the dispatch.
    ///
    /// Derived from the kernel this reference mirrors: the reference knows
    /// the kernel's `@workgroup_size` and output shape, so it computes
    /// `ceil(elements / workgroup_size)` per axis.
    #[must_use]
    fn workgroups(&self) -> (u32, u32, u32);

    /// Compute the FP32 CPU reference output from the same binary inputs.
    ///
    /// Returns a flat vector of f32 values — the expected kernel output,
    /// `output_len()` elements long.
    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32>;
}

/// Reference for the `add_one` kernel: `out[i] = in[i] + 1`.
#[derive(Debug, Clone)]
pub struct AddOneReference {
    /// Number of elements in the input/output buffer.
    pub len: usize,
}

/// Reference for the `scale` kernel: `out[i] = alpha * in[i]`.
#[derive(Debug, Clone)]
pub struct ScaleReference {
    /// Number of elements.
    pub len: usize,
    /// Scale factor. Use 1.0 for exact-match (keeps products exact).
    pub alpha: f32,
}

/// Reference for the `saxpy` kernel: `out[i] = alpha * x[i] + y[i]`.
#[derive(Debug, Clone)]
pub struct SaxpyReference {
    /// Number of elements.
    pub len: usize,
    /// Scale factor applied to x. Use 1.0 for exact-match.
    pub alpha: f32,
}

/// Reference for tiled matrix multiplication: `C = A @ B`.
#[derive(Debug, Clone)]
pub struct MatmulReference {
    /// Rows of A / C.
    pub m: usize,
    /// Inner dimension (columns of A, rows of B).
    pub k: usize,
    /// Columns of B / C.
    pub n: usize,
}

/// Reference for the IA geometric product kernel (5D Geometric Algebra).
///
/// Computes `c = a * b` where `a` and `b` are multivectors with `blades`
/// components. The sign table is computed independently from the algebraic
/// structure of Cl(n,0) — if the WGSL kernel's sign table has a bug, this
/// reference will catch it.
///
/// Bilinear with binary {0,1} inputs — exact-match protocol applies.
#[derive(Debug, Clone)]
pub struct GeometricProductReference {
    /// Number of blades (32 for 5D GA, 2^n for n-dimensional GA).
    pub blades: usize,
}

// ── Trait implementations ──────────────────────────────────────────

/// Workgroup size of the linear element-wise kernels these references
/// mirror (`add_one`, `scale`, `saxpy` — one thread per element, 64
/// threads per group, as in `examples/determinism_check.rs`).
const LINEAR_WORKGROUP_SIZE: u32 = 64;

impl NumericalReference for AddOneReference {
    fn generate_inputs(
        &self,
        _trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>> {
        vec![sample_binary_f32(self.len, cfg.p_zero, rng)]
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn workgroups(&self) -> (u32, u32, u32) {
        (
            self.len.div_ceil(LINEAR_WORKGROUP_SIZE as usize) as u32,
            1,
            1,
        )
    }

    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), 1);
        debug_assert_eq!(inputs[0].len(), self.len);
        inputs[0].iter().map(|&x| x + 1.0).collect()
    }
}

impl NumericalReference for ScaleReference {
    fn generate_inputs(
        &self,
        _trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>> {
        vec![sample_binary_f32(self.len, cfg.p_zero, rng)]
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn workgroups(&self) -> (u32, u32, u32) {
        (
            self.len.div_ceil(LINEAR_WORKGROUP_SIZE as usize) as u32,
            1,
            1,
        )
    }

    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), 1);
        debug_assert_eq!(inputs[0].len(), self.len);
        inputs[0].iter().map(|&x| x * self.alpha).collect()
    }
}

impl NumericalReference for SaxpyReference {
    fn generate_inputs(
        &self,
        _trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>> {
        vec![
            sample_binary_f32(self.len, cfg.p_zero, rng),
            sample_binary_f32(self.len, cfg.p_zero, rng),
        ]
    }

    fn output_len(&self) -> usize {
        self.len
    }

    fn workgroups(&self) -> (u32, u32, u32) {
        (
            self.len.div_ceil(LINEAR_WORKGROUP_SIZE as usize) as u32,
            1,
            1,
        )
    }

    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(inputs[0].len(), self.len);
        debug_assert_eq!(inputs[1].len(), self.len);
        inputs[0]
            .iter()
            .zip(inputs[1].iter())
            .map(|(&x, &y)| self.alpha * x + y)
            .collect()
    }
}

impl NumericalReference for MatmulReference {
    fn generate_inputs(
        &self,
        _trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>> {
        // Row-major: A is m×k, B is k×n.
        vec![
            sample_binary_f32(self.m * self.k, cfg.p_zero, rng),
            sample_binary_f32(self.k * self.n, cfg.p_zero, rng),
        ]
    }

    fn output_len(&self) -> usize {
        self.m * self.n
    }

    fn workgroups(&self) -> (u32, u32, u32) {
        // Mirrors the 16×16 tiled matmul kernel (`examples/tiled_matmul.rs`).
        (self.m.div_ceil(16) as u32, self.n.div_ceil(16) as u32, 1)
    }

    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(inputs[0].len(), self.m * self.k);
        debug_assert_eq!(inputs[1].len(), self.k * self.n);
        let a = &inputs[0];
        let b = &inputs[1];
        // Row-major matmul: C[i][j] = sum_k A[i][k] * B[k][j].
        let mut c = vec![0.0f32; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = 0.0f32;
                for k in 0..self.k {
                    sum += a[i * self.k + k] * b[k * self.n + j];
                }
                c[i * self.n + j] = sum;
            }
        }
        c
    }
}

// ── Geometric product sign table (independent computation) ─────────

/// Compute the output blade index for the geometric product of two basis
/// blades.
///
/// In Cl(n,0), the product of blade `i` and blade `j` is the blade
/// indexed by the symmetric difference (XOR) of their basis vector sets.
fn blade_product_output(i: usize, j: usize) -> usize {
    i ^ j
}

/// Compute the sign (+1 or -1) for the geometric product of two basis
/// blades.
///
/// The sign is `(-1)^s` where `s` is the number of basis vector swaps
/// needed to bring the common factors into cancellation position.
/// Specifically, `s` counts pairs `(a, b)` where bit `a` is set in `i`,
/// bit `b` is set in `j`, and `a > b`.
fn blade_product_sign(i: usize, j: usize) -> i8 {
    let mut swaps = 0u32;
    let mut remaining_i = i;
    while remaining_i != 0 {
        let a = remaining_i.trailing_zeros();
        remaining_i &= remaining_i - 1;
        // Count bits set in j at positions below a
        let mask = (1usize << a) - 1;
        swaps += (j & mask).count_ones();
    }
    if swaps % 2 == 0 { 1 } else { -1 }
}

/// Build the dense `blades³` sign table the geometric-product kernel reads
/// at binding 0: `table[i][j][k]` is the sign of `a[i]·b[j]`'s contribution
/// to output blade `k` — nonzero only at `k = i ^ j`.
///
/// Built from the same independently-tested bit arithmetic the reference
/// uses ([`blade_product_output`], [`blade_product_sign`]).
pub(crate) fn ga_sign_table(blades: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; blades * blades * blades];
    for i in 0..blades {
        for j in 0..blades {
            let flat = i * blades * blades + j * blades + (i ^ j);
            table[flat] = blade_product_sign(i, j) as f32;
        }
    }
    table
}

impl NumericalReference for GeometricProductReference {
    fn generate_inputs(
        &self,
        _trial_idx: u32,
        cfg: &ExactMatchConfig,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<Vec<f32>> {
        let a = sample_binary_f32(self.blades, cfg.p_zero, rng);
        let b = sample_binary_f32(self.blades, cfg.p_zero, rng);
        // Binding order per `kernels::GEOMETRIC_PRODUCT`:
        // 0 = sign table, 1 = a, 2 = b. The driver appends output at 3.
        vec![ga_sign_table(self.blades), a, b]
    }

    fn output_len(&self) -> usize {
        self.blades
    }

    fn workgroups(&self) -> (u32, u32, u32) {
        // Mirrors `kernels::GEOMETRIC_PRODUCT` — @workgroup_size(32),
        // one thread per output blade.
        (self.blades.div_ceil(32) as u32, 1, 1)
    }

    fn compute_reference(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), 3, "sign table + two multivectors");
        debug_assert_eq!(inputs[0].len(), self.blades * self.blades * self.blades);
        debug_assert_eq!(inputs[1].len(), self.blades);
        debug_assert_eq!(inputs[2].len(), self.blades);
        // The reference is purely algebraic — it does not consult the sign
        // table buffer, so a bug in table construction cannot hide behind
        // the same bug in the reference.
        let a = &inputs[1];
        let b = &inputs[2];
        let mut c = vec![0.0f32; self.blades];
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                if bj == 0.0 {
                    continue;
                }
                let out = blade_product_output(i, j);
                let sign = blade_product_sign(i, j);
                c[out] += sign as f32 * ai * bj;
            }
        }
        c
    }
}

/// Sample `n` binary `{0.0, 1.0}` f32s. Each element is 0 with probability
/// `p_zero`, 1 otherwise — uploaded as `array<f32>` storage, matching the
/// f32-reading kernels this protocol targets.
fn sample_binary_f32(n: usize, p_zero: f32, rng: &mut dyn rand::RngCore) -> Vec<f32> {
    use rand::Rng;
    (0..n)
        .map(|_| {
            if rng.gen_bool(p_zero as f64) {
                0.0
            } else {
                1.0
            }
        })
        .collect()
}

// ── Comparison logic ───────────────────────────────────────────────

/// Compare GPU output against FP32 reference with threshold gating.
///
/// This is the core of the exact-match protocol. For each position where the
/// reference value is at or below `threshold`, the GPU output must match
/// exactly. Positions above the threshold are ignored.
///
/// This function is pure and testable without a GPU.
pub fn compare_outputs(
    gpu_output: &[f32],
    reference: &[f32],
    threshold: f32,
) -> NumericalCheckResult {
    let mut positions_checked = 0usize;
    let mut positions_exact = 0usize;
    let mut max_diff_below_threshold = 0.0f32;

    for (gpu_val, ref_val) in gpu_output.iter().zip(reference.iter()) {
        if *ref_val <= threshold {
            positions_checked += 1;
            let diff = (gpu_val - ref_val).abs();
            if diff == 0.0 {
                positions_exact += 1;
            }
            if diff > max_diff_below_threshold {
                max_diff_below_threshold = diff;
            }
        }
    }

    let passed = positions_checked > 0 && positions_exact == positions_checked;

    NumericalCheckResult {
        trials: 1,
        positions_checked,
        positions_exact,
        max_diff_below_threshold,
        passed,
    }
}

// ── Full protocol (GPU-dependent) ──────────────────────────────────

/// Run the exact-match protocol against a GPU kernel.
///
/// For each trial: generates binary inputs, computes the FP32 CPU reference,
/// dispatches the kernel on GPU, reads back the output, and compares with
/// bit-exact equality at positions where the reference is at or below the
/// threshold.
///
/// Returns a combined [`NumericalCheckResult`] across all trials. The check
/// passes only if every position at or below the threshold matched exactly
/// across every trial.
///
/// # Errors
///
/// Returns an error if GPU dispatch or buffer read-back fails.
pub fn verify_numerical<G: GpuBackend, R: NumericalReference>(
    gpu: &G,
    pipeline: &ComputePipeline,
    reference: &R,
    cfg: &ExactMatchConfig,
) -> Result<NumericalCheckResult> {
    let mut combined = NumericalCheckResult {
        trials: 0,
        positions_checked: 0,
        positions_exact: 0,
        max_diff_below_threshold: 0.0,
        passed: true,
    };

    let mut rng = rand::thread_rng();

    for trial in 0..cfg.trials {
        let input_data = reference.generate_inputs(trial, cfg, &mut rng);
        let ref_output = reference.compute_reference(&input_data);

        // Upload binary inputs to GPU buffers (all input bindings, in
        // order — constant tables included) as f32 storage.
        let input_buffers: Vec<GpuBuffer> = input_data
            .iter()
            .map(|data| gpu.create_buffer(data))
            .collect::<Result<Vec<_>>>()?;

        // Allocate the OUTPUT buffer — binding index = number of inputs.
        // The kernel writes here; nothing else in this loop touches it.
        let output_buffer = gpu.create_buffer_uninit::<f32>(reference.output_len())?;

        let buffer_refs: Vec<&GpuBuffer> = input_buffers
            .iter()
            .chain(std::iter::once(&output_buffer))
            .collect();

        // Dispatch with the reference's kernel metadata (workgroup sizing
        // is the reference's to compute — it mirrors the kernel).
        gpu.dispatch(pipeline, &buffer_refs, reference.workgroups())?;

        // Read back the output buffer and compare with threshold gating.
        let gpu_output: Vec<f32> = gpu.read_buffer(&output_buffer)?;
        let trial_result = compare_outputs(&gpu_output, &ref_output, cfg.threshold);

        combined.trials += 1;
        combined.positions_checked += trial_result.positions_checked;
        combined.positions_exact += trial_result.positions_exact;
        if trial_result.max_diff_below_threshold > combined.max_diff_below_threshold {
            combined.max_diff_below_threshold = trial_result.max_diff_below_threshold;
        }
        if !trial_result.passed {
            combined.passed = false;
        }
    }

    Ok(combined)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_fp16_threshold() {
        let cfg = ExactMatchConfig::default();
        assert_eq!(cfg.threshold, 2048.0);
        assert!(cfg.trials >= 1);
        assert!(cfg.p_zero > 0.0 && cfg.p_zero < 1.0);
    }

    // ── compare_outputs: correct kernel ──────────────────────────

    #[test]
    fn compare_outputs_exact_match_passes() {
        let gpu_output = vec![1.0, 2.0, 3.0, 4.0];
        let reference = vec![1.0, 2.0, 3.0, 4.0];

        let result = compare_outputs(&gpu_output, &reference, 2048.0);

        assert!(result.passed);
        assert_eq!(result.positions_checked, 4);
        assert_eq!(result.positions_exact, 4);
        assert_eq!(result.max_diff_below_threshold, 0.0);
    }

    // ── compare_outputs: incorrect kernel detected ───────────────

    #[test]
    fn compare_outputs_mismatch_detected() {
        let gpu_output = vec![1.0, 2.0, 3.0, 5.0]; // last element wrong
        let reference = vec![1.0, 2.0, 3.0, 4.0];

        let result = compare_outputs(&gpu_output, &reference, 2048.0);

        assert!(!result.passed);
        assert_eq!(result.positions_checked, 4);
        assert_eq!(result.positions_exact, 3);
        assert_eq!(result.max_diff_below_threshold, 1.0);
    }

    // ── compare_outputs: threshold gating ────────────────────────

    #[test]
    fn compare_outputs_ignores_positions_above_threshold() {
        // GPU output differs from reference at a position above threshold.
        // That position should be ignored, not counted as a failure.
        let gpu_output = vec![1.0, 2.0, 3000.0]; // 3000 > 2048 threshold
        let reference = vec![1.0, 2.0, 2049.0]; // 2049 > 2048, ignored

        let result = compare_outputs(&gpu_output, &reference, 2048.0);

        assert!(result.passed);
        assert_eq!(result.positions_checked, 2); // only positions 0 and 1
        assert_eq!(result.positions_exact, 2);
    }

    #[test]
    fn compare_outputs_threshold_boundary_inclusive() {
        // Reference value exactly at threshold should be checked.
        let gpu_output = vec![2048.0];
        let reference = vec![2048.0];

        let result = compare_outputs(&gpu_output, &reference, 2048.0);

        assert!(result.passed);
        assert_eq!(result.positions_checked, 1);
    }

    // ── compare_outputs: edge cases ──────────────────────────────

    #[test]
    fn compare_outputs_all_above_threshold_reports_no_evidence() {
        let gpu_output = vec![5000.0, 6000.0];
        let reference = vec![5000.0, 6000.0];

        let result = compare_outputs(&gpu_output, &reference, 2048.0);

        // No positions checked means no verification happened — not a pass.
        // The caller should adjust inputs to get positions below threshold.
        assert!(!result.passed, "vacuous pass hides lack of evidence");
        assert_eq!(result.positions_checked, 0);
    }

    // ── Reference implementations ────────────────────────────────

    #[test]
    fn add_one_reference_computes_correctly() {
        let reference = AddOneReference { len: 4 };
        // Binary inputs: [0, 1, 0, 1] as bytes
        let inputs = vec![vec![0.0f32, 1.0, 0.0, 1.0]];

        let output = reference.compute_reference(&inputs);

        // add_one: 0+1=1, 1+1=2, 0+1=1, 1+1=2
        assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn matmul_reference_computes_2x2_correctly() {
        let reference = MatmulReference { m: 2, k: 2, n: 2 };
        // A = [[1, 0], [0, 1]], B = [[1, 1], [0, 0]] (binary, column-major or row-major?)
        // We use row-major flat.
        // A @ B = [[1*1+0*0, 1*1+0*0], [0*1+1*0, 0*1+1*0]] = [[1, 1], [0, 0]]
        let inputs = vec![
            vec![1.0f32, 0.0, 0.0, 1.0], // A row-major
            vec![1.0f32, 1.0, 0.0, 0.0], // B row-major
        ];

        let output = reference.compute_reference(&inputs);

        assert_eq!(output.len(), 4);
        assert_eq!(output, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn saxpy_reference_computes_correctly() {
        let reference = SaxpyReference { len: 3, alpha: 1.0 };
        // x = [1, 0, 1], y = [0, 1, 0]
        // saxpy: alpha*x + y = [1, 1, 1]
        let inputs = vec![vec![1.0f32, 0.0, 1.0], vec![0.0f32, 1.0, 0.0]];

        let output = reference.compute_reference(&inputs);

        assert_eq!(output, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn add_one_generates_binary_inputs() {
        let reference = AddOneReference { len: 100 };
        let cfg = ExactMatchConfig::default();
        let mut rng = rand::thread_rng();

        let inputs = reference.generate_inputs(0, &cfg, &mut rng);

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].len(), 100);
        // All values must be 0.0 or 1.0.
        for &x in &inputs[0] {
            assert!(x == 0.0 || x == 1.0, "non-binary value: {x}");
        }
    }

    #[test]
    fn references_expose_output_len_and_workgroups() {
        // Element-wise: output_len == len; workgroups = ceil(len/64).
        let add_one = AddOneReference { len: 200 };
        assert_eq!(add_one.output_len(), 200);
        assert_eq!(add_one.workgroups(), (4, 1, 1)); // ceil(200/64) = 4

        let saxpy = SaxpyReference {
            len: 64,
            alpha: 1.0,
        };
        assert_eq!(saxpy.output_len(), 64);
        assert_eq!(saxpy.workgroups(), (1, 1, 1));

        // Matmul: output m*n; 16×16 tiles.
        let mm = MatmulReference { m: 33, k: 4, n: 17 };
        assert_eq!(mm.output_len(), 33 * 17);
        assert_eq!(mm.workgroups(), (3, 2, 1)); // ceil(33/16), ceil(17/16)

        // Geometric product: output blades; @workgroup_size(32).
        let gp = GeometricProductReference { blades: 32 };
        assert_eq!(gp.output_len(), 32);
        assert_eq!(gp.workgroups(), (1, 1, 1));
    }

    // ── Geometric product reference ──────────────────────────────

    #[test]
    fn gp_scalar_times_scalar_is_scalar() {
        // 1D GA (2 blades): blade 0 = scalar, blade 1 = vector e0
        // e0 * e0 = 1 (scalar), 1 * 1 = 1, 1 * e0 = e0, e0 * 1 = e0
        let reference = GeometricProductReference { blades: 2 };
        let table = ga_sign_table(2);
        // a = [1, 0] (scalar 1), b = [1, 0] (scalar 1)
        let inputs = vec![table, vec![1.0f32, 0.0], vec![1.0f32, 0.0]];
        let output = reference.compute_reference(&inputs);
        // 1 * 1 = 1 → c[0] = 1
        assert_eq!(output, vec![1.0, 0.0]);
    }

    #[test]
    fn gp_vector_times_vector_is_scalar() {
        // e0 * e0 = 1 (blade 0), sign = +1 (Cl(1,0): e0² = +1)
        let reference = GeometricProductReference { blades: 2 };
        let table = ga_sign_table(2);
        // a = [0, 1] (vector e0), b = [0, 1] (vector e0)
        let inputs = vec![table, vec![0.0f32, 1.0], vec![0.0f32, 1.0]];
        let output = reference.compute_reference(&inputs);
        // e0 * e0: output blade = 1 XOR 1 = 0 (scalar)
        // sign: no inversion pairs → +1
        assert_eq!(output[0], 1.0); // scalar component
    }

    #[test]
    fn gp_generates_table_and_binary_multivectors() {
        let reference = GeometricProductReference { blades: 32 };
        let cfg = ExactMatchConfig::default();
        let mut rng = rand::thread_rng();

        let inputs = reference.generate_inputs(0, &cfg, &mut rng);

        // Binding order: sign table (0), a (1), b (2) — the kernel's contract.
        assert_eq!(inputs.len(), 3);
        // Dense 32×32×32 f32 table.
        assert_eq!(inputs[0].len(), 32 * 32 * 32);
        assert_eq!(inputs[1].len(), 32);
        assert_eq!(inputs[2].len(), 32);
        for buf in &inputs[1..] {
            for &x in buf {
                assert!(x == 0.0 || x == 1.0, "non-binary value: {x}");
            }
        }
    }

    #[test]
    fn gp_sign_table_matches_kernel_indexing() {
        // The table must be nonzero only at k = i ^ j, with the sign of
        // blade_product(i, j) — exactly what kernels::GEOMETRIC_PRODUCT reads.
        let blades = 32;
        let table = ga_sign_table(blades);
        for i in 0..blades {
            for j in 0..blades {
                for k in 0..blades {
                    let value = table[i * blades * blades + j * blades + k];
                    if k == i ^ j {
                        assert_eq!(
                            value,
                            blade_product_sign(i, j) as f32,
                            "sign mismatch at ({i},{j},{k})"
                        );
                    } else {
                        assert_eq!(value, 0.0, "stray entry at ({i},{j},{k})");
                    }
                }
            }
        }
    }

    #[test]
    fn gp_blade_product_output_is_xor() {
        // Output blade = XOR of input blades
        assert_eq!(blade_product_output(0b101, 0b011), 0b110);
        assert_eq!(blade_product_output(0, 0), 0);
        assert_eq!(blade_product_output(0b11111, 0b11111), 0);
    }

    #[test]
    fn gp_blade_product_sign_antisymmetric_for_vectors() {
        // e1 * e2 = +e12 (blade 0b110)
        // e2 * e1 = -e12 (blade 0b110)
        // i=2 (bit 1), j=4 (bit 2): a=1, b=2, a>b → 1 swap → sign +1
        let s12 = blade_product_sign(2, 4);
        // i=4 (bit 2), j=2 (bit 1): i=4 → bit 2, j=2 → bit 1
        // mask = (1 << 2) - 1 = 0b11, j & mask = 0b10 & 0b11 = 0b10 → 1 one → 1 swap
        let s21 = blade_product_sign(4, 2);
        // They should be opposite signs
        assert_eq!(s12, -s21);
        assert_eq!(s12 * s21, -1);
    }

    // ═════════════════════════════════════════════════════════════
    // Recording fake backend — driver wiring testable without hardware.
    // (Pattern from baedeker_core::runtime::verify; the 2026-09-29
    // research dive found the GPU driver had never been exercised.)
    // ═════════════════════════════════════════════════════════════

    use std::cell::RefCell;

    /// One recorded dispatch: the bound buffers' addresses + workgroups.
    type DispatchRecord = (Vec<usize>, (u32, u32, u32));

    /// A fake `GpuBackend` that simulates a linear `out[i] = in[i] + c`
    /// kernel (binding 0 in, last binding out) and records every dispatch
    /// and readback for structural assertions.
    struct RecordingBackend {
        /// One entry per dispatch.
        dispatches: RefCell<Vec<DispatchRecord>>,
        /// Addresses of buffers passed to `read_buffer`.
        reads: RefCell<Vec<usize>>,
        /// The `c` in `out = in + c`. `1` simulates a correct kernel,
        /// anything else a broken one (mutation testing).
        constant: f32,
    }

    impl RecordingBackend {
        fn new(constant: f32) -> Self {
            Self {
                dispatches: RefCell::new(Vec::new()),
                reads: RefCell::new(Vec::new()),
                constant,
            }
        }

        fn buffer_bytes(buffer: &crate::GpuBuffer) -> &Vec<u8> {
            // SAFETY: every buffer this fake creates stores a heap `Vec<u8>`
            // behind `raw`; it is never freed while the handle is alive.
            unsafe { &*(buffer.raw as *const Vec<u8>) }
        }
    }

    fn drop_storage(raw: *mut std::ffi::c_void) {
        // SAFETY: paired with `Box::into_raw` in the constructors below.
        drop(unsafe { Box::from_raw(raw as *mut Vec<u8>) });
    }

    fn storage_ptr(raw: *mut std::ffi::c_void) -> *const std::ffi::c_void {
        // SAFETY: see `buffer_bytes`.
        unsafe { (*(raw as *const Vec<u8>)).as_ptr() as *const std::ffi::c_void }
    }

    impl crate::GpuBackend for RecordingBackend {
        fn init() -> crate::Result<Self> {
            Ok(Self::new(1.0))
        }

        fn compile(
            &self,
            _entry_point: &str,
            _wgsl_source: &str,
        ) -> crate::Result<crate::ComputePipeline> {
            Ok(crate::ComputePipeline {
                raw: std::ptr::null_mut(),
                drop_fn: |_| {},
            })
        }

        fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> crate::Result<crate::GpuBuffer> {
            let storage: Vec<u8> = bytemuck::cast_slice::<T, u8>(data).to_vec();
            Ok(crate::GpuBuffer {
                raw: Box::into_raw(Box::new(storage)) as *mut std::ffi::c_void,
                len: data.len(),
                element_size: std::mem::size_of::<T>(),
                drop_fn: drop_storage,
                contents_fn: storage_ptr,
            })
        }

        fn create_buffer_uninit<T: bytemuck::Pod>(
            &self,
            len: usize,
        ) -> crate::Result<crate::GpuBuffer> {
            let storage = vec![0u8; len * std::mem::size_of::<T>()];
            Ok(crate::GpuBuffer {
                raw: Box::into_raw(Box::new(storage)) as *mut std::ffi::c_void,
                len,
                element_size: std::mem::size_of::<T>(),
                drop_fn: drop_storage,
                contents_fn: storage_ptr,
            })
        }

        fn dispatch(
            &self,
            _pipeline: &crate::ComputePipeline,
            buffers: &[&crate::GpuBuffer],
            workgroups: (u32, u32, u32),
        ) -> crate::Result<()> {
            self.dispatches
                .borrow_mut()
                .push((buffers.iter().map(|b| b.raw as usize).collect(), workgroups));

            // Simulate the linear kernel: out[i] = in[i] + constant, reading
            // binding 0 (f32 storage) and writing the last binding. With
            // fewer than two bindings there is no output buffer to write —
            // mirror what a real backend does and leave storage untouched.
            if buffers.len() >= 2 {
                let bytes = Self::buffer_bytes(buffers[0]).clone();
                let input: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let computed: Vec<f32> = input.iter().map(|&x| x + self.constant).collect();
                // SAFETY: the output handle was created by this fake; its
                // storage is a heap `Vec<u8>` we own.
                unsafe {
                    *(buffers[buffers.len() - 1].raw as *mut Vec<u8>) =
                        bytemuck::cast_slice(&computed).to_vec();
                }
            }
            Ok(())
        }

        fn dispatch_ex(
            &self,
            pipeline: &crate::ComputePipeline,
            buffers: &[&crate::GpuBuffer],
            workgroups: (u32, u32, u32),
            _threads_per_group: (u32, u32, u32),
        ) -> crate::Result<()> {
            self.dispatch(pipeline, buffers, workgroups)
        }

        fn read_buffer<T: bytemuck::Pod>(
            &self,
            buffer: &crate::GpuBuffer,
        ) -> crate::Result<Vec<T>> {
            self.reads.borrow_mut().push(buffer.raw as usize);
            let bytes = Self::buffer_bytes(buffer);
            let elems = bytes.len() / std::mem::size_of::<T>();
            let mut out = Vec::with_capacity(elems);
            for i in 0..elems {
                let chunk =
                    &bytes[i * std::mem::size_of::<T>()..(i + 1) * std::mem::size_of::<T>()];
                out.push(*bytemuck::from_bytes(chunk));
            }
            Ok(out)
        }

        fn timestamp(&self) -> crate::Result<u64> {
            Ok(0)
        }
    }

    #[test]
    fn driver_chains_output_buffer_and_reads_it_back() {
        let gpu = RecordingBackend::new(1.0);
        let pipeline = gpu.compile("main", "unused").unwrap();
        let reference = AddOneReference { len: 200 };
        let cfg = ExactMatchConfig {
            trials: 1,
            ..Default::default()
        };

        let result =
            verify_numerical(&gpu, &pipeline, &reference, &cfg).expect("verification dispatch");

        // Exactly one dispatch, with inputs AND a chained output buffer.
        let dispatches = gpu.dispatches.borrow();
        assert_eq!(dispatches.len(), 1, "one dispatch per trial");
        let (addrs, workgroups) = &dispatches[0];
        assert_eq!(addrs.len(), 2, "driver must chain input + output buffer");
        let input_addr = addrs[0];
        let output_addr = addrs[1];
        assert_ne!(input_addr, output_addr, "output must be its own buffer");

        // Workgroups come from the reference (ceil(200/64) = 4), not a
        // hardcoded (1,1,1).
        assert_eq!(*workgroups, (4, 1, 1));

        // The driver must read back the OUTPUT buffer, not an input.
        let reads = gpu.reads.borrow();
        assert_eq!(reads.len(), 1);
        assert_eq!(
            reads[0], output_addr,
            "driver read an input, not the output"
        );

        // And with a correct simulated kernel, the verdict is PASS.
        assert!(result.passed, "correct kernel must pass: {result:#?}");
        assert!(result.positions_checked > 0);
    }

    #[test]
    fn driver_detects_broken_kernel() {
        // Mutation: the "kernel" computes in + 2 instead of in + 1.
        // A driver that cannot produce FAIL is not a verifier.
        let gpu = RecordingBackend::new(2.0);
        let pipeline = gpu.compile("main", "unused").unwrap();
        let reference = AddOneReference { len: 64 };
        let cfg = ExactMatchConfig {
            trials: 1,
            ..Default::default()
        };

        let result =
            verify_numerical(&gpu, &pipeline, &reference, &cfg).expect("verification dispatch");

        assert!(!result.passed, "broken kernel must fail: {result:#?}");
        assert!(
            result.positions_checked > 0,
            "positions must have been compared"
        );
    }

    #[test]
    fn driver_uses_uninit_output_allocation() {
        // The output buffer must be allocated via create_buffer_uninit,
        // not uploaded from host data (it is written by the kernel).
        // Distinguish by element_size bookkeeping: create_buffer::<f32>
        // records len=elems element_size=4 — same as uninit — so instead
        // assert the write path: a correct dispatch leaves the input
        // buffer's bytes untouched (binary {0,1}), while the output holds
        // the computed floats. This is covered structurally by
        // driver_chains_output_buffer_and_reads_it_back (input and output
        // are distinct buffers) plus the value comparison here.
        let gpu = RecordingBackend::new(1.0);
        let pipeline = gpu.compile("main", "unused").unwrap();
        let reference = AddOneReference { len: 32 };
        let cfg = ExactMatchConfig {
            trials: 1,
            ..Default::default()
        };

        let result = verify_numerical(&gpu, &pipeline, &reference, &cfg).unwrap();
        assert!(result.passed);
    }

    // ═════════════════════════════════════════════════════════════
    // Hardware tests — run with `cargo test --features vulkan -- --ignored`
    // (the CI GPU job's invocation; these were the first #[ignore]d GPU
    // tests in the repo — before this PR that CI step selected nothing).
    // ═════════════════════════════════════════════════════════════

    #[cfg(feature = "vulkan")]
    mod hardware {
        use super::super::*;
        use crate::{GpuBackend, init, kernels};
        use serial_test::serial;

        #[test]
        #[ignore = "requires a Vulkan GPU"]
        #[serial]
        fn gp_kernel_verifies_on_hardware() {
            let gpu = init().expect("vulkan device");
            let pipeline = gpu
                .compile("gp", kernels::GEOMETRIC_PRODUCT)
                .expect("compile GP kernel");

            let reference = GeometricProductReference { blades: 32 };
            let cfg = ExactMatchConfig {
                trials: 5,
                ..Default::default()
            };

            let result = verify_numerical(&gpu, &pipeline, &reference, &cfg).expect("run protocol");

            assert!(
                result.passed,
                "geometric product must verify on hardware: {result:#?}"
            );
            assert!(result.positions_checked > 0);
        }

        #[test]
        #[ignore = "requires a Vulkan GPU"]
        #[serial]
        fn gp_mutation_fails_on_hardware() {
            // A genuinely wrong kernel (sign flipped) must FAIL the
            // protocol. If it cannot fail, the protocol verifies nothing.
            let gpu = init().expect("vulkan device");
            let broken = kernels::GEOMETRIC_PRODUCT
                .replace("sum += sign * a[i] * b[j];", "sum -= sign * a[i] * b[j];");
            assert_ne!(broken, kernels::GEOMETRIC_PRODUCT, "mutation applied");
            let pipeline = gpu.compile("gp", &broken).expect("compile broken kernel");

            let reference = GeometricProductReference { blades: 32 };
            let cfg = ExactMatchConfig {
                trials: 5,
                ..Default::default()
            };

            let result = verify_numerical(&gpu, &pipeline, &reference, &cfg).expect("run protocol");

            assert!(!result.passed, "sign-flipped kernel must FAIL: {result:#?}");
        }
    }
}
