// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Numerical verification example — runs the exact-match protocol
//! on the geometric product kernel using a real GPU.
//!
//! Run with:
//! ```sh
//! cargo run --features vulkan,verify --example numerical_verification
//! ```
//!
//! **Exit codes are the contract**: 0 = verified, 1 = verification FAILED,
//! 1 = could not run (no GPU backend). A verification tool that cannot
//! fail is not a verification tool.

use std::process::ExitCode;

use borsalino::numerical_check::{ExactMatchConfig, GeometricProductReference, verify_numerical};
use borsalino::{GpuBackend, init};

fn main() -> ExitCode {
    let gpu = match init() {
        Ok(g) => g,
        Err(e) => {
            // "Could not verify" is not "verified" — fail loudly rather
            // than exit 0 on a machine where the check silently skipped.
            eprintln!("numerical_verification: no GPU backend available: {e}");
            return ExitCode::FAILURE;
        }
    };

    let gp_wgsl = borsalino::kernels::GEOMETRIC_PRODUCT;
    let pipeline = match gpu.compile("gp", gp_wgsl) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("numerical_verification: compile failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let reference = GeometricProductReference { blades: 32 };
    let cfg = ExactMatchConfig {
        trials: 5,
        ..Default::default()
    };

    println!("Running numerical verification on geometric product...");
    let result = match verify_numerical(&gpu, &pipeline, &reference, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("numerical_verification: dispatch failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("{result:#?}");

    if result.passed {
        println!("✅ PASS: geometric product kernel is numerically correct");
        ExitCode::SUCCESS
    } else {
        println!("❌ FAIL: geometric product kernel has numerical errors");
        ExitCode::FAILURE
    }
}
