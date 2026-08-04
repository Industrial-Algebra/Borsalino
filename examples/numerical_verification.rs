// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Numerical verification example — runs the exact-match protocol
//! on the geometric product kernel using a real GPU.
//!
//! Run with:
//! ```sh
//! cargo run --features vulkan,verify --example numerical_verification
//! ```

use borsalino::numerical_check::{ExactMatchConfig, GeometricProductReference, verify_numerical};
use borsalino::{GpuBackend, init};

fn main() {
    let gpu = match init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU backend available: {e}");
            return;
        }
    };

    let gp_wgsl = borsalino::kernels::GEOMETRIC_PRODUCT;
    let pipeline = gpu.compile("gp", gp_wgsl).expect("compile GP kernel");

    let reference = GeometricProductReference { blades: 32 };
    let cfg = ExactMatchConfig {
        trials: 5,
        ..Default::default()
    };

    println!("Running numerical verification on geometric product...");
    let result =
        verify_numerical(&gpu, &pipeline, &reference, &cfg).expect("verification dispatch");

    println!("{result:#?}");

    if result.passed {
        println!("✅ PASS: geometric product kernel is numerically correct");
    } else {
        println!("❌ FAIL: geometric product kernel has numerical errors");
    }
}
