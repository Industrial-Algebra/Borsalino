// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Borsalino Industrial Algebra benchmark — geometric product of 32-element
//! multivectors (5D GA, 32 basis blades).
//!
//! Demonstrates Borsalino as a compute backend for the Amari ecosystem.
//! The 32-blade geometric product is the fundamental operation in 5D
//! geometric algebra — all other products (wedge, inner, contraction)
//! are derived from it.
//!
//! # Run
//!
//! ```sh
//! cargo run --example ia_geometric_product --features vulkan --release
//! cargo run --example ia_geometric_product --features metal --release
//! ```

use std::time::Instant;

use borsalino::GpuBackend;

const BLADES: u32 = 32;

/// Geometric product kernel for 32-blade multivectors.
/// Each thread computes one output blade as: out[k] = Σ sign(i,j,k) · a[i] · b[j]
const KERNEL_GEOMETRIC_PRODUCT: &str = r#"
@group(0) @binding(0) var<storage, read> sign_table: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(32)
fn gp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    var sum = 0.0;
    for (var i = 0u; i < 32u; i++) {
        for (var j = 0u; j < 32u; j++) {
            let sign = sign_table[i * 32u * 32u + j * 32u + k];
            sum += sign * a[i] * b[j];
        }
    }
    out[k] = sum;
}
"#;

/// Kernel for batched geometric products: N multivectors in one dispatch.
const KERNEL_GEOMETRIC_PRODUCT_BATCHED: &str = r#"
@group(0) @binding(0) var<storage, read> sign_table: array<f32>;
@group(0) @binding(1) var<storage, read> a_batch: array<f32>;
@group(0) @binding(2) var<storage, read> b_batch: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_batch: array<f32>;

@compute @workgroup_size(256)
fn gp_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let mv = idx / 32u;      // which multivector in the batch
    let blade = idx % 32u;   // which output blade
    let base = mv * 32u;

    var sum = 0.0;
    for (var i = 0u; i < 32u; i++) {
        for (var j = 0u; j < 32u; j++) {
            let sign = sign_table[i * 32u * 32u + j * 32u + blade];
            sum += sign * a_batch[base + i] * b_batch[base + j];
        }
    }
    out_batch[base + blade] = sum;
}
"#;

/// Generate the 5D GA multiplication table: sign_table[i][j][k] = { -1, 0, +1 }
/// indicating the contribution of a[i]*b[j] to output blade k.
fn build_sign_table() -> Vec<f32> {
    let mut table = vec![0.0f32; (BLADES * BLADES * BLADES) as usize];
    for i in 0..BLADES {
        for j in 0..BLADES {
            let product_blade = i ^ j; // XOR = blade multiplication in basis
            let sign = blade_sign(i, j);
            table[(i * BLADES * BLADES + j * BLADES + product_blade) as usize] = sign;
        }
    }
    table
}

/// Compute the sign of the geometric product of basis blades i and j.
/// For 5D GA, sign = (-1)^(popcount(i&j) + grade_count), simplified as:
/// sign = parity of swap count to bring i's set bits past j's set bits.
fn blade_sign(a: u32, b: u32) -> f32 {
    let mut sign = 1.0f32;
    let mut b_shifted = b;
    for bit in 0..5 {
        if (a >> bit) & 1 != 0 {
            let count = (b_shifted & ((1 << bit) - 1)).count_ones();
            if count % 2 != 0 {
                sign = -sign;
            }
        }
        b_shifted = (b_shifted & !(1 << bit)) | ((b_shifted >> bit & 1) << bit);
    }
    sign
}

/// CPU reference: geometric product of two 32-blade multivectors.
fn cpu_geometric_product(a: &[f32; 32], b: &[f32; 32], sign_table: &[f32]) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    for k in 0..BLADES {
        let mut sum = 0.0;
        for i in 0..BLADES {
            for j in 0..BLADES {
                let sign = sign_table[(i * BLADES * BLADES + j * BLADES + k) as usize];
                sum += sign * a[i as usize] * b[j as usize];
            }
        }
        out[k as usize] = sum;
    }
    out
}

fn main() -> Result<(), borsalino::GpuError> {
    let gpu = borsalino::init()?;
    let batch_count: u32 = 1000;

    println!("Borsalino — Industrial Algebra Kernel Benchmark");
    println!("  operation: geometric product");
    println!("  blades: {BLADES} (5D GA)");
    println!("  batch: {batch_count} dispatches");
    println!();

    // ── Generate sign table + test data ─────────────────────────

    let sign_table = build_sign_table();
    let a: [f32; 32] = std::array::from_fn(|i| (i as f32 + 1.0) * 0.1);
    let b: [f32; 32] = std::array::from_fn(|i| ((i * 3 + 7) % 32) as f32 * 0.1);

    // ── CPU reference ───────────────────────────────────────────

    let cpu_start = Instant::now();
    let expected = cpu_geometric_product(&a, &b, &sign_table);
    let cpu_time = cpu_start.elapsed();
    println!("--- CPU Reference ---");
    println!("  time: {:>8.1} µs", cpu_time.as_micros() as f64);

    // ── GPU compile + upload ────────────────────────────────────

    let compile_start = Instant::now();
    let pipeline = gpu.compile("gp", KERNEL_GEOMETRIC_PRODUCT)?;
    println!("\n--- GPU Setup ---");
    println!(
        "  compile: {:>8.1} ms",
        compile_start.elapsed().as_secs_f64() * 1e3
    );

    let buf_sign = gpu.create_buffer(&sign_table)?;
    let buf_a = gpu.create_buffer(&a)?;
    let buf_b = gpu.create_buffer(&b)?;
    let buf_out = gpu.create_buffer_uninit::<f32>(BLADES as usize)?;

    // ── GPU batch dispatch ──────────────────────────────────────

    println!("\n--- GPU Performance ---");
    let gpu_start = Instant::now();
    for _ in 0..batch_count {
        gpu.dispatch(&pipeline, &[&buf_sign, &buf_a, &buf_b, &buf_out], (1, 1, 1))?;
    }
    let gpu_time = gpu_start.elapsed();

    let result: Vec<f32> = gpu.read_buffer(&buf_out)?;
    let gpu_per = gpu_time.as_micros() as f64 / batch_count as f64;
    println!(
        "  total ({batch_count} dispatches): {:>8.1} ms",
        gpu_time.as_secs_f64() * 1e3
    );
    println!("  per-dispatch: {:>8.1} µs", gpu_per);

    // ── Verify ──────────────────────────────────────────────────

    let mut max_err = 0.0f32;
    for i in 0..BLADES as usize {
        let err = (result[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    print!("\n--- Verification ---\n  ");
    if max_err < 1e-4 {
        println!("✅ all {BLADES} blades correct (max err: {max_err:.6})");
    } else {
        println!("❌ max error: {max_err:.6}");
    }

    // ── Summary ─────────────────────────────────────────────────

    let speedup = cpu_time.as_secs_f64() / (gpu_time.as_secs_f64() / batch_count as f64);
    println!("\n--- Summary (single dispatch) ---");
    println!("  CPU: {:>8.1} µs", cpu_time.as_micros() as f64);
    println!("  GPU: {:>8.1} µs/dispatch", gpu_per);
    println!("  speedup: {speedup:.1}×");
    println!("  note: GPU overhead dominates for single multivectors — batch them!");

    // ── Batched variant (many multivectors in one dispatch) ───

    let batch_size: u32 = 4096;
    let flat_size = (batch_size * BLADES) as usize;
    let a_batch: Vec<f32> = (0..flat_size).map(|i| (i % 32) as f32 * 0.1).collect();
    let b_batch: Vec<f32> = (0..flat_size)
        .map(|i| ((i * 3 + 7) % 32) as f32 * 0.1)
        .collect();

    let pipeline_batched = gpu.compile("gp_batched", KERNEL_GEOMETRIC_PRODUCT_BATCHED)?;
    let buf_a_batch = gpu.create_buffer(&a_batch)?;
    let buf_b_batch = gpu.create_buffer(&b_batch)?;
    let buf_out_batch = gpu.create_buffer_uninit::<f32>(flat_size)?;
    let wgs = (batch_size * BLADES).div_ceil(256);

    println!("\n--- Batched GP ({batch_size} multivectors, one dispatch) ---");

    // CPU reference for batched
    let cpu_batch_start = Instant::now();
    for mv in 0..batch_size as usize {
        let base = mv * BLADES as usize;
        let a_mv: [f32; 32] = std::array::from_fn(|i| a_batch[base + i]);
        let b_mv: [f32; 32] = std::array::from_fn(|i| b_batch[base + i]);
        let _ = cpu_geometric_product(&a_mv, &b_mv, &sign_table);
    }
    let cpu_batch_time = cpu_batch_start.elapsed();

    let gpu_batch_start = Instant::now();
    gpu.dispatch(
        &pipeline_batched,
        &[&buf_sign, &buf_a_batch, &buf_b_batch, &buf_out_batch],
        (wgs, 1, 1),
    )?;
    let gpu_batch_time = gpu_batch_start.elapsed();

    let result_batch: Vec<f32> = gpu.read_buffer(&buf_out_batch)?;

    // Verify first multivector in batch
    let mut batch_ok = true;
    for mv in 0..batch_size as usize {
        let base = mv * BLADES as usize;
        let a_mv: [f32; 32] = std::array::from_fn(|i| a_batch[base + i]);
        let b_mv: [f32; 32] = std::array::from_fn(|i| b_batch[base + i]);
        let expected_mv = cpu_geometric_product(&a_mv, &b_mv, &sign_table);
        for i in 0..BLADES as usize {
            if (result_batch[base + i] - expected_mv[i]).abs() > 1e-4 {
                batch_ok = false;
            }
        }
        if !batch_ok {
            break;
        }
    }

    println!(
        "  CPU ({batch_size} MVs):  {:>8.1} ms  ({:.1} µs/MV)",
        cpu_batch_time.as_secs_f64() * 1e3,
        cpu_batch_time.as_micros() as f64 / batch_size as f64
    );
    println!(
        "  GPU (single dispatch): {:>8.1} ms  ({:.1} µs/MV)",
        gpu_batch_time.as_secs_f64() * 1e3,
        gpu_batch_time.as_micros() as f64 / batch_size as f64
    );
    let batch_speedup = cpu_batch_time.as_secs_f64() / gpu_batch_time.as_secs_f64();
    let batch_gflops =
        (batch_size as f64 * (32.0 * 32.0 * 32.0) * 2.0) / gpu_batch_time.as_secs_f64() / 1e9;
    println!("  speedup: {batch_speedup:.1}×  ({batch_gflops:.1} GFLOPS)");
    println!(
        "  verification: {}",
        if batch_ok {
            "✅ all correct"
        } else {
            "❌ errors"
        }
    );

    Ok(())
}
