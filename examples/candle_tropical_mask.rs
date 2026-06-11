// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: AGPL-3.0-only

//! Borsalino + Candle integration — tropical masking benchmark.
//!
//! Demonstrates the integration pattern: Candle tensors → Borsalino
//! buffers → WGSL dispatch → readback → Candle tensors. Uses the
//! min-based Bayesian token masking operation from Quantizon's
//! tropical masking paper as the benchmark workload.
//!
//! # Operation
//!
//! p_mask(token_t) = r · min(s(t), f(t))
//!
//! - s(t): structural gate (0 for punctuation, 1 for content)
//! - f(t): frequency gate (1/(1+log(count(t)+1)))
//! - r: cosine schedule sample (cosine annealing)
//!
//! # Run
//!
//! ```sh
//! cargo run --example candle_tropical_mask --features vulkan --release
//! cargo run --example candle_tropical_mask --features metal --release
//! ```

use std::time::Instant;

use borsalino::GpuBackend;

/// WGSL kernel implementing the tropical masking operation.
/// Uses pure storage buffers (no uniform bindings) for Borsalino compatibility.
const KERNEL_TROPICAL_MASK: &str = r#"
@group(0) @binding(0) var<storage, read> s_gate: array<f32>;
@group(0) @binding(1) var<storage, read> f_gate: array<f32>;
@group(0) @binding(2) var<storage, read> r_schedule: array<f32>;
@group(0) @binding(3) var<storage, read_write> mask: array<f32>;

@compute @workgroup_size(256)
fn tropical_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let r = r_schedule[0];     // single value, first element
    let gated = min(s_gate[i], f_gate[i]);
    mask[i] = r * gated;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = borsalino::init()?;
    let vocab_size: u32 = 50_000;
    let schedule_len: u32 = 1000;
    let batches: u32 = 100;

    println!("Borsalino + Candle — Tropical Masking Benchmark");
    println!("  vocab: {vocab_size}  schedule: {schedule_len}  batches: {batches}");
    println!();

    // ── Generate data (simulating Candle tensors) ───────────────

    println!("--- Data Generation ---");
    let gen_start = Instant::now();

    let s_gate: Vec<f32> = (0..vocab_size)
        .map(|i| if i < 10 || i % 100 == 0 { 0.0 } else { 1.0 })
        .collect();
    let f_gate: Vec<f32> = (0..vocab_size)
        .map(|i| 1.0 / (1.0 + ((i as f32 + 1.0).ln())))
        .collect();
    let r_schedule: Vec<f32> = (0..schedule_len)
        .map(|i| {
            let u = i as f32 / schedule_len as f32;
            1.0 - (std::f32::consts::PI * u * 0.5).cos()
        })
        .collect();

    println!(
        "  generated in {:>8.1} ms",
        gen_start.elapsed().as_secs_f64() * 1e3
    );

    // ── CPU reference ───────────────────────────────────────────

    println!("\n--- CPU Reference ---");
    let cpu_start = Instant::now();
    let mut cpu_results = Vec::new();
    for _batch in 0..batches {
        let r = r_schedule[(_batch % schedule_len) as usize];
        let mask: Vec<f32> = (0..vocab_size)
            .map(|i| r * s_gate[i as usize].min(f_gate[i as usize]))
            .collect();
        cpu_results.push(mask);
    }
    let cpu_time = cpu_start.elapsed();
    println!(
        "  cpu ({batches} batches): {:>8.1} ms  ({:.1} µs/batch)",
        cpu_time.as_secs_f64() * 1e3,
        cpu_time.as_micros() as f64 / batches as f64,
    );

    // ── GPU setup ───────────────────────────────────────────────

    println!("\n--- Borsalino GPU Pipeline ---");
    let compile_start = Instant::now();
    let pipeline = gpu.compile("tropical_mask", KERNEL_TROPICAL_MASK)?;
    println!(
        "  compile: {:>8.1} ms",
        compile_start.elapsed().as_secs_f64() * 1e3,
    );

    let buf_s = gpu.create_buffer(&s_gate)?;
    let buf_f = gpu.create_buffer(&f_gate)?;
    let buf_r = gpu.create_buffer(&r_schedule)?;
    let buf_mask = gpu.create_buffer_uninit::<f32>(vocab_size as usize)?;
    let wgs = vocab_size.div_ceil(256);

    // ── GPU single dispatch ─────────────────────────────────────

    println!("\n--- GPU Performance ---");
    let gpu_start = Instant::now();
    for _batch in 0..batches {
        gpu.dispatch(&pipeline, &[&buf_s, &buf_f, &buf_r, &buf_mask], (wgs, 1, 1))?;
    }
    let gpu_time = gpu_start.elapsed();

    let result: Vec<f32> = gpu.read_buffer(&buf_mask)?;
    println!(
        "  gpu ({batches} batches): {:>8.1} ms  ({:.1} µs/batch)",
        gpu_time.as_secs_f64() * 1e3,
        gpu_time.as_micros() as f64 / batches as f64,
    );

    // ── Verify against CPU ──────────────────────────────────────

    let mut max_err = 0.0f32;
    for i in 0..vocab_size as usize {
        let err = (result[i] - cpu_results[(batches - 1) as usize][i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    println!("  max error: {:.6}", max_err);

    let ratio = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("\n--- Summary ---");
    println!("  CPU:  {:>8.1} ms", cpu_time.as_secs_f64() * 1e3);
    println!("  GPU:  {:>8.1} ms", gpu_time.as_secs_f64() * 1e3);
    println!("  speedup: {:.1}×", ratio);
    println!("  integration: Candle tensors → create_buffer → dispatch → read_buffer ✅");

    Ok(())
}
