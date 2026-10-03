# Getting Started

```rust
use borsalino::GpuBackend;

let wgsl = r#"
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(256)
    fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= 4u) { return; }
        output[i] = input[i] + 1.0;
    }
"#;

let gpu = borsalino::init()?;
let pipeline = gpu.compile("add_one", wgsl)?;
let input = gpu.create_buffer(&[1.0f32, 2.0, 3.0, 4.0])?;
let output = gpu.create_buffer_uninit::<f32>(4)?;

gpu.dispatch(&pipeline, &[&input, &output], (1, 1, 1))?;
let result: Vec<f32> = gpu.read_buffer(&output)?;
assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
```

Run the shipped example:

```sh
cargo run --features metal --example hello_compute    # macOS
cargo run --features vulkan --example hello_compute   # Linux / Windows
```

The `if (i >= 4u) { return; }` guard is load-bearing: the workgroup launches
256 invocations but the buffers hold four elements, and pipelines compile
with `Unchecked` bounds policies — see [Writing Kernels](./guide/wgsl.md).
