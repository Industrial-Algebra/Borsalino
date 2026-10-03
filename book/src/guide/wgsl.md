# Writing Kernels in WGSL

Kernels are authored in **WGSL**; [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
translates to each backend's native format:

- Metal: WGSL → MSL → Metal compiler
- Vulkan: WGSL → SPIR-V → `vkCreateComputePipelines`

Buffer bindings use `@group(0) @binding(N)`; the dispatch buffer position
maps directly: `buffers[0]` → `@binding(0)`, `buffers[1]` → `@binding(1)`.

## Bound your indices

Pipelines compile with `Unchecked` bounds policies — bounds checks are the
kernel's responsibility:

```rust
@compute @workgroup_size(256)
fn add_one(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 4u) { return; }        // required when buffer < workgroup
    output[i] = input[i] + 1.0;
}
```

A dispatch of `(1, 1, 1)` workgroups with `@workgroup_size(256)` launches 256
invocations regardless of buffer length — the guard is not optional.
