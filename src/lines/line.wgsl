struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) thickness: f32,
};

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) thickness: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u_camera.view_proj * vec4<f32>(input.position, 0.0, 1.0);
    out.color = input.color;
    out.thickness = input.thickness;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use the per-vertex color
    // Note: thickness affects lines width in the GPU, but for custom thickness
    // you might need to implement thick lines using geometry shaders or quads
    return input.color;
}
