
struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) position: vec2<f32>,
}

struct RadiusInput {
    @location(2) radius: f32,
}
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local_pos: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput, radius: RadiusInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vec3<f32>(0.4, 0.3, 0.1);
    out.local_pos = model.position;

    let scaled_position = model.position * radius.radius;
    let world_position = scaled_position + instance.position;

    out.clip_position = u_camera.view_proj * vec4<f32>(world_position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>{
    // Distance of the current pixel from the center (0.0, 0.0)
    let dist_sq = dot(in.local_pos, in.local_pos);

    // Compute alpha value for this pixel
    // When dist_sq is <= 0.2304 smoothstep is 0. Thus, alpha = 1 and the pixel is close to the center.
    // When dist_sq is >= 0.25 smoothstep is 1. Thus, alpha = 0 and the pixel is far from the center.
    // When dist_sq is in between 0.2304 0.25, smoothstep is in between 0 and 1. Creates a smooth fading effect
    let alpha = 1.0 - smoothstep(0.2304, 0.25, dist_sq);

    return vec4<f32>(in.color, alpha);
}
