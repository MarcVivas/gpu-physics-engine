struct VertexInput {
    @location(0) position: vec2<f32>,
};


struct Camera {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> u_camera: Camera;
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> previous_positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> radius: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) local_pos: vec2<f32>,
};

@vertex
fn vs_main(@builtin(instance_index) instance_id : u32,
model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let particle_pos = positions[instance_id];
    let radius = radius[instance_id];
    let vel = particle_pos - previous_positions[instance_id];

    out.color = get_particle_color(vel);
    out.local_pos = model.position;

    let scaled_position = model.position * radius * 2.0;
    let world_position = scaled_position + particle_pos;

    out.clip_position = u_camera.view_proj * vec4<f32>(world_position, 0.0, 1.0);
    return out;
}


const MAX_VELOCITY = 0.6;
fn get_particle_color(particle_velocity: vec2<f32>) -> vec3<f32>{

    // Compute the magnitude of the particles's velocity
    let velocity_magnitude = length(particle_velocity);

    // Compute a normalized velocity value between 0 and 1
    let normalizedVelocity = clamp(velocity_magnitude / MAX_VELOCITY, 0.0, 1.0);

    // Define three colors for the gradient (e.g. red, orange, yellow)
    let colorLow = vec3<f32>(0.0, 0.0, 1.0); // blue (slowest)
    let colorMid = vec3<f32>(1.0, 0.5, 1.0); // orange (middle)
    let colorHigh = vec3<f32>(1.0, 1.0, 0.0); // yellow (fastest)

    /*
    vec3 colorLow = vec3(1.0, 0.0, 0.0); // red (slowest)
    vec3 colorMid = vec3(1.0, 0.5, 0.0); // orange (middle)
    vec3 colorHigh = vec3(1.0, 1.0, 0.0); // yellow (fastest)
*/
    // Interpolate between the three colors based on the normalized velocity value
    let smoothNormalizedVelocity1 = smoothstep(0.0, 0.5, normalizedVelocity);
    let smoothNormalizedVelocity2 = smoothstep(0.5, 1.0, normalizedVelocity);

    var color = mix(colorLow, colorMid, smoothNormalizedVelocity1);
    color = mix(color, colorHigh, smoothNormalizedVelocity2);

    return color; // pass the velocity to the fragment shader
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
