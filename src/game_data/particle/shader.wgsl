
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;

// The Compute Shader
// WORKGROUP_SIZE is the number of threads we run in a block. 
const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Prevent running on more particles than we have
    let num_particles = arrayLength(&positions);
    if (index >= num_particles) {
        return;
    }

    // Get the current particle's data
    var pos = positions[index];
    var vel = velocities[index];

    // --- Simulation Logic ---
    // Update position based on velocity and delta_time
    pos += vel * sim_params.delta_time;

    // Boundary check (bouncing)
    if (pos.x < 0.0) {
        pos.x = 0.0;
        vel.x *= -1.0;
    }
    if (pos.x > sim_params.world_width) {
        pos.x = sim_params.world_width;
        vel.x *= -1.0;
    }
    if (pos.y < 0.0) {
        pos.y = 0.0;
        vel.y *= -1.0;
    }
    if (pos.y > sim_params.world_height) {
        pos.y = sim_params.world_height;
        vel.y *= -1.0;
    }
    // --- End Simulation Logic ---

    // Write the updated data back to the buffer
    positions[index] = pos;
    velocities[index] = vel;
}

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
    @location(2) radius: f32,
};

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput, radius: RadiusInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = vec3<f32>(0.4, 0.3, 0.1);
    out.local_pos = model.position;

    let scaled_position = model.position * radius.radius;
    let world_position = scaled_position + instance.position;

    out.clip_position = u_camera.view_proj * vec4<f32>(world_position, 0.0, 1.0);
    out.radius = radius.radius;
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
    let alpha = select(
                    1.0 - smoothstep(0.2304, 0.25, dist_sq), // This is returned if 'condition' is false
                    1.0,                                     // This is returned if 'condition' is true
                    in.radius < 2 // The boolean condition
                );

    return vec4<f32>(in.color, alpha);
}
