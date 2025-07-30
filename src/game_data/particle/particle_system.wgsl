
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> previous_positions: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;


// The Compute Shader
// WORKGROUP_SIZE is the number of threads we run in a block. 
const WORKGROUP_SIZE: u32 = 64u;

const FORCE_OF_GRAVITY: vec2<f32> = vec2<f32>(0.0, -39.3);

@compute @workgroup_size(WORKGROUP_SIZE)
fn verlet_integration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Prevent running on more particles than we have
    let num_particles = arrayLength(&positions);
    if (index >= num_particles) {
        return;
    }

    // Get the particle's data
    let current_position = positions[index];
    let previous_position = previous_positions[index];



    // Update the previous position
    // The current position becomes the old position
    previous_positions[index] = current_position;


    // Verlet integration
    let velocity = current_position - previous_position;
    velocities[index] = velocity;

    // Predict the next position without applying constraints
    var predicted_position = current_position + velocity + FORCE_OF_GRAVITY * 0.003 * 0.003;


    let particle_radius = radius[index];
    
    // Apply boundary constraints
    // Boundary check (bouncing)
    predicted_position.x = clamp(predicted_position.x, particle_radius, sim_params.world_width - particle_radius);
    predicted_position.y = clamp(predicted_position.y, particle_radius, sim_params.world_height - particle_radius);


    // Write the updated data back to the buffer
    positions[index] = predicted_position;
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

struct ColorInput {
    @location(3) color: vec4<f32>,
}

struct VelocityInput {
    @location(4) velocity: vec2<f32>,
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
fn vs_main(@builtin(vertex_index) vertex_id : u32,
model: VertexInput, instance: InstanceInput, radius: RadiusInput, color: ColorInput, velocity: VelocityInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = get_particle_color(velocity.velocity);
    out.local_pos = model.position;

    let scaled_position = model.position * radius.radius * 2.0;
    let world_position = scaled_position + instance.position;

    out.clip_position = u_camera.view_proj * vec4<f32>(world_position, 0.0, 1.0);
    return out;
}


const MAX_VELOCITY = 0.2;
fn get_particle_color(particle_velocity: vec2<f32>) -> vec3<f32>{

    // Compute the magnitude of the particle's velocity
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
