
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
    is_mouse_pressed: u32,
    mouse_pos: vec2<f32>,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> previous_positions: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;


// The Compute Shader
// WORKGROUP_SIZE is the number of threads we run in a block. 
const WORKGROUP_SIZE: u32 = 64u;

const FORCE_OF_GRAVITY: vec2<f32> = vec2<f32>(0.0, -39.3);
const MOUSE_ATTRACTION_STRENGTH: f32 = 150.0;

const DELTA_TIME: f32 = 0.003;
const DELTA_TIME_SQUARED: f32 = DELTA_TIME * DELTA_TIME;

@compute @workgroup_size(WORKGROUP_SIZE)
fn verlet_integration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Prevent running on more particles than we have
    let num_particles = arrayLength(&positions);
    if (index >= num_particles) {
        return;
    }

    // Get the particles's data
    let current_position = positions[index];
    let previous_position = previous_positions[index];



    // Update the previous position
    // The current position becomes the old position
    previous_positions[index] = current_position;


    // Verlet integration
    let velocity = current_position - previous_position;

    var total_acceleration = FORCE_OF_GRAVITY;

    if (sim_params.is_mouse_pressed == 1u) {
        // Calculate a vector pointing from the particle to the mouse
        let direction_to_mouse = sim_params.mouse_pos - current_position;

        // Normalize the direction to get a unit vector, then scale by our strength constant.
        // This creates the mouse attraction acceleration.
        let mouse_attraction_accel = normalize(direction_to_mouse) * MOUSE_ATTRACTION_STRENGTH;

        // Add the mouse attraction to the total acceleration
        total_acceleration += mouse_attraction_accel;

    }

    // Predict the next position without applying constraints
    var predicted_position = current_position + velocity + total_acceleration * DELTA_TIME_SQUARED;


    let particle_radius = radius[index];
    
    // Apply boundary constraints
    // Boundary check (bouncing)
    predicted_position.x = clamp(predicted_position.x, particle_radius, sim_params.world_width - particle_radius);
    predicted_position.y = clamp(predicted_position.y, particle_radius, sim_params.world_height - particle_radius);

    /*
    // Circle world
    let world_center = vec2<f32>(sim_params.world_width/2.0, sim_params.world_height/2.0);
    let world_radius = min(sim_params.world_width/2.0, sim_params.world_height/2.0);
    let vec_particle_world_center = predicted_position-world_center;
    let distance_from_center = dot(vec_particle_world_center, vec_particle_world_center);
    let max_radius =  world_radius - particle_radius;

    if(distance_from_center > max_radius * max_radius){
            // Move the particle back to the nearest point on the circle's edge
            predicted_position = world_center + max_radius * normalize(vec_particle_world_center);
    }
    */

    // Write the updated data back to the buffer
    positions[index] = predicted_position;
}

struct VertexInput {
    @location(0) position: vec2<f32>,
};


struct Camera {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> u_camera: Camera;

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
