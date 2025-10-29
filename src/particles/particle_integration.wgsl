
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
    is_mouse_pressed: u32,
    mouse_pos: vec2<f32>,
    num_particles: u32,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> previous_positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> radius: array<f32>;


var<push_constant> push_constants: SimParams;

const WORKGROUP_SIZE: u32 = 64u;

const FORCE_OF_GRAVITY: vec2<f32> = vec2<f32>(0.0, 0.0);
const MOUSE_ATTRACTION_STRENGTH: f32 = 150.0;

@compute @workgroup_size(WORKGROUP_SIZE)
fn verlet_integration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Prevent running on more particles than we have
    if (index >= push_constants.num_particles) {
        return;
    }

    // Get the particles's data
    let current_position = positions[index];
    let previous_position = previous_positions[index];



    // Verlet integration
    let velocity: vec2<f32> = (current_position - previous_position);

    var total_acceleration = FORCE_OF_GRAVITY;

    if (push_constants.is_mouse_pressed == 1u) {
        // Calculate a vector pointing from the particle to the mouse
        let direction_to_mouse = push_constants.mouse_pos - current_position;

        // Normalize the direction to get a unit vector, then scale by our strength constant.
        // This creates the mouse attraction acceleration.
        let mouse_attraction_accel = normalize(direction_to_mouse) * MOUSE_ATTRACTION_STRENGTH;

        // Add the mouse attraction to the total acceleration
        total_acceleration += mouse_attraction_accel;

    }

    // Predict the next position without applying constraints
    let dt_squared = push_constants.delta_time * push_constants.delta_time;
    var predicted_position: vec2<f32> = current_position + velocity + total_acceleration * dt_squared;


    // Update the previous position
    // The current position becomes the old position
    previous_positions[index] = current_position;

    let particle_radius = radius[index];
    
    // Apply boundary constraints
    // Boundary check (bouncing)
    predicted_position.x = clamp(predicted_position.x, particle_radius, push_constants.world_width - particle_radius);
    predicted_position.y = clamp(predicted_position.y, particle_radius, push_constants.world_height - particle_radius);



    // Write the updated data back to the buffer
    positions[index] = predicted_position;
}

    /*
    // Circle world
    let world_center = vec2<f32>(push_constants.world_width/2.0, push_constants.world_height/2.0);
    let world_radius = min(push_constants.world_width/2.0, push_constants.world_height/2.0);
    let vec_particle_world_center = predicted_position-world_center;
    let distance_from_center = dot(vec_particle_world_center, vec_particle_world_center);
    let max_radius =  world_radius - particle_radius;

    if(distance_from_center > max_radius * max_radius){
            // Move the particle back to the nearest point on the circle's edge
            predicted_position = world_center + max_radius * normalize(vec_particle_world_center);
    }
    */

