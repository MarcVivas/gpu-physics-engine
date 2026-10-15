// Port of src/particles/particle_integration.wgsl (Verlet integration).
//
// The push constants are SimParams { delta_time, world_width, world_height,
// is_mouse_pressed, mouse_pos: vec2<f32>, num_particles }. The vec2 is passed
// as two scalars, mouse_x and mouse_y: they land at byte offsets 16 and 20,
// exactly where the vec2 sits in the WGSL struct.

const MOUSE_ATTRACTION_STRENGTH: f32 = 150.0;

#[kernel(local_size = 64)]
fn verlet_integration(
    positions: &mut [f32],
    previous_positions: &mut [f32],
    radius: &[f32],
    delta_time: f32,
    world_width: f32,
    world_height: f32,
    is_mouse_pressed: u32,
    mouse_x: f32,
    mouse_y: f32,
    num_particles: u32,
) {
    let index: u32 = gpu_global_id(0);

    // Prevent running on more particles than we have
    if index >= num_particles {
        return;
    }

    let i: usize = (index * 2) as usize;
    let cx: f32 = positions[i];
    let cy: f32 = positions[i + 1];
    let px: f32 = previous_positions[i];
    let py: f32 = previous_positions[i + 1];

    // Verlet integration
    let vx: f32 = cx - px;
    let vy: f32 = cy - py;

    let mut ax: f32 = 0.0;
    let mut ay: f32 = 0.0;

    if is_mouse_pressed == 1 {
        // Unit vector from the particle to the mouse, scaled by the strength.
        let dx: f32 = mouse_x - cx;
        let dy: f32 = mouse_y - cy;
        let len: f32 = (dx * dx + dy * dy).sqrt();
        ax += dx / len * MOUSE_ATTRACTION_STRENGTH;
        ay += dy / len * MOUSE_ATTRACTION_STRENGTH;
    }

    // Predict the next position without applying constraints
    let dt_squared: f32 = delta_time * delta_time;
    let mut nx: f32 = cx + vx + ax * dt_squared;
    let mut ny: f32 = cy + vy + ay * dt_squared;

    // The current position becomes the old position
    previous_positions[i] = cx;
    previous_positions[i + 1] = cy;

    // Boundary constraints
    let r: f32 = radius[index as usize];
    nx = nx.clamp(r, world_width - r);
    ny = ny.clamp(r, world_height - r);

    positions[i] = nx;
    positions[i + 1] = ny;
}
