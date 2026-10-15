// Port of src/particles/home_cell_ids.wgsl to the Crust Rust subset.
//
// Bindings and the push-constant layout match the WGSL, so the host side does
// not change: positions is an interleaved [x0, y0, x1, y1, ..] f32 buffer (the
// same bytes as array<vec2<f32>>), and the scalar parameters, in order, are
// the PushConstantsData struct { num_particles: u32, cell_size: f32 }.

/// Spreads the lower 16 bits of an integer to every other bit.
/// Example (2-bit): n = 3 (binary 11) becomes 5 (binary 0101).
fn split_by_bits(n: u32) -> u32 {
    let mut x: u32 = n & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Encodes 2D coordinates (16-bit max) into a 1D Morton index.
fn morton_encode(x: i32, y: i32) -> u32 {
    split_by_bits(x as u32) | (split_by_bits(y as u32) << 1)
}

#[kernel(local_size = 64)]
fn create_home_cell_ids(
    positions: &mut [f32],
    home_cell_ids: &mut [u32],
    particle_ids: &mut [u32],
    num_particles: u32,
    cell_size: f32,
) {
    let obj_id: u32 = gpu_global_id(0);
    if obj_id >= num_particles {
        return;
    }

    let px: f32 = positions[(obj_id * 2) as usize];
    let py: f32 = positions[(obj_id * 2 + 1) as usize];

    // Convert to grid coordinates.
    let cx: i32 = (px / cell_size).floor() as i32;
    let cy: i32 = (py / cell_size).floor() as i32;

    home_cell_ids[obj_id as usize] = morton_encode(cx, cy);
    particle_ids[obj_id as usize] = obj_id;
}
