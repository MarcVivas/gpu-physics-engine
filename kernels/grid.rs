// Port of src/grid/grid.wgsl: builds the (cell id, object id) pairs, one home
// cell plus up to three phantom cells per particle (MAX_CELLS_PER_OBJECT = 4).
//
// WGSL binding 1 is a uniform buffer this entry point never reads, so it has
// no parameter here; #[binding(N)] keeps every other buffer where the host
// puts it. Push constants: PushConstantsBuildGrid { cell_size, num_particles }.

const MAX_CELLS_PER_OBJECT: u32 = 4;
const UNUSED_CELL_ID: u32 = 0xffffffff;

fn split_by_bits(n: u32) -> u32 {
    let mut x: u32 = n & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

fn morton_encode(x: i32, y: i32) -> u32 {
    split_by_bits(x as u32) | (split_by_bits(y as u32) << 1)
}

/// Does the circle at (px, py) with squared radius sq_radius touch the cell?
fn is_obj_in_cell(px: f32, py: f32, sq_radius: f32, cell_x: i32, cell_y: i32, cell_size: f32) -> bool {
    let min_x: f32 = (cell_x as f32) * cell_size;
    let min_y: f32 = (cell_y as f32) * cell_size;
    let max_x: f32 = min_x + cell_size;
    let max_y: f32 = min_y + cell_size;

    // Closest point of the cell to the object center
    let qx: f32 = px.clamp(min_x, max_x);
    let qy: f32 = py.clamp(min_y, max_y);

    let dx: f32 = px - qx;
    let dy: f32 = py - qy;
    dx * dx + dy * dy < sq_radius
}

#[kernel(local_size = 64)]
fn build_cell_ids_array(
    #[binding(0)] positions: &mut [f32],
    #[binding(2)] cell_ids: &mut [u32],
    #[binding(3)] object_ids: &mut [u32],
    #[binding(4)] radius: &[f32],
    cell_size: f32,
    num_particles: u32,
) {
    let obj_id: u32 = gpu_global_id(0);
    if obj_id >= num_particles {
        return;
    }

    let px: f32 = positions[(obj_id * 2) as usize];
    let py: f32 = positions[(obj_id * 2 + 1) as usize];
    let r: f32 = radius[obj_id as usize];
    let sq_radius: f32 = r * r;

    // Convert to grid coordinates.
    let home_x: i32 = (px / cell_size).floor() as i32;
    let home_y: i32 = (py / cell_size).floor() as i32;

    // Base index for this object in the cell ids and object ids arrays.
    let base: u32 = obj_id * MAX_CELLS_PER_OBJECT;

    // Step 1: always store the home (H) cell, where the center is.
    cell_ids[base as usize] = morton_encode(home_x, home_y);
    object_ids[base as usize] = obj_id;

    // Step 2: check the 8 neighbours and store the phantom (P) cells.
    let mut p_cell_count: u32 = 0;
    for y in -1..=1 {
        for x in -1..=1 {
            if x == 0 && y == 0 {
                // The home cell, already stored.
                continue;
            }
            let nx: i32 = home_x + x;
            let ny: i32 = home_y + y;
            if is_obj_in_cell(px, py, sq_radius, nx, ny, cell_size) {
                p_cell_count += 1;
                let idx: usize = (base + p_cell_count) as usize;
                cell_ids[idx] = morton_encode(nx, ny);
                object_ids[idx] = obj_id;
            }
        }
    }

    // Step 3: mark the unused slots invalid.
    for slot in (p_cell_count + 1)..MAX_CELLS_PER_OBJECT {
        cell_ids[(base + slot) as usize] = UNUSED_CELL_ID;
    }
}
