// Port of src/physics/collision_solver.wgsl: one invocation per collision
// cell; if the cell's colour is the current pass's colour it pushes every
// overlapping pair in the cell apart.
//
// The WGSL keeps the cell count in `var<workgroup>`: invocation 0 loads
// chunk_obj_count[num_counting_chunks - 1] and a barrier publishes it. Every
// invocation reads the same value, so each one simply loads it itself here --
// identical result, no workgroup memory and no barrier (the barrier sat before
// the early return, so nothing else depended on it).
//
// WGSL binding 6 is `var<uniform> UniformData { num_counting_chunks,
// total_cell_ids }`; it is a read-only storage buffer of two u32 here.
// Push constant: current_cell_color.

const STIFFNESS: f32 = 0.6;
const MIN_DISTANCE: f32 = 0.0001;

/// Compacts bits from every other position to the lower 16 bits.
fn unsplit_by_bits(n: u32) -> u32 {
    let mut x: u32 = n & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    x
}

fn get_cell_color(cell_hash: u32) -> u32 {
    let gx: u32 = unsplit_by_bits(cell_hash);
    let gy: u32 = unsplit_by_bits(cell_hash >> 1);
    1 + (gx % 2) + (gy % 2) * 2
}

#[kernel(local_size = 64)]
fn solve_collisions(
    #[binding(0)] chunk_obj_count: &[u32],
    #[binding(1)] collision_cells: &[u32],
    #[binding(2)] cell_ids: &[u32],
    #[binding(3)] object_ids: &[u32],
    #[binding(4)] positions: &mut [f32],
    #[binding(5)] radius: &[f32],
    #[binding(6)] uniform_data: &[u32],
    current_cell_color: u32,
) {
    let tid: u32 = gpu_global_id(0);
    let num_counting_chunks: u32 = uniform_data[0];
    let total_cell_ids: u32 = uniform_data[1];
    let num_collision_cells: u32 = chunk_obj_count[(num_counting_chunks - 1) as usize];

    if tid >= num_collision_cells {
        return;
    }

    let start: u32 = collision_cells[tid as usize];
    let cell_hash: u32 = cell_ids[start as usize];
    if get_cell_color(cell_hash) != current_cell_color {
        return;
    }

    let mut i: u32 = start;
    while i < total_cell_ids {
        if cell_ids[i as usize] != cell_hash {
            break;
        }
        let a: usize = object_ids[i as usize] as usize;

        // Collide this object with the rest of the objects in the cell.
        let mut j: u32 = i + 1;
        while j < total_cell_ids {
            if cell_ids[j as usize] != cell_hash {
                break;
            }
            let b: usize = object_ids[j as usize] as usize;

            let ax: f32 = positions[a * 2];
            let ay: f32 = positions[a * 2 + 1];
            let bx: f32 = positions[b * 2];
            let by: f32 = positions[b * 2 + 1];
            let ra: f32 = radius[a];
            let rb: f32 = radius[b];

            let vx: f32 = ax - bx;
            let vy: f32 = ay - by;
            let distance: f32 = (vx * vx + vy * vy).sqrt();

            let radius_sum: f32 = ra + rb;
            if radius_sum * radius_sum > distance * distance && distance > MIN_DISTANCE {
                let penetration: f32 = radius_sum - distance;
                let dx: f32 = vx / distance;
                let dy: f32 = vy / distance;
                let cx: f32 = dx * penetration * STIFFNESS;
                let cy: f32 = dy * penetration * STIFFNESS;

                let inv_a: f32 = 1.0 / ra;
                let inv_b: f32 = 1.0 / rb;
                let fa: f32 = inv_a / (inv_a + inv_b);
                let fb: f32 = inv_b / (inv_a + inv_b);

                positions[a * 2] = ax + cx * fa;
                positions[a * 2 + 1] = ay + cy * fa;
                positions[b * 2] = bx - cx * fb;
                positions[b * 2 + 1] = by - cy * fb;
            }
            j += 1;
        }
        i += 1;
    }
}
