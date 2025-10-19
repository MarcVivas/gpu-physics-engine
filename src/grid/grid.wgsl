override WORKGROUP_SIZE = 64u;
// For 2D, an object can touch at most 2^2 = 4 cells.
override MAX_CELLS_PER_OBJECT = 4u;

const UNUSED_CELL_ID = 0xffffffffu;

struct UniformData {
    num_particles: u32,
    num_collision_cells: u32,
    cell_size: f32,
    delta_time: f32,
    cell_color: u32,
    num_counting_chunks: u32,
};

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> uniform_data: UniformData;
@group(0) @binding(2) var<storage, read_write> cell_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> object_ids: array<u32>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;


struct PushConstantsBuildGrid {
    cell_size: f32,
    num_particles: u32,
}

var<push_constant> push_constants_build_grid: PushConstantsBuildGrid;


@compute @workgroup_size(WORKGROUP_SIZE)
fn build_cell_ids_array(@builtin(global_invocation_id) global_id: vec3<u32>){

    let obj_id = global_id.x;

    if obj_id >= push_constants_build_grid.num_particles {
        return;
    }

    let pos = positions[obj_id];
    let radius = radius[obj_id];
    let sq_radius = radius*radius;

    // Convert to grid coordinates.
    let home_cell_coord = vec2<i32>(floor(pos / push_constants_build_grid.cell_size));

    // This is the base index for the cell ids and object ids array, for this object.
    let output_base_idx: u32 = obj_id * MAX_CELLS_PER_OBJECT;


    // Step 1:
    // Always store the home (H) cell
    // The H cell is where the object center is located
    let home_cell_hash = morton_encode(home_cell_coord);
    cell_ids[output_base_idx] = home_cell_hash;
    object_ids[output_base_idx] = obj_id;

    // Step 2:
    // Check the 8 surrounding neighbours and store the phantom (P) cells
    var p_cell_count = 0u;
    for (var y = -1; y <= 1; y++ ){
        for (var x = -1; x <= 1; x++){
            if x == 0 && y == 0 {
                // This would be the home cell
                // Skip it since it was previously stored
                continue;
            }

            let offset = vec2<i32>(x, y);
            let neighbour_coord = home_cell_coord + offset;

            if is_obj_in_cell(pos, sq_radius, neighbour_coord) {
                // The object was found in the neighbour cell
                // Thus, this is a phantom (P) cell
                p_cell_count++;
                let output_idx = output_base_idx + p_cell_count;
                cell_ids[output_idx] = morton_encode(neighbour_coord);
                object_ids[output_idx] = obj_id;
            }
        }
    }

    // Step 3:
    // Mark as invalid the unused slots
    for(var x = p_cell_count+1; x < MAX_CELLS_PER_OBJECT; x++){
        cell_ids[output_base_idx + x] = UNUSED_CELL_ID;
    }


}

/// Spreads the lower 16 bits of an integer to every other bit.
/// Example (2-bit): n = 3 (binary 11) becomes 5 (binary 0101).
fn split_by_bits(n: u32) -> u32 {
    var x = n & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

/// Encodes 2D coordinates (16-bit max) into a 1D Morton index.
/// Example: (x=3, y=3) -> (binary 11, 11) -> interleaved 1111 -> 15.
fn morton_encode(v: vec2<i32>) -> u32 {
    return split_by_bits(u32(v.x)) | (split_by_bits(u32(v.y)) << 1);
}


fn is_obj_in_cell(particle_pos: vec2<f32>, particle_sq_radius: f32, cell_coord: vec2<i32>) -> bool {
    let cell_bottom_left_corner: vec2<f32> = vec2<f32>(cell_coord) * push_constants_build_grid.cell_size;
    let cell_top_right_corner: vec2<f32> = cell_bottom_left_corner + vec2<f32>(push_constants_build_grid.cell_size);

    // Closest point to the object center
    let closest_point = clamp(particle_pos, cell_bottom_left_corner, cell_top_right_corner);

    // Calculate distance between particles and closest point
    let distance_vec: vec2<f32> = particle_pos - closest_point;
    let dist_sq: f32 = dot(distance_vec, distance_vec);

    return dist_sq < particle_sq_radius;
}


