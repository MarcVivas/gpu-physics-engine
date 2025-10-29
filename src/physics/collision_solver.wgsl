override WORKGROUP_SIZE = 64u;
const STIFFNESS: f32 = 0.6;


struct UniformData {
    num_counting_chunks: u32,
    total_cell_ids: u32,
};

@group(0) @binding(0) var<storage, read> chunk_obj_count: array<u32>;
@group(0) @binding(1) var<storage, read> collision_cells: array<u32>;
@group(0) @binding(2) var<storage, read> cell_ids: array<u32>;
@group(0) @binding(3) var<storage, read> object_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> radius: array<f32>;
@group(0) @binding(6) var<uniform> uniform_data: UniformData;



var<workgroup> num_collision_cells: u32;

var<push_constant> current_cell_color: u32;

// Use the collision cells to solve the collisions between objects.
@compute @workgroup_size(WORKGROUP_SIZE)
fn solve_collisions(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>){

    let tid: u32 = global_id.x;

    load_number_of_collision_cells(local_id.x);

    if tid >= num_collision_cells {
        // tid is out of bounds
        return;
    }

    let start = collision_cells[tid];
    let cell_hash: u32 = cell_ids[start];
    let cell_color: u32 = get_cell_color(cell_hash);

    // Only resolve collisions if the cell color matches the current one
    if cell_color == current_cell_color {
        resolve_cell_collisons(cell_hash, start);
    }

}

fn load_number_of_collision_cells(local_id: u32) {
    if local_id == 0 {
        num_collision_cells = chunk_obj_count[uniform_data.num_counting_chunks - 1u];
    }
    workgroupBarrier();
}

fn get_cell_color(cell_hash: u32) -> u32 {
    let cell_grid_coords: vec2<u32> = morton_decode(cell_hash);
    return 1u + (cell_grid_coords.x % 2u) + (cell_grid_coords.y % 2u) * 2u;
}

fn are_colliding(sq_distance: f32, rad_1: f32, rad_2: f32) -> bool {
    let radius_sum = rad_1 + rad_2;
    let sq_radius_sum = radius_sum * radius_sum;
    return sq_radius_sum > sq_distance;
}

fn resolve_cell_collisons(cell_hash: u32, start: u32) {

    for(var i: u32 = start; i < uniform_data.total_cell_ids; i++){
        if cell_ids[i] != cell_hash {
            break;
        }
        let object_id = object_ids[i];



        // Check collisions with the current object and the rest of the objects in the cell
        for(var j: u32 = i + 1; j < uniform_data.total_cell_ids; j++){
            let other_cell_hash = cell_ids[j];

            // Check if the other object is inside the same cell
            if other_cell_hash != cell_hash {break;}

            let other_object_id = object_ids[j];

            let obj_1_pos = positions[object_id];
            let obj_2_pos = positions[other_object_id];
            let obj_1_radius = radius[object_id];
            let obj_2_radius = radius[other_object_id];


            let vec_i_j = obj_1_pos - obj_2_pos;

            let distance = length(vec_i_j);

            if are_colliding(distance * distance, obj_1_radius, obj_2_radius) && distance > 0.0001{
                // Solve collision
                let penetration_depth = (obj_1_radius + obj_2_radius) - distance;
                let collision_direction_vector = vec_i_j / distance;


                let correction_vector: vec2<f32> = collision_direction_vector * penetration_depth * STIFFNESS;

                let inv_mass_1 = 1/obj_1_radius;
                let inv_mass_2 = 1/obj_2_radius;

                // Displace each particles by half of the penetration depth along the collision normal.
                let displacement = correction_vector * (inv_mass_1 / (inv_mass_1+inv_mass_2));
                let displacement_2 = correction_vector * (inv_mass_2 / (inv_mass_1+inv_mass_2));

                positions[object_id] += displacement;
                positions[other_object_id] -= displacement_2;
            }

        }

    }

}

/// Compacts bits from every other position to the lower 16 bits.
/// This is the inverse of `split_by_bits`.
/// Example (2-bit): n = 5 (binary 0101) becomes 3 (binary 11).
fn unsplit_by_bits(n: u32) -> u32 {
    var x = n & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

/// Decodes a 1D Morton index back into 2D coordinates.
/// Example: 15 (binary 1111) -> (x=3, y=3).
fn morton_decode(morton_code: u32) -> vec2<u32> {
    return vec2<u32>(unsplit_by_bits(morton_code), unsplit_by_bits(morton_code >> 1));
}