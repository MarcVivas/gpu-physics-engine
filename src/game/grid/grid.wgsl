
const WORKGROUP_SIZE = 64u;
// For 2D, an object can touch at most 2^2 = 4 cells.
const MAX_CELLS_PER_OBJECT = 4u;

const UNUSED_CELL_ID = 0xffffffffu;

struct UniformData {
    num_particles: u32,
    num_cell_ids: u32,
    cell_size: f32,
    dim: u32
};

// Bindings for the Compute Shader
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> uniform_data: UniformData;
@group(0) @binding(2) var<storage, read_write> cell_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> object_ids: array<u32>;
@group(0) @binding(4) var<storage, read> radius: array<f32>;
@group(0) @binding(5) var<storage, read_write> collision_cells: array<u32>;
@group(0) @binding(6) var<storage, read_write> chunk_obj_count: array<u32>;




@compute @workgroup_size(WORKGROUP_SIZE)
fn build_cell_ids_array(@builtin(global_invocation_id) global_id: vec3<u32>){

    let obj_id = global_id.x;
    let CELL_SIZE = uniform_data.cell_size;

    if obj_id >= uniform_data.num_particles {
        return;
    }



    let pos = positions[obj_id];
    let radius = radius[obj_id];
    let sq_radius = radius*radius;

    // Convert to grid coordinates.
    let home_cell_coord = vec2<i32>(floor(pos / CELL_SIZE));

    // This is the base index for the cell ids and object ids array, for this object.
    let output_base_idx: u32 = obj_id * MAX_CELLS_PER_OBJECT;


    // Step 1:
    // Always store the home (H) cell
    // The H cell is where the object center is located
    let home_cell_hash = cell_coord_to_hash(home_cell_coord);
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
                cell_ids[output_idx] = cell_coord_to_hash(neighbour_coord);
                object_ids[output_idx] = obj_id;
            }
        }
    }

    // Step 3:
    // Mark as invalid the unused slots
    for(var x = p_cell_count+1; x < MAX_CELLS_PER_OBJECT; x++){
        cell_ids[output_base_idx + x] = UNUSED_CELL_ID;
    }


    //positions[obj_id] = vec2<f32>(f32(obj_id), f32(obj_id));

}

// Shift the y-coordinate into the upper 16 bits, and leave x in the lower 16 bits.
fn cell_coord_to_hash(cell_coord: vec2<i32>) -> u32{
    return (u32(cell_coord.y) << 16) | u32(cell_coord.x);
}

fn is_obj_in_cell(particle_pos: vec2<f32>, particle_sq_radius: f32, cell_coord: vec2<i32>) -> bool {
    let cell_bottom_left_corner: vec2<f32> = vec2<f32>(cell_coord) * uniform_data.cell_size;
    let cell_top_right_corner: vec2<f32> = cell_bottom_left_corner + vec2<f32>(uniform_data.cell_size);

    // Closest point to the object center
    let closest_point = clamp(particle_pos, cell_bottom_left_corner, cell_top_right_corner);

    // Calculate distance between particle and closest point
    let distance_vec: vec2<f32> = particle_pos - closest_point;
    let dist_sq: f32 = dot(distance_vec, distance_vec);

    return dist_sq < particle_sq_radius;
}

const CHUNK_SIZE: u32 = 4;

@compute @workgroup_size(WORKGROUP_SIZE)
fn count_objects_for_each_chunk(@builtin(global_invocation_id) global_id: vec3<u32>){
    let chunk_id = global_id.x;
    let total_cell_ids = uniform_data.num_particles * MAX_CELLS_PER_OBJECT;

    let total_chunks = get_total_chunks(total_cell_ids);
    if chunk_id >= total_chunks{
        return;
    }

    // Get the first index of the chunk
    var first_idx = chunk_id * CHUNK_SIZE;

    // first_idx >= 1 ? cell_ids[first_idx-1] : UNUSED_CELL
    var prev_cell_id = select(UNUSED_CELL_ID, cell_ids[first_idx - 1], first_idx >= 1);


    var obj_count: u32 = 0;
    var currently_counting_cell:u32 =  UNUSED_CELL_ID;
    var current_count: u32 = 0;

    let next_chunk_first_idx = first_idx+CHUNK_SIZE;

    // Count the number of obj in each chunk that share the same cell id
    for(var i: u32 = first_idx; i < total_cell_ids; i++){
        // Get the current cell
        let cell_id = cell_ids[i];

        // Exit contitions
        let is_out_of_bounds: bool = i >= next_chunk_first_idx;
        let is_cell_unused: bool = cell_id == UNUSED_CELL_ID;
        let is_it_a_transition: bool = cell_id != prev_cell_id;
        if (is_it_a_transition && is_out_of_bounds) || is_cell_unused || (current_count == 0 && is_out_of_bounds && !is_it_a_transition) { break; }

        // Counting condition
        let cell_was_seen_before: bool = currently_counting_cell == cell_id;
        if cell_was_seen_before {
            // The cell has more than one object inside
            // Therefore, the objects that are in this cell have to be counted by the thread.
            if current_count == 1 {
                obj_count+=1;
            }
            current_count += 1;
            obj_count += 1;
        }

        if cell_id != prev_cell_id {
            // Transition
            // Reset the current counter
            current_count = 1;
            currently_counting_cell = cell_id;
        }

        // Update the previous cell id
        prev_cell_id = cell_id;
    }

    // Write to memory the number of objects per chunk
    chunk_obj_count[chunk_id] = obj_count;

}

fn get_total_chunks(total_cell_ids: u32) -> u32 {
    return (total_cell_ids + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
}


fn get_val_from_prefix_sum(index: i32) -> u32 {
    return select(0u, chunk_obj_count[index], index >= 0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_collision_cells_array(@builtin(global_invocation_id) global_id: vec3<u32>){
    let total_cell_ids = uniform_data.num_particles * MAX_CELLS_PER_OBJECT;

    let chunk_id: u32 = global_id.x;

    if global_id.x >= get_total_chunks(total_cell_ids) {
        return;
    }

    let start_index = get_val_from_prefix_sum(i32(chunk_id) - 1);
    let end_index = get_val_from_prefix_sum(i32(chunk_id));
    let num_objects_to_manage = end_index - start_index;

    // Determine if this thread has to work with the Prefix sum
    let has_work_to_do: bool = num_objects_to_manage > 0;
    if !has_work_to_do {
        return;
    }

    // This thread has work to do
    // Create collision cells for every object that shares a cell id in the chunk
    // Get the first index of the chunk
    var first_idx = chunk_id * CHUNK_SIZE;

    // first_idx >= 1 ? cell_ids[first_idx-1] : UNUSED_CELL
    var prev_cell_id = select(UNUSED_CELL_ID, cell_ids[first_idx - 1], first_idx >= 1);


    var obj_count: u32 = 0;
    var currently_counting_cell:u32 =  UNUSED_CELL_ID;
    var current_count: u32 = 0;

    let next_chunk_first_idx = first_idx+CHUNK_SIZE;

    var write_index = start_index; 
    
    // Count the number of obj in each chunk that share the same cell id
    for(var i: u32 = first_idx; i < total_cell_ids && write_index != end_index; i++){
        // Get the current cell
        let cell_id = cell_ids[i];

        // Exit contitions
        let is_out_of_bounds: bool = i >= next_chunk_first_idx;
        let is_cell_unused: bool = cell_id == UNUSED_CELL_ID;
        let is_it_a_transition: bool = cell_id != prev_cell_id;
        if (is_it_a_transition && is_out_of_bounds) || is_cell_unused || (current_count == 0 && is_out_of_bounds && !is_it_a_transition) { break; }

        // Counting condition
        let cell_was_seen_before: bool = currently_counting_cell == cell_id;
        if cell_was_seen_before {
            // The cell has more than one object inside
            // Therefore, the objects that are in this cell have to be counted by the thread.
            if current_count == 1 {
                // The previous obj shared the same cell id
                // Store it
                collision_cells[write_index] = i - 1;
                write_index++; 
            }
            // Store the current object
            collision_cells[write_index] = i;
            write_index++;
            current_count += 1;
        }

        if cell_id != prev_cell_id {
            // Transition
            // Reset the current counter
            current_count = 1;
            currently_counting_cell = cell_id;
        }

        // Update the previous cell id
        prev_cell_id = cell_id;
    }

}

