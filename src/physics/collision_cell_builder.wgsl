override CHUNK_SIZE: u32 = 4;
override WORKGROUP_SIZE = 64u;
// For 2D, an object can touch at most 2^2 = 4 cells.
override MAX_CELLS_PER_OBJECT = 4u;
const UNUSED_CELL_ID = 0xffffffffu;

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
};

struct UniformData {
    num_counting_chunks: u32,
    total_cell_ids: u32,
};

@group(0) @binding(0) var<storage, read_write> chunk_obj_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> collision_cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect_args: DispatchArgs;
@group(0) @binding(3) var<uniform> uniform_data: UniformData;
@group(0) @binding(4) var<storage, read> cell_ids: array<u32>;



@compute @workgroup_size(WORKGROUP_SIZE)
fn count_objects_for_each_chunk(@builtin(global_invocation_id) global_id: vec3<u32>){
    let chunk_id = global_id.x;
    let total_cell_ids = uniform_data.total_cell_ids;

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

fn prepare_dispatch_buffer(){

    // Prepare the dispatch buffer
    // Read the total count from the last element of the prefix sum buffer
    let total_items = chunk_obj_count[uniform_data.num_counting_chunks - 1];

    // Calculate workgroups needed
    let workgroups_x = (total_items + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Write to the indirect buffer
    indirect_args.x = workgroups_x;
    indirect_args.y = 1u;
    indirect_args.z = 1u;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_collision_cells_array(@builtin(global_invocation_id) global_id: vec3<u32>){
    let total_cell_ids = uniform_data.total_cell_ids;

    let chunk_id: u32 = global_id.x;

    if global_id.x == 0 {
        prepare_dispatch_buffer();
    }

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
                // Store the first position of the cell
                collision_cells[write_index] = i - 1;
                write_index++; 
            }
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
