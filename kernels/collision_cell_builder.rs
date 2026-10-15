// Port of src/physics/collision_cell_builder.wgsl: two entry points that find
// the cells holding more than one object, in chunks of CHUNK_SIZE sorted cell
// ids, so the solver can later walk only the cells that can collide.
//
// Bindings are the WGSL's. Two differences from the original:
//   * binding 3 is a `var<uniform>` UniformData in WGSL. It is a read-only
//     storage buffer here (Crust has no uniform buffer class yet); its two
//     u32 fields are read as uniform_data[0] (num_counting_chunks) and
//     uniform_data[1] (total_cell_ids). The host's bind group layout for that
//     binding must say read-only storage.
//   * binding 2 is the DispatchArgs { x, y, z } struct, indexed 0, 1, 2.

const CHUNK_SIZE: u32 = 4;
const WORKGROUP_SIZE: u32 = 64;
const UNUSED_CELL_ID: u32 = 0xffffffff;

fn get_total_chunks(total_cell_ids: u32) -> u32 {
    (total_cell_ids + CHUNK_SIZE - 1) / CHUNK_SIZE
}

#[kernel(local_size = 64)]
fn count_objects_for_each_chunk(
    #[binding(0)] chunk_obj_count: &mut [u32],
    #[binding(3)] uniform_data: &[u32],
    #[binding(4)] cell_ids: &[u32],
) {
    let chunk_id: u32 = gpu_global_id(0);
    let total_cell_ids: u32 = uniform_data[1];

    if chunk_id >= get_total_chunks(total_cell_ids) {
        return;
    }

    // Get the first index of the chunk
    let first_idx: u32 = chunk_id * CHUNK_SIZE;

    // first_idx >= 1 ? cell_ids[first_idx - 1] : UNUSED_CELL_ID
    let mut prev_cell_id: u32 = UNUSED_CELL_ID;
    if first_idx >= 1 {
        prev_cell_id = cell_ids[(first_idx - 1) as usize];
    }

    let mut obj_count: u32 = 0;
    let mut currently_counting_cell: u32 = UNUSED_CELL_ID;
    let mut current_count: u32 = 0;

    let next_chunk_first_idx: u32 = first_idx + CHUNK_SIZE;

    // Count the number of objects in the chunk that share the same cell id
    let mut i: u32 = first_idx;
    while i < total_cell_ids {
        let cell_id: u32 = cell_ids[i as usize];

        // Exit conditions
        let is_out_of_bounds: bool = i >= next_chunk_first_idx;
        let is_cell_unused: bool = cell_id == UNUSED_CELL_ID;
        let is_it_a_transition: bool = cell_id != prev_cell_id;
        if (is_it_a_transition && is_out_of_bounds)
            || is_cell_unused
            || (current_count == 0 && is_out_of_bounds && !is_it_a_transition)
        {
            break;
        }

        // Counting condition
        let cell_was_seen_before: bool = currently_counting_cell == cell_id;
        if cell_was_seen_before {
            // The cell has more than one object inside, so the objects in
            // this cell have to be counted by this thread.
            if current_count == 1 {
                obj_count += 1;
            }
            current_count += 1;
        }

        if cell_id != prev_cell_id {
            // Transition: reset the current counter
            current_count = 1;
            currently_counting_cell = cell_id;
        }

        prev_cell_id = cell_id;
        i += 1;
    }

    // Write to memory the number of objects per chunk
    chunk_obj_count[chunk_id as usize] = obj_count;
}

#[kernel(local_size = 64)]
fn build_collision_cells_array(
    #[binding(0)] chunk_obj_count: &mut [u32],
    #[binding(1)] collision_cells: &mut [u32],
    #[binding(2)] indirect_args: &mut [u32],
    #[binding(3)] uniform_data: &[u32],
    #[binding(4)] cell_ids: &[u32],
) {
    let total_cell_ids: u32 = uniform_data[1];
    let chunk_id: u32 = gpu_global_id(0);

    if chunk_id == 0 {
        // Prepare the dispatch buffer: the total count is the last element
        // of the prefix sum buffer.
        let total_items: u32 = chunk_obj_count[(uniform_data[0] - 1) as usize];
        let workgroups_x: u32 = (total_items + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        indirect_args[0] = workgroups_x;
        indirect_args[1] = 1;
        indirect_args[2] = 1;
    }

    if chunk_id >= get_total_chunks(total_cell_ids) {
        return;
    }

    // Inclusive prefix sum of the chunk counts: where this chunk's output
    // starts and ends (the previous element is 0 for the first chunk).
    let mut start_index: u32 = 0;
    if chunk_id >= 1 {
        start_index = chunk_obj_count[(chunk_id - 1) as usize];
    }
    let end_index: u32 = chunk_obj_count[chunk_id as usize];
    let num_objects_to_manage: u32 = end_index - start_index;

    // Does this thread have work to do?
    if num_objects_to_manage == 0 {
        return;
    }

    // Create collision cells for every object that shares a cell id in the chunk
    let first_idx: u32 = chunk_id * CHUNK_SIZE;

    let mut prev_cell_id: u32 = UNUSED_CELL_ID;
    if first_idx >= 1 {
        prev_cell_id = cell_ids[(first_idx - 1) as usize];
    }

    let mut currently_counting_cell: u32 = UNUSED_CELL_ID;
    let mut current_count: u32 = 0;

    let next_chunk_first_idx: u32 = first_idx + CHUNK_SIZE;

    let mut write_index: u32 = start_index;

    let mut i: u32 = first_idx;
    while i < total_cell_ids && write_index != end_index {
        let cell_id: u32 = cell_ids[i as usize];

        // Exit conditions
        let is_out_of_bounds: bool = i >= next_chunk_first_idx;
        let is_cell_unused: bool = cell_id == UNUSED_CELL_ID;
        let is_it_a_transition: bool = cell_id != prev_cell_id;
        if (is_it_a_transition && is_out_of_bounds)
            || is_cell_unused
            || (current_count == 0 && is_out_of_bounds && !is_it_a_transition)
        {
            break;
        }

        // Counting condition
        let cell_was_seen_before: bool = currently_counting_cell == cell_id;
        if cell_was_seen_before {
            if current_count == 1 {
                // The previous object shared the same cell id: store the
                // first position of the cell.
                collision_cells[write_index as usize] = i - 1;
                write_index += 1;
            }
            current_count += 1;
        }

        if cell_id != prev_cell_id {
            current_count = 1;
            currently_counting_cell = cell_id;
        }

        prev_cell_id = cell_id;
        i += 1;
    }
}
