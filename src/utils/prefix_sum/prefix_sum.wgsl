// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

const WORKGROUP_SIZE = 256u;
const ELEMENTS_PER_THREAD = 2u;
const BLOCK_ELEMENTS = (WORKGROUP_SIZE * ELEMENTS_PER_THREAD);
const WORKGROUPS_PER_BLOCK = BLOCK_ELEMENTS / WORKGROUP_SIZE;

var<workgroup> shared_data: array<u32, BLOCK_ELEMENTS>;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> num_elems: u32;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

/// First pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum_of_each_block(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    ){

    // Each thread works with 2 elements
    // Get the elements indexes and values.
    let value_1_idx = workgroup_id.x * BLOCK_ELEMENTS + local_id.x;
    let value_2_idx = value_1_idx + WORKGROUP_SIZE;
    let value_1 = get_value(workgroup_id.x * BLOCK_ELEMENTS + local_id.x);
    let value_2 = get_value(value_2_idx);

    // Load the elements into shared memory
    shared_data[local_id.x] = value_1;
    shared_data[local_id.x + WORKGROUP_SIZE] = value_2;
    workgroupBarrier();

    upsweep_phase(local_id.x);


    // Only the last thread of the workgroup
    if local_id.x == WORKGROUP_SIZE - 1 {
        // Store the total sum of the block to global memory
        block_sums[workgroup_id.x] = shared_data[BLOCK_ELEMENTS - 1];
        // Reset for the down sweep phase
        shared_data[BLOCK_ELEMENTS - 1] = 0;
    }
    workgroupBarrier();

    downsweep_phase(local_id.x);


    // Prefix sum of each block completed and stored in shared memory

    // Write back to global memory
    data[value_1_idx] = shared_data[local_id.x] + value_1;
    data[value_2_idx] = shared_data[local_id.x + WORKGROUP_SIZE] + value_2;

    // This only does the prefix sum of the values within the WORKGROUP BLOCK.
    // The results have to be combined in another shader.
}

fn upsweep_phase(local_id: u32){
    // Up-sweep phase (reduction)
    // Computes partial sums.
    // The last element in shared_data will hold the total sum of the block.
    var stride:u32 = 1u;
    for (var depth: u32 = 0; depth < log2_u32(BLOCK_ELEMENTS); depth++){
        stride = 1u << depth;  // stride = 2^depth

        let write_index: u32 = (local_id + 1u) * (stride * 2u) - 1u;
        let read_index: u32 = write_index - stride;

        if write_index < BLOCK_ELEMENTS {
            shared_data[write_index] += shared_data[read_index];
        }
        workgroupBarrier();
    }
}

fn downsweep_phase(local_id: u32){
    // Down sweep phase
    // The same loop but reversed
    for (var depth: i32 = i32(log2_u32(BLOCK_ELEMENTS)) - 1; depth >= 0; depth--){
        let stride:u32 = 1u << u32(depth); // stride = 2^depth

        let write_index: u32 = (local_id + 1u) * (stride * 2u) - 1u;
        let read_index: u32 = write_index - stride;

        if write_index < BLOCK_ELEMENTS {
            let tmp = shared_data[read_index];
            shared_data[read_index] = shared_data[write_index];
            shared_data[write_index] += tmp;
        }

        workgroupBarrier();
    }
}

fn get_value(idx: u32) -> u32 {
    if (idx < num_elems){
        return data[idx];
    }
    return 0;
}

fn get_block_value(idx: u32) -> u32 {
    if (idx < u32(ceil(f32(num_elems)/f32(BLOCK_ELEMENTS))) ){
        return block_sums[idx];
    }
    return 0;
}

fn log2_u32(n: u32) -> u32 {
    if (n == 0u) {
        // Handle 0 case (log2(0) is undefined, typically -infinity)
        return 0u;
    }
    var count = 0u;
    var val = n;
    if (val >= 1u << 16u) { val >>= 16u; count += 16u; }
    if (val >= 1u << 8u)  { val >>= 8u;  count += 8u;  }
    if (val >= 1u << 4u)  { val >>= 4u;  count += 4u;  }
    if (val >= 1u << 2u)  { val >>= 2u;  count += 2u;  }
    if (val >= 1u << 1u)  {              count += 1u;  } // If val is 2 or 3
    return count;
}

/// Second pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum_of_the_block_sums(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    ){
    // Each thread works with 2 elements
    // Get the elements indexes and values.
    let value_1_idx = workgroup_id.x * BLOCK_ELEMENTS + local_id.x;
    let value_2_idx = value_1_idx + WORKGROUP_SIZE;
    let value_1 = get_block_value(workgroup_id.x * BLOCK_ELEMENTS + local_id.x);
    let value_2 = get_block_value(value_2_idx);

    // Load the elements into shared memory
    shared_data[local_id.x] = value_1;
    shared_data[local_id.x + WORKGROUP_SIZE] = value_2;
    workgroupBarrier();

    upsweep_phase(local_id.x);

    // Only the last thread of the workgroup
    if local_id.x == WORKGROUP_SIZE - 1 {
        // Reset for the down sweep phase
        shared_data[BLOCK_ELEMENTS - 1] = 0;
    }

    downsweep_phase(local_id.x);

     // Prefix sum of each block completed and stored in shared memory

     // Write back to global memory
     block_sums[value_1_idx] = shared_data[local_id.x] + value_1;
     block_sums[value_2_idx] = shared_data[local_id.x + WORKGROUP_SIZE] + value_2;
}

var<workgroup> previous_block_sum: u32;

/// Third pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn add_block_prefix_sums_to_the_buffer(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    if global_id.x >= num_elems {return;} // Out of bounds

    let block_id = workgroup_id.x / WORKGROUPS_PER_BLOCK;

    // No need to compute the first block, as it does not have a preceding block
    if block_id == 0 {return;}

    // One thread loads the data to be read to shared memory
    if local_id.x == 0 {
        previous_block_sum = block_sums[block_id - 1];
    }
    workgroupBarrier();

    data[global_id.x] += previous_block_sum;
}