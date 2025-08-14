override WORKGROUP_SIZE: u32 = 256;
override SUBGROUP_SIZE: u32 = 64;
override SHARED_MEMORY_SIZE: u32 = 64;

var<workgroup> shared_data: array<u32, SHARED_MEMORY_SIZE>;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;

var<push_constant> total_elems: u32;

/// First pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum_of_each_block(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_thread_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    ){

    if global_id.x >= total_elems {return;}

    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = WORKGROUP_SIZE / subgroup_size;

    // Load value
    let value = get_value(global_id.x);


    // Subgroup prefix sum
    let thread_val = subgroupInclusiveAdd(value);

    // The last thread of the subgroup has the total sum of the subgroup prefix sum
    if subgroup_thread_id == subgroup_size - 1u {
        // Store the subgroup total sum in shared memory
        shared_data[subgroup_id] = thread_val;
    }
    workgroupBarrier();

    // The first subgroup does an exclusive prefix sum of the total sums of the subgroups
    if subgroup_id == 0 {
        let block_val = select(0, shared_data[subgroup_thread_id], subgroup_thread_id < num_subgroups);
        let prefix_val = subgroupExclusiveAdd(block_val);
        if subgroup_thread_id < num_subgroups {
            shared_data[subgroup_thread_id] = prefix_val;
        }
    }
    workgroupBarrier();

    // Calculate the final value with the subgroup val and the block val
    let final_value = thread_val + shared_data[subgroup_id];


    // Only the last thread of the workgroup
    if local_id.x == WORKGROUP_SIZE - 1 {
        // Store the total sum of the block to global memory
        block_sums[workgroup_id.x] = final_value;
    }


    // Write back to global memory
    data[global_id.x] = final_value;
}



fn get_value(idx: u32) -> u32 {
     return data[idx];
}

fn get_block_value(idx: u32) -> u32 {
    return block_sums[idx];
}


/// Second pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn prefix_sum_of_the_block_sums(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_thread_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    ){

    let subgroup_id = local_id.x / subgroup_size;
    let num_subgroups = WORKGROUP_SIZE / subgroup_size;

    // Load value
    let block_sum = get_block_value(global_id.x);


    // Subgroup prefix sum
    let thread_val = subgroupInclusiveAdd(block_sum);

    // The last thread of the subgroup has the total sum of the subgroup prefix sum
    if subgroup_thread_id == subgroup_size - 1u {
        // Store the subgroup total sum in shared memory
        shared_data[subgroup_id] = thread_val;
    }
    workgroupBarrier();

    // The first subgroup does an exclusive prefix sum of the total sums of the subgroups
    if subgroup_id == 0 {
        let block_val = select(0, shared_data[subgroup_thread_id], subgroup_thread_id < num_subgroups);
        let prefix_val = subgroupExclusiveAdd(block_val);
        if subgroup_thread_id < num_subgroups {
            shared_data[subgroup_thread_id] = prefix_val;
        }
    }
    workgroupBarrier();

    // Calculate the final value with the subgroup val and the block val
    let final_value = thread_val + shared_data[subgroup_id];



    // Write back to global memory
    block_sums[global_id.x] = final_value;

}

var<workgroup> previous_block_sum: u32;

/// Third pass
@compute @workgroup_size(WORKGROUP_SIZE)
fn add_block_prefix_sums_to_the_buffer(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    if global_id.x >= total_elems {return;} // Out of bounds

    let block_id = global_id.x / WORKGROUP_SIZE;

    // No need to compute the first block, as it does not have a preceding block
    if block_id == 0 {return;}

    // One thread loads the data to be read to shared memory
    if local_id.x == 0 {
        previous_block_sum = block_sums[block_id - 1];
    }
    workgroupBarrier();

    data[global_id.x] += previous_block_sum;
}