override WORKGROUP_SIZE: u32 = 256;
override RADIX_SORT_BUCKETS: u32 = 256;
override SUBGROUP_SIZE: u32 = 64;
override FLAGS_PER_BUCKET: u32 = WORKGROUP_SIZE / 32;

struct PushConstants {
    num_elements: u32,
    current_shift: u32,
    num_workgroups: u32,
    num_blocks_per_workgroup: u32,
}

var<push_constant> push_constants: PushConstants;
var<workgroup> shared_histogram: array<atomic<u32>, RADIX_SORT_BUCKETS>;

@group(0) @binding(0) var<storage, read_write> keys_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
@group(0) @binding(2) var<storage, read_write> payload_a: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> payload_b: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
)
{
    let local_idx = local_id.x;
    let workgroup_idx = workgroup_id.x;
    let global_idx = global_id.x;

    // Set to 0 the shared histogram values
    if local_id.x < RADIX_SORT_BUCKETS {
        atomicStore(&shared_histogram[local_idx], 0u);
    }
    workgroupBarrier();

    let num_blocks_per_workgroup = push_constants.num_blocks_per_workgroup;
    let num_elements = push_constants.num_elements;
    let current_shift = push_constants.current_shift;

    // Each workgroup processes MULTIPLE blocks/histograms
    for(var i: u32 = 0; i < num_blocks_per_workgroup; i++){
        let index = workgroup_idx * num_blocks_per_workgroup * WORKGROUP_SIZE + i * WORKGROUP_SIZE + local_idx;
        if index < num_elements {
            // Determine in which bucket the current portion of the number goes
            let bucket_id: u32 = (keys_a[index] >> current_shift) & (RADIX_SORT_BUCKETS - 1u);
            // Count 1
            atomicAdd(&shared_histogram[bucket_id], 1);
        }
    }
    workgroupBarrier();

    if local_idx < RADIX_SORT_BUCKETS {
        histogram[RADIX_SORT_BUCKETS * workgroup_idx + local_idx] = atomicLoad(&shared_histogram[local_idx]);
    }

}


override NUM_SUBGROUPS: u32 = WORKGROUP_SIZE / SUBGROUP_SIZE;


// Used for subgroup reductions
var<workgroup> shared_sums: array<u32, NUM_SUBGROUPS>;

// Exclusive prefix sum. Where each bucket starts in the final output
var<workgroup> shared_global_offsets: array<atomic<u32>, RADIX_SORT_BUCKETS>;

override TOTAL_BIN_FLAGS: u32 = RADIX_SORT_BUCKETS * FLAGS_PER_BUCKET;

// Per-bucket binary masks (used for local reordering inside workgroup)
var<workgroup> shared_bin_flags: array<atomic<u32>, TOTAL_BIN_FLAGS>;

var<workgroup> shared_bucket_counts:  array<u32, RADIX_SORT_BUCKETS>;
var<workgroup> shared_bucket_prefix:  array<u32, RADIX_SORT_BUCKETS>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn scatter_keys(    @builtin(global_invocation_id) g_id: vec3<u32>,
                    @builtin(local_invocation_id) l_id: vec3<u32>,
                    @builtin(workgroup_id) w_id: vec3<u32>,
                    @builtin(subgroup_invocation_id) sg_tid: u32,
               )
{

    let global_id = g_id.x;
    let local_id = l_id.x;
    let workgroup_id = w_id.x;
    let subgroup_tid = sg_tid;
    let num_workgroups = push_constants.num_workgroups;
    let num_blocks_per_workgroup = push_constants.num_blocks_per_workgroup;
    let num_elements = push_constants.num_elements;
    let current_shift = push_constants.current_shift;


    var local_histogram = 0u;
    var histogram_count = 0u;

    // STEP 1: Build global bucket counts and workgroup-local exclusive bases (deterministic)

    if (local_id < RADIX_SORT_BUCKETS) {
        // Exclusive prefix of this bucket over all workgroups (where this WG starts within the bucket)
        var accum = 0u;
        for (var i: u32 = 0u; i < num_workgroups; i++) {
            let bucket_value = histogram[i * RADIX_SORT_BUCKETS + local_id];
            if (i == workgroup_id) {
                local_histogram = accum; // exclusive: elems of earlier WGs in this bucket
            }
            accum += bucket_value;
        }
        histogram_count = accum;                         // total count for this bucket
        shared_bucket_counts[local_id] = histogram_count; // stash for WG-wide scan
    }
    workgroupBarrier();


    // Do a WG-wide exclusive scan over buckets in a deterministic order
    if (local_id == 0u) {
        var acc = 0u;
        for (var b: u32 = 0u; b < RADIX_SORT_BUCKETS; b++) {
            shared_bucket_prefix[b] = acc;       // exclusive start of bucket b
            acc += shared_bucket_counts[b];
        }
    }
    workgroupBarrier();

    if (local_id < RADIX_SORT_BUCKETS) {
        // Global base for this bucket slice owned by this workgroup
        let base = shared_bucket_prefix[local_id] + local_histogram;
        atomicStore(&shared_global_offsets[local_id], base);
    }


    // Step 2: Scatter elements
    let flag_offset = local_id / 32u;
    let flags_bit = 1u << (local_id % 32u);

    // For each block of elements
    for(var i: u32 = 0; i < num_blocks_per_workgroup; i++){
        // Get the element index
        let index = workgroup_id * num_blocks_per_workgroup * WORKGROUP_SIZE + i * WORKGROUP_SIZE + local_id;

        // Initialize bin flags to 0
        if local_id < RADIX_SORT_BUCKETS {
            // For each flag in the bucket
            for(var j: u32 = 0; j < FLAGS_PER_BUCKET; j++){
                let bin_flag_id = local_id * FLAGS_PER_BUCKET + j;
                atomicStore(&shared_bin_flags[bin_flag_id], 0u);
            }
        }
        workgroupBarrier();

        var element: u32 = 0;
        var payload: u32 = 0;
        var bucket_id: u32 = 0;
        var bucket_offset: u32 = 0;
        if index < num_elements {
            element = keys_a[index];
            payload = payload_a[index];
            bucket_id = (element >> current_shift) & (RADIX_SORT_BUCKETS - 1u);
            bucket_offset = atomicLoad(&shared_global_offsets[bucket_id]);
            let bin_flag_id = bucket_id * FLAGS_PER_BUCKET + flag_offset;
            atomicOr(&shared_bin_flags[bin_flag_id], flags_bit);
        }
        workgroupBarrier();

        if index < num_elements {
            var prefix = 0u;
            var count = 0u;
            for(var j: u32 = 0; j < FLAGS_PER_BUCKET; j++){
                let bin_flag_id = bucket_id * FLAGS_PER_BUCKET + j;
                let bits = atomicLoad(&shared_bin_flags[bin_flag_id]);
                let full_count = countOneBits(bits);
                let partial_count = countOneBits(bits & (flags_bit - 1u));
                prefix += select(0u, full_count, j < flag_offset);
                prefix += select(0u, partial_count, j == flag_offset);
                count += full_count;
            }
            keys_b[bucket_offset + prefix] = element;
            payload_b[bucket_offset + prefix] = payload;
            if prefix == count - 1 {
                atomicAdd(&shared_global_offsets[bucket_id], count);
            }
        }
        workgroupBarrier();
    }

}

// The following piece of code does not work and I don't know why.
// It should be faster than the current.

    /*
    // STEP 1: Compute global histogram and
    if local_id < RADIX_SORT_BUCKETS {
        var accum = 0u;

        // For each workgroup historgrams (all of them)
        for(var i: u32 = 0; i < num_workgroups; i++){
            // Get the current histogram
            let histogram_id = RADIX_SORT_BUCKETS * i;
            let histogram_bucket_id = histogram_id + local_id;
            // Get the value of the bucket in the current histogram
            let bucket_value = histogram[histogram_bucket_id];
            // Store the accumulated value if i == workgroup_id (Exclusive prefix sum)
            // This stores the number of elements stored before in bucket[local_id]
            // Where should the value start within the bucket
            local_histogram = select(local_histogram, accum, i == workgroup_id);

            accum += bucket_value;
        }
        // Once the loop finishes each workgroup will have the global histogram in local memory
        // This value is equal to the total global count of elements in the bucket[local_id]
        histogram_count = accum;
        // Compute how many elements are in the subgroup, the sum of the global counts
        let sum = subgroupAdd(histogram_count);
        // Compute the starting index of each of the buckets in the subgroup
        bucket_start_idx_subgroup = subgroupExclusiveAdd(histogram_count);
        // The subgroup leader stores to shared memory the total sum of the subgroup
        if subgroup_tid == 0 {
            shared_sums[subgroup_id] = sum;
        }
    }
    workgroupBarrier();

        // subgroupShuffle(subgroupExclusiveAdd(shared_sums[subgroup_tid]), subgroup_id)

    if (local_id == 0u) {
        var acc = 0u;
        for (var s: u32 = 0u; s < NUM_SUBGROUPS; s++) {
            let t = shared_sums[s];
            shared_sums[s] = acc; // now holds exclusive prefix for subgroup s
            acc += t;
        }
    }
    workgroupBarrier();

    if local_id < RADIX_SORT_BUCKETS {
        // Compute number of items in earlier subgroups, which is the global starting index of the first bucket in the subgroup
        let subgroup_first_bucket_start_idx_global = shared_sums[subgroup_id];
        // Store the base index of the bucket. Remember: each local_id has its own bucket.
        let global_histogram_base_idx = subgroup_first_bucket_start_idx_global + bucket_start_idx_subgroup;
        // Store the global start index for this slice of the bucket
        atomicStore(&shared_global_offsets[local_id], global_histogram_base_idx + local_histogram);
    }
    workgroupBarrier();
    */