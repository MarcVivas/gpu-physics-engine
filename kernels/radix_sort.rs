// Port of src/utils/radix_sort/radix_sort.wgsl: one pass of a stable
// least-significant-digit radix sort (8 bits per pass) of u32 keys with a
// u32 payload. Two kernels per pass:
//
//   build_histogram  every workgroup counts the digits of its slice of
//                    keys_a into histogram[workgroup * 256 + digit]
//   scatter_keys     every workgroup works out where each digit's run starts
//                    (histograms of all workgroups, then a scan over the 256
//                    digits) and writes its keys, in order, to keys_b/payload_b
//
// The counting and ranking use workgroup atomics (atomicAdd, atomicOr,
// atomicLoad, atomicStore) and countOneBits, which Crust lowers directly. The
// WGSL's unused `subgroup_invocation_id` input is not carried over.
//
// WORKGROUP_SIZE, RADIX_SORT_BUCKETS and FLAGS_PER_BUCKET are WGSL override
// constants; they are fixed at their defaults here (256, 256, 256 / 32 = 8).
//
// Both kernels declare all four PushConstants members in the WGSL order
// (num_elements, current_shift, num_workgroups, num_blocks_per_workgroup),
// so the host pushes the same 16 bytes it always did.

const WORKGROUP_SIZE: u32 = 256;
const RADIX_SORT_BUCKETS: u32 = 256;
const FLAGS_PER_BUCKET: u32 = 8;

#[workgroup]
static mut SHARED_HISTOGRAM: [u32; 256] = [0; 256];
#[workgroup]
static mut SHARED_GLOBAL_OFFSETS: [u32; 256] = [0; 256];
#[workgroup]
static mut SHARED_BIN_FLAGS: [u32; 2048] = [0; 2048];
#[workgroup]
static mut SHARED_BUCKET_COUNTS: [u32; 256] = [0; 256];
#[workgroup]
static mut SHARED_BUCKET_PREFIX: [u32; 256] = [0; 256];

#[kernel(local_size = 256)]
fn build_histogram(
    #[binding(0)] keys_a: &[u32],
    #[binding(1)] histogram: &mut [u32],
    num_elements: u32,
    current_shift: u32,
    num_workgroups: u32,
    num_blocks_per_workgroup: u32,
) {
    let local_idx: u32 = gpu_local_id(0);
    let workgroup_idx: u32 = gpu_group_id(0);

    // Set to 0 the shared histogram values
    unsafe {
        gpu_atomic_store(&mut SHARED_HISTOGRAM[local_idx as usize], 0);
    }
    gpu_barrier();

    // Each workgroup processes MULTIPLE blocks/histograms
    for i in 0..num_blocks_per_workgroup {
        let index: u32 = workgroup_idx * num_blocks_per_workgroup * WORKGROUP_SIZE
            + i * WORKGROUP_SIZE
            + local_idx;
        if index < num_elements {
            // Determine in which bucket the current portion of the number goes
            let bucket_id: u32 = (keys_a[index as usize] >> current_shift) & (RADIX_SORT_BUCKETS - 1);
            // Count 1
            unsafe {
                gpu_atomic_add(&mut SHARED_HISTOGRAM[bucket_id as usize], 1);
            }
        }
    }
    gpu_barrier();

    unsafe {
        histogram[(RADIX_SORT_BUCKETS * workgroup_idx + local_idx) as usize] =
            gpu_atomic_load(&mut SHARED_HISTOGRAM[local_idx as usize]);
    }
}

#[kernel(local_size = 256)]
fn scatter_keys(
    #[binding(0)] keys_a: &[u32],
    #[binding(1)] histogram: &[u32],
    #[binding(2)] payload_a: &[u32],
    #[binding(3)] keys_b: &mut [u32],
    #[binding(4)] payload_b: &mut [u32],
    num_elements: u32,
    current_shift: u32,
    num_workgroups: u32,
    num_blocks_per_workgroup: u32,
) {
    let local_id: u32 = gpu_local_id(0);
    let workgroup_id: u32 = gpu_group_id(0);

    // STEP 1: global bucket counts and this workgroup's exclusive base in
    // each bucket (deterministic).
    let mut local_histogram: u32 = 0;
    let mut accum: u32 = 0;
    for i in 0..num_workgroups {
        let bucket_value: u32 = histogram[(i * RADIX_SORT_BUCKETS + local_id) as usize];
        if i == workgroup_id {
            // exclusive: elements of earlier workgroups in this bucket
            local_histogram = accum;
        }
        accum += bucket_value;
    }
    // total count for this bucket, stashed for the workgroup-wide scan
    unsafe {
        SHARED_BUCKET_COUNTS[local_id as usize] = accum;
    }
    gpu_barrier();

    // Workgroup-wide exclusive scan over the buckets, in a fixed order.
    if local_id == 0 {
        let mut acc: u32 = 0;
        for b in 0..RADIX_SORT_BUCKETS {
            unsafe {
                SHARED_BUCKET_PREFIX[b as usize] = acc; // exclusive start of bucket b
                acc += SHARED_BUCKET_COUNTS[b as usize];
            }
        }
    }
    gpu_barrier();

    // Global base for the slice of each bucket owned by this workgroup
    unsafe {
        let base: u32 = SHARED_BUCKET_PREFIX[local_id as usize] + local_histogram;
        gpu_atomic_store(&mut SHARED_GLOBAL_OFFSETS[local_id as usize], base);
    }

    // STEP 2: scatter the elements, block by block
    let flag_offset: u32 = local_id / 32;
    let flags_bit: u32 = 1u32 << (local_id % 32);

    for i in 0..num_blocks_per_workgroup {
        let index: u32 = workgroup_id * num_blocks_per_workgroup * WORKGROUP_SIZE
            + i * WORKGROUP_SIZE
            + local_id;

        // Initialize this thread's bin flags to 0
        for j in 0..FLAGS_PER_BUCKET {
            unsafe {
                gpu_atomic_store(&mut SHARED_BIN_FLAGS[(local_id * FLAGS_PER_BUCKET + j) as usize], 0);
            }
        }
        gpu_barrier();

        let mut element: u32 = 0;
        let mut payload: u32 = 0;
        let mut bucket_id: u32 = 0;
        let mut bucket_offset: u32 = 0;
        if index < num_elements {
            element = keys_a[index as usize];
            payload = payload_a[index as usize];
            bucket_id = (element >> current_shift) & (RADIX_SORT_BUCKETS - 1);
            unsafe {
                bucket_offset = gpu_atomic_load(&mut SHARED_GLOBAL_OFFSETS[bucket_id as usize]);
                gpu_atomic_or(&mut SHARED_BIN_FLAGS[(bucket_id * FLAGS_PER_BUCKET + flag_offset) as usize], flags_bit);
            }
        }
        gpu_barrier();

        if index < num_elements {
            let mut prefix: u32 = 0;
            let mut count: u32 = 0;
            for j in 0..FLAGS_PER_BUCKET {
                let mut bits: u32 = 0;
                unsafe {
                    bits = gpu_atomic_load(&mut SHARED_BIN_FLAGS[(bucket_id * FLAGS_PER_BUCKET + j) as usize]);
                }
                let full_count: u32 = bits.count_ones();
                let partial_count: u32 = (bits & (flags_bit - 1)).count_ones();
                if j < flag_offset {
                    prefix += full_count;
                }
                if j == flag_offset {
                    prefix += partial_count;
                }
                count += full_count;
            }
            keys_b[(bucket_offset + prefix) as usize] = element;
            payload_b[(bucket_offset + prefix) as usize] = payload;
            if prefix == count - 1 {
                unsafe {
                    gpu_atomic_add(&mut SHARED_GLOBAL_OFFSETS[bucket_id as usize], count);
                }
            }
        }
        gpu_barrier();
    }
}
