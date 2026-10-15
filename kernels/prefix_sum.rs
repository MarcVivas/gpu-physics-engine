// Port of src/utils/prefix_sum/prefix_sum.wgsl: a three-pass inclusive scan
// of a large u32 buffer (scan every block, scan the block sums, add each
// block's predecessor total back).
//
// Two deliberate differences from the WGSL, both invisible in the results:
//
//  * The WGSL scans inside a subgroup with subgroupInclusiveAdd and stitches
//    the subgroups together through shared memory. Subgroup size is a device
//    property and SPIR-V subgroup ops are not lowered yet, so this does the
//    whole 256-wide block scan in workgroup memory (Hillis-Steele, one
//    barrier pair per doubling step). It computes the same inclusive scan.
//
//  * The WGSL returns early for threads past `total_elems` and then calls
//    workgroupBarrier(); that is undefined where a barrier is not reached by
//    the whole workgroup, and Crust refuses it. Here every invocation
//    reaches every barrier, and only the memory accesses are guarded. The
//    block-sum write keeps the original condition, so a partial last block
//    still leaves its block_sums entry untouched.
//
// WORKGROUP_SIZE is fixed at 256 (the WGSL default; it is an override
// constant there). Push constant: total_elems (not used by the second pass).

#[workgroup]
static mut SCAN: [u32; 256] = [0; 256];
#[workgroup]
static mut PREVIOUS_BLOCK_SUM: u32 = 0;

/// Inclusive scan of SCAN[0..256), already filled in by every invocation.
/// All invocations must call this.
fn scan_block(l: u32) {
    let mut offset: u32 = 1;
    while offset < 256 {
        gpu_barrier();
        let mut add: u32 = 0;
        if l >= offset {
            unsafe {
                add = SCAN[(l - offset) as usize];
            }
        }
        gpu_barrier();
        unsafe {
            SCAN[l as usize] = SCAN[l as usize] + add;
        }
        offset = offset << 1;
    }
    gpu_barrier();
}

/// First pass: scan each 256-element block of `data`, and record each
/// block's total in `block_sums`.
#[kernel(local_size = 256)]
fn prefix_sum_of_each_block(
    #[binding(0)] data: &mut [u32],
    #[binding(1)] block_sums: &mut [u32],
    total_elems: u32,
) {
    let g: u32 = gpu_global_id(0);
    let l: u32 = gpu_local_id(0);

    let mut value: u32 = 0;
    if g < total_elems {
        value = data[g as usize];
    }
    unsafe {
        SCAN[l as usize] = value;
    }
    scan_block(l);

    let mut final_value: u32 = 0;
    unsafe {
        final_value = SCAN[l as usize];
    }
    if g < total_elems {
        // The last thread of the block holds the block total.
        if l == 255 {
            block_sums[gpu_group_id(0) as usize] = final_value;
        }
        data[g as usize] = final_value;
    }
}

/// Second pass: scan the block sums (one workgroup covers them all).
#[kernel(local_size = 256)]
fn prefix_sum_of_the_block_sums(
    #[binding(0)] data: &mut [u32],
    #[binding(1)] block_sums: &mut [u32],
) {
    let g: u32 = gpu_global_id(0);
    let l: u32 = gpu_local_id(0);

    unsafe {
        SCAN[l as usize] = block_sums[g as usize];
    }
    scan_block(l);

    unsafe {
        block_sums[g as usize] = SCAN[l as usize];
    }
}

/// Third pass: add the scanned total of all earlier blocks to every element.
#[kernel(local_size = 256)]
fn add_block_prefix_sums_to_the_buffer(
    #[binding(0)] data: &mut [u32],
    #[binding(1)] block_sums: &[u32],
    total_elems: u32,
) {
    let g: u32 = gpu_global_id(0);
    let l: u32 = gpu_local_id(0);
    let block_id: u32 = g / 256;

    // One thread loads the preceding block's total for the whole workgroup.
    // The first block has no predecessor, and a block entirely past the end
    // has nothing to add to.
    if l == 0 {
        if block_id > 0 {
            if g < total_elems {
                unsafe {
                    PREVIOUS_BLOCK_SUM = block_sums[(block_id - 1) as usize];
                }
            }
        }
    }
    gpu_barrier();

    if g < total_elems {
        if block_id > 0 {
            unsafe {
                data[g as usize] = data[g as usize] + PREVIOUS_BLOCK_SUM;
            }
        }
    }
}
