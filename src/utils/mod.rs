use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::radix_sort::radix_sort::GPUSorter;

pub mod gpu_buffer;
pub mod compute_shader;
pub mod radix_sort;


/// Function guesses the best subgroup size by testing the sorter with
/// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
pub fn guess_workgroup_size(wgpu_context: &WgpuContext) -> Option<u32> {
    let mut current_sorter: GPUSorter;
    let device = wgpu_context.get_device();
    log::debug!("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

    let mut best = None;
    for subgroup_size in [1, 8, 16, 32, 64, 128] {
        log::debug!("Checking sorting with subgroupsize {}", subgroup_size);

        current_sorter = GPUSorter::new(device, subgroup_size);
        let sort_success = current_sorter.test_sort(wgpu_context);

        log::debug!("{} worked: {}", subgroup_size, sort_success);

        if !sort_success {
            break;
        } else {
            best = Some(subgroup_size)
        }
    }
    best
}
