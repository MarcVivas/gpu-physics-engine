use std::num::NonZeroU32;
use wgpu::wgt::PollType::WaitForSubmissionIndex;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::utils::radix_sort::radix_sort::GPUSorter;

pub mod gpu_buffer;
pub mod compute_shader;
pub mod radix_sort;
pub mod prefix_sum;
pub(crate) mod gpu_timer;
pub mod render_timer;
pub mod input_manager;
pub mod bind_resources;

/// Returns the maximum subgroup size of the GPU.
pub fn get_subgroup_size(wgpu_context: &WgpuContext) -> Option<u32> {
    Some(wgpu_context.get_adapter().limits().max_subgroup_size)
}
