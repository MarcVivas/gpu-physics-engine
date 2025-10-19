use crate::renderer::wgpu_context::WgpuContext;

pub mod gpu_buffer;
pub mod compute_shader;
pub mod radix_sort;
pub mod prefix_sum;
pub mod render_timer;
pub mod input_manager;
pub mod bind_resources;

/// Returns the maximum subgroup size of the GPU.
pub fn get_subgroup_size(wgpu_context: &WgpuContext) -> Option<u32> {
    Some(wgpu_context.get_adapter().limits().max_subgroup_size)
}
