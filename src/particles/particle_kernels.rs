use std::num::NonZeroU32;
use wgpu::PushConstantRange;
use crate::particles::particle_buffers::ParticleBuffers;
use crate::particles::particle_push_constants::{SimParams};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::radix_sort::radix_sort::GPUSorter;

pub(crate) struct ParticleKernels {
    
    pub integration_pass: ComputeShader,
    pub gpu_sorter: GPUSorter,
}



impl ParticleKernels {
    pub fn new(wgpu_context: &WgpuContext, num_particles: usize, particle_binding_group: &BindResources, particle_buffers: &ParticleBuffers) -> Self {
        let integration_pass = Self::create_integration_pass(wgpu_context, particle_binding_group);
        let gpu_sorter = Self::create_gpu_sorter(wgpu_context, num_particles as u32, particle_buffers);
        Self {
            integration_pass,
            gpu_sorter,
        }
    }

    /// Creates the integration kernel
    fn create_integration_pass(wgpu_context: &WgpuContext, particle_binding_group: &BindResources) -> ComputeShader {
        ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("particle_system.wgsl"),
            "verlet_integration",
            &particle_binding_group.bind_group_layout,
            crate::particles::particle_system::WORKGROUP_SIZE,
            &vec![],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<SimParams>() as u32
                }
            ]
        )
    }
    
    
    /// Creates the GPU sorter for the particles (key-value radix sort)
    fn create_gpu_sorter(wgpu_context: &WgpuContext, total_particles: u32, particle_buffers: &ParticleBuffers) -> GPUSorter {
        GPUSorter::new(wgpu_context, NonZeroU32::new(total_particles).unwrap(), &particle_buffers.cell_ids, &particle_buffers.particle_ids)
    }
}