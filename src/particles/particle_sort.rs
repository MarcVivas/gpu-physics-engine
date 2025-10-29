use std::num::NonZeroU32;
use wgpu_profiler::GpuProfiler;
use crate::particles::particle_buffers::ParticleBuffers;
use crate::particles::particle_home_cell_ids_kernel::ParticleHomeCellIdsKernel;
use crate::particles::particle_rearrange::ParticleRearrangeKernel;
use crate::particles::particle_system::ParticleSystem;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::utils::radix_sort::radix_sort::GPUSorter;



/// This struct sorts the particles by their home cell ID
/// The cell id is computed using morton encoding
pub struct ParticleSort {
    home_cell_ids_pass: ParticleHomeCellIdsKernel,
    rearrange_pass: ParticleRearrangeKernel,
    gpu_sorter: GPUSorter,
    particle_ids: GpuBuffer<u32>,
}



pub const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

impl ParticleSort{

    pub fn new(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_buffers_copy: &ParticleBuffers) -> Self {
        let particle_ids = (0u32..particle_buffers.home_cell_ids.len() as u32).collect();
        let particle_ids_buffer = GpuBuffer::new(wgpu_context, particle_ids, wgpu::BufferUsages::STORAGE);
        let home_cell_ids_pass = ParticleHomeCellIdsKernel::new(wgpu_context, particle_buffers, &particle_ids_buffer);
        let rearrange_pass = ParticleRearrangeKernel::new(wgpu_context, particle_buffers, &particle_ids_buffer, &particle_buffers_copy);
        let gpu_sorter = GPUSorter::new(wgpu_context, NonZeroU32::new(particle_buffers.home_cell_ids.len() as u32).unwrap(), &particle_buffers.home_cell_ids, &particle_ids_buffer);
        Self{rearrange_pass, gpu_sorter, particle_ids: particle_ids_buffer, home_cell_ids_pass}
    }

    
    
    
    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_buffers_copy: &ParticleBuffers) {
        self.home_cell_ids_pass.refresh(wgpu_context, particle_buffers, &self.particle_ids);
        self.rearrange_pass.refresh(wgpu_context, particle_buffers, &self.particle_ids, particle_buffers_copy);
        let prev_len = self.particle_ids.len() as u32;
        let curr_len = particle_buffers.home_cell_ids.len() as u32;
        let new_particle_ids: Vec<u32> = (prev_len..curr_len).collect();
        self.particle_ids.push_all(
            &new_particle_ids,
            wgpu_context
        );
        self.gpu_sorter.update_sorting_buffers(wgpu_context, NonZeroU32::new(self.particle_ids.len() as u32).unwrap(), &particle_buffers.home_cell_ids, &self.particle_ids);
    }
    
    


   
    
    pub fn sort(&self, encoder: &mut wgpu::CommandEncoder, gpu_profiler: &mut GpuProfiler, particle_system: &ParticleSystem, cell_size: f32) {
        // Compute the home cell ids using morton encoding
        self.home_cell_ids_pass.create_home_cell_ids(encoder, gpu_profiler, particle_system.len() as u32, cell_size);

        {
            // Sort the particles by their home cell id
            let mut scope = gpu_profiler.scope("Particle sort", encoder);
            self.gpu_sorter.sort(&mut scope, None);
        }
        // Rearrange the particles in the correct order
        self.rearrange_pass.rearrange(encoder, gpu_profiler, particle_system.buffers(), particle_system.copy_buffers());
    }

    pub fn download_particle_ids(&mut self, wgpu_context: &WgpuContext) -> Vec<u32>{
        self.particle_ids.download(wgpu_context).unwrap().clone()
    }
    
    
}