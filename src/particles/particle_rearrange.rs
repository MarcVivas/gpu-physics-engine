use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, PushConstantRange};
use wgpu_profiler::GpuProfiler;
use crate::particles::particle_buffers::ParticleBuffers;
use crate::particles::particle_sort::WORKGROUP_SIZE;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;

// Rearrange particles after they have been sorted
pub struct ParticleRearrangeKernel {
    rearrange_pass: ComputeShader,
    bind_resources: BindResources,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstantData{
    num_particles: u32,
}


impl ParticleRearrangeKernel {
    pub fn new(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>, particle_copy_buffers: &ParticleBuffers) -> Self {
        let bind_resources = Self::create_bind_resources(wgpu_context, particle_buffers, particle_ids, particle_copy_buffers);
        let rearrange_pass = Self::create_rearrange_pass(wgpu_context, &bind_resources);

        Self {
            bind_resources,
            rearrange_pass,       
        }
    }

    fn create_bind_resources(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>, particle_copy_buffers: &ParticleBuffers) -> BindResources {
        let bind_group_layout = Self::create_bind_group_layout(wgpu_context);
        let bind_group = Self::create_bind_group(wgpu_context, &bind_group_layout, particle_buffers, particle_ids, particle_copy_buffers);
        BindResources {
            bind_group_layout,
            bind_group
        }
    }

    fn create_bind_group_layout(wgpu_context: &WgpuContext) -> BindGroupLayout {
        let compute_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle rearrange Binding Group Layout"),
            entries: &[
                // Positions read
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Radius read
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Previous positions read
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Positions write
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Radius writing
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Previous positions writing
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        wgpu_context.get_device().create_bind_group_layout(&compute_bind_group_layout)
    }

    fn create_bind_group(wgpu_context: &WgpuContext, binding_group_layout: &BindGroupLayout, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>, particle_copy_buffers: &ParticleBuffers) -> BindGroup {
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: binding_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers.current_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers.radii.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers.previous_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: particle_ids.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: particle_copy_buffers.current_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: particle_copy_buffers.radii.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: particle_copy_buffers.previous_positions.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }
    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>, particle_copy_buffers: &ParticleBuffers){
        self.bind_resources.bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, particle_buffers, particle_ids, particle_copy_buffers);
    }

    /// Creates the rearranging kernel
    fn create_rearrange_pass(wgpu_context: &WgpuContext, binding_group: &BindResources) -> ComputeShader {
        ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("rearrange.wgsl"),
            "rearrange",
            &binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<PushConstantData>() as u32
                }
            ]
        )
    }
    
    pub fn rearrange(&self, encoder: &mut CommandEncoder, gpu_profiler: &mut GpuProfiler, particle_buffers: &ParticleBuffers, particle_copy_buffers: &ParticleBuffers){
        let num_particles = particle_buffers.current_positions.len() as u32;
        
        {
            let mut scope = gpu_profiler.scope("Particle rearranging", encoder);
            self.rearrange_pass.dispatch_by_items(
                &mut scope,
                (num_particles, 1, 1),
                Some(vec![(0u32, bytemuck::bytes_of(&PushConstantData {
                    num_particles,
                }))]),
                &self.bind_resources.bind_group
            );
        }
        
        // Not using ping pong buffers because it would complicate the code
        // Copy the buffers back to the original buffers
        {
            let mut scope = gpu_profiler.scope("Particle position rearranging copy", encoder);
            scope.copy_buffer_to_buffer(
                particle_copy_buffers.current_positions.buffer(),
                0,
                particle_buffers.current_positions.buffer(),
                0,
                particle_copy_buffers.current_positions.buffer().size(),
            );
        }

        {
            let mut scope = gpu_profiler.scope("Particle radii rearranging copy", encoder);
            scope.copy_buffer_to_buffer(
                particle_copy_buffers.radii.buffer(),
                0,
                particle_buffers.radii.buffer(),
                0,
                particle_copy_buffers.radii.buffer().size(),
            );
        }

        {
            let mut scope = gpu_profiler.scope("Particle previous position rearranging copy", encoder);
            scope.copy_buffer_to_buffer(
                particle_copy_buffers.previous_positions.buffer(),
                0,
                particle_buffers.previous_positions.buffer(),
                0,
                particle_copy_buffers.previous_positions.buffer().size(),
            );
        }
        
    }
}