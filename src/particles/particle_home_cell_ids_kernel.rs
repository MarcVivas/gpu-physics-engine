use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, PushConstantRange};
use wgpu_profiler::GpuProfiler;
use crate::particles::particle_buffers::ParticleBuffers;
use crate::particles::particle_sort::WORKGROUP_SIZE;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct ParticleHomeCellIdsKernel {
    bind_resources: BindResources,
    home_cell_ids_pass: ComputeShader,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstantData{
    num_particles: u32,
    cell_size: f32,
}

impl ParticleHomeCellIdsKernel {
    pub fn new(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids_buffer: &GpuBuffer<u32>) -> Self {
        let bind_resources = Self::create_bind_resources(wgpu_context, particle_buffers, &particle_ids_buffer);
        let home_cell_ids_pass = Self::create_home_cell_ids_pass(wgpu_context, &bind_resources);

        Self {
            bind_resources,
            home_cell_ids_pass
        }
    }

    /// Creates the home cell ids kernel
    fn create_home_cell_ids_pass(wgpu_context: &WgpuContext, binding_group: &BindResources) -> ComputeShader {
        ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("home_cell_ids.wgsl"),
            "create_home_cell_ids",
            &binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
            ],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<PushConstantData>() as u32
                }
            ]
        )
    }

    fn create_bind_resources(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>) -> BindResources {
        let bind_group_layout = Self::create_bind_group_layout(wgpu_context);
        let bind_group = Self::create_bind_group(wgpu_context, &bind_group_layout, particle_buffers, particle_ids);
        BindResources {
            bind_group_layout,
            bind_group
        }
    }

    fn create_bind_group_layout(wgpu_context: &WgpuContext) -> BindGroupLayout {
        let compute_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Home cell ids Binding Group Layout"),
            entries: &[
                // Positions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Home cell IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle IDs
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

    fn create_bind_group(wgpu_context: &WgpuContext, binding_group_layout: &BindGroupLayout, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>) -> BindGroup {
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
                        resource: particle_buffers.home_cell_ids.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_ids.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }
    
    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, particle_ids: &GpuBuffer<u32>) {
        self.bind_resources.bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, particle_buffers, particle_ids);
    }
    
    pub fn create_home_cell_ids(&self, encoder: &mut CommandEncoder, gpu_profiler: &mut GpuProfiler, num_particles: u32, cell_size: f32) {
        {
            let mut scope = gpu_profiler.scope("Particle home cells", encoder);
            self.home_cell_ids_pass.dispatch_by_items(
                &mut scope,
                (num_particles, 1, 1),
                Some(vec![(0u32, bytemuck::bytes_of(&PushConstantData {
                    num_particles,
                    cell_size
                }))]),
                &self.bind_resources.bind_group
            );
        }
    }
}