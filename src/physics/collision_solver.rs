use wgpu::{BindGroupLayout, PushConstantRange};
use wgpu_profiler::GpuProfiler;
use crate::grid::grid::{Grid, MAX_CELLS_PER_OBJECT};
use crate::particles::particle_system::ParticleSystem;
use crate::physics::collision_cell_builder::{CollisionCellBuilder};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;

const WORKGROUP_SIZE: u32 = 64;

pub struct CollisionSolver {
    collision_solver_shader: ComputeShader,
    bind_resources: BindResources,
    uniform_data: GpuBuffer<UniformData>
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CellColor{
    color: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformData {
    num_counting_chunks: u32,
    total_cell_ids: u32, 
}

impl CollisionSolver {
    pub fn new(wgpu_context: &WgpuContext, particle_system: &ParticleSystem, grid: &Grid, collision_cell_builder: &CollisionCellBuilder) -> Self {
        let uniform_data = GpuBuffer::new(
            wgpu_context,
            vec![UniformData{
                num_counting_chunks: collision_cell_builder.get_num_counting_chunks(),
                total_cell_ids: particle_system.len() as u32 * MAX_CELLS_PER_OBJECT,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        
        let bind_resources = Self::create_bind_resources(wgpu_context, particle_system, grid, collision_cell_builder, &uniform_data);
        
        let collision_solver_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("collision_solver.wgsl"),
            "solve_collisions",
            &bind_resources.bind_group_layout,
            (WORKGROUP_SIZE, 1, 1),
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE as f64),
            ],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<CellColor>() as u32,
                }
            ]
        );
        
        Self {
            collision_solver_shader,
            bind_resources,
            uniform_data,
        }
    }

    pub fn refresh_buffers(&mut self, wgpu_context: &WgpuContext, particle_system: &ParticleSystem, grid: &Grid, collision_cell_builder: &CollisionCellBuilder) {
        let new_uniform = UniformData {
            num_counting_chunks: collision_cell_builder.get_num_counting_chunks(),
            total_cell_ids: grid.cell_ids().len() as u32,
        };
        
        self.uniform_data.replace_elem(new_uniform, 0, wgpu_context);
        
        let bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, particle_system, grid, collision_cell_builder, &self.uniform_data);
        self.bind_resources.bind_group = bind_group;
    }
    
    fn create_bind_resources(wgpu_context: &WgpuContext, particle_system: &ParticleSystem, grid: &Grid, collision_cell_builder: &CollisionCellBuilder, uniform_data: &GpuBuffer<UniformData>) -> BindResources {
        let bind_group_layout = Self::create_bind_group_layout(wgpu_context);
        let bind_group = Self::create_bind_group(wgpu_context, &bind_group_layout, particle_system, grid, collision_cell_builder, uniform_data);
        BindResources {
            bind_group,
            bind_group_layout,
        }
    }
    
    fn create_bind_group(wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout, particle_system: &ParticleSystem, grid: &Grid, collision_cell_builder: &CollisionCellBuilder, uniform_data: &GpuBuffer<UniformData>) -> wgpu::BindGroup {
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: collision_cell_builder.chunk_obj_count().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: collision_cell_builder.collision_cells().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid.cell_ids().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid.object_ids().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: particle_system.positions().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: particle_system.radius().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: uniform_data.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }
    
    fn create_bind_group_layout(wgpu_context: &WgpuContext) -> BindGroupLayout{
        let compute_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Collision solver bind group layout"),
            entries: &[
                // Chunk obj count
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
                // Collision cells
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true},
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell IDs
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
                // Object IDs
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
                // Positions
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
                // Radius
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform data
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        wgpu_context.get_device().create_bind_group_layout(&compute_bind_group_layout)
    }
    


    /// Step 4: Solves collisions between objects in the same cell.
    pub fn solve_collisions(&mut self, wgpu_context: &WgpuContext, gpu_profiler: &mut GpuProfiler, indirect_dispatch_buffer: &GpuBuffer<u32>){
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Collision Encoder Color") }
        );
        
        for color in 1u32..=4u32 {
            
            let scope_label = format!("Solve Collisions - Color {}", color);
            
            {
                let mut scope = gpu_profiler.scope(scope_label, &mut encoder);

                self.collision_solver_shader.indirect_dispatch(
                    &mut scope,
                    indirect_dispatch_buffer.buffer(),
                    0,
                    Some(vec![(0u32, bytemuck::bytes_of(&CellColor {
                        color
                    }))]),
                    &self.bind_resources.bind_group
                );
            }
            gpu_profiler.resolve_queries(&mut encoder);
        }
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
    }
}