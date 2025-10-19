use wgpu::{BindGroupLayout, CommandEncoder};
use wgpu_profiler::GpuProfiler;
use crate::grid::grid::{Grid, MAX_CELLS_PER_OBJECT, UNUSED_CELL_ID};
use crate::physics::collision_cell_buffers::CollisionCellBuffers;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::utils::prefix_sum::prefix_sum::PrefixSum;

const WORKGROUP_SIZE: (u32, u32, u32) = (64u32, 1u32, 1u32);
/// The value must match in the compute shader.
pub const COUNTING_CHUNK_SIZE: u32 = 4;

pub struct CollisionCellBuilder{
    bind_resources: BindResources,
    prefix_sum: PrefixSum,
    count_objects_per_chunk_shader: ComputeShader,
    build_collision_cells_shader: ComputeShader,
    collision_cell_buffers: CollisionCellBuffers,
    uniform_data: GpuBuffer<UniformData>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformData {
    num_counting_chunks: u32,
    total_cell_ids: u32,
}

impl CollisionCellBuilder{
    pub fn new(wgpu_context: &WgpuContext, total_particles: usize, dim: u32, grid: &Grid) -> Self {
        let buffer_len = total_particles * 2usize.pow(dim); // A particle can be in 2**dim different cells
        let collision_cell_buffers = CollisionCellBuffers::new(wgpu_context, buffer_len);

        let uniform_data = GpuBuffer::new(
            wgpu_context,
            vec![UniformData {
                total_cell_ids: grid.cell_ids().len() as u32,
                num_counting_chunks: Self::calc_num_counting_chunks(collision_cell_buffers.get_collision_cells().len() as u32),
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        
        let bind_resources = Self::create_bind_resources(wgpu_context, &collision_cell_buffers, &uniform_data, grid);
        
        let count_objects_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("collision_cell_builder.wgsl"),
            "count_objects_for_each_chunk",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
                ("MAX_CELLS_PER_OBJECT", MAX_CELLS_PER_OBJECT as f64),
                ("CHUNK_SIZE", COUNTING_CHUNK_SIZE as f64)
            ],
            &vec![]
        );

        let build_collision_cells_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("collision_cell_builder.wgsl"),
            "build_collision_cells_array",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
                ("MAX_CELLS_PER_OBJECT", MAX_CELLS_PER_OBJECT as f64),
                ("CHUNK_SIZE", COUNTING_CHUNK_SIZE as f64)
            ],
            &vec![]
        );

        let prefix_sum = PrefixSum::new(wgpu_context, &collision_cell_buffers.get_chunk_counting());

        Self {
            prefix_sum,
            count_objects_per_chunk_shader: count_objects_shader, 
            build_collision_cells_shader,
            collision_cell_buffers,
            bind_resources,
            uniform_data,
        }
    }
    
    fn create_bind_resources(wgpu_context: &WgpuContext, buffers: &CollisionCellBuffers, uniform_data: &GpuBuffer<UniformData>, grid: &Grid) -> BindResources {
        let bind_group_layout = Self::create_bind_group_layout(wgpu_context);
        let bind_group = Self::create_bind_group(wgpu_context, &bind_group_layout, buffers, uniform_data, grid);

        BindResources{
            bind_group_layout,
            bind_group, 
        }
    }
    
    fn create_bind_group(wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout, buffers: &CollisionCellBuffers, uniform_data: &GpuBuffer<UniformData>, grid: &Grid) -> wgpu::BindGroup {
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.get_chunk_counting().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.get_collision_cells().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.get_indirect_dispatch().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: grid.cell_ids().buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }
    
    fn create_bind_group_layout(wgpu_context: &WgpuContext) -> BindGroupLayout {
        let bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout Descriptor"),
            entries: &[
                // Binding 0: The chunk counting buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: The collision cells
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
                // Binding 2: The indirect dispatch buffer
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
                // Binding 3: The uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: The cell ids
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true},
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                
            ],
        };
        wgpu_context.get_device().create_bind_group_layout(&bind_group_layout_descriptor)
    }
    
    pub fn refresh_buffers(&mut self, wgpu_context: &WgpuContext, new_buffer_size: usize, grid: &Grid) {
        self.collision_cell_buffers.push_all_to_chunk_counting(wgpu_context, &vec![0; ((new_buffer_size as u32 + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE) as usize]);
        self.collision_cell_buffers.push_all_to_collision_cells(wgpu_context, &vec![UNUSED_CELL_ID; new_buffer_size]);
        self.prefix_sum.update_buffers(wgpu_context, &self.collision_cell_buffers.get_chunk_counting());

        let new_uniform = UniformData {
            num_counting_chunks: self.get_num_counting_chunks(),
            total_cell_ids: grid.cell_ids().len() as u32,
        };

        self.uniform_data.replace_elem(new_uniform, 0, wgpu_context);
        
        self.bind_resources.bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, &self.collision_cell_buffers, &self.uniform_data, grid);
    }

    /// Step 3: Builds the collision cell list.
    /// Key: cell id; Value: Object id
    /// Collision cells are cells that contain more than one object, and therefore they need to be checked for potential collisions 
    pub fn build_collision_cells(&self, wgpu_context: &WgpuContext,  encoder: &mut CommandEncoder, gpu_profiler: &mut GpuProfiler){
        let num_chunks = self.get_num_counting_chunks();

        // Step 3.1 Count the number of objects in each chunk that share the same cell id
        {
            let mut scope = gpu_profiler.scope("Collision cell count objects per chunk", encoder);
            self.count_objects_per_chunk_shader.dispatch_by_items(
                &mut scope,
                (num_chunks, 1, 1),
                None,
                &self.bind_resources.bind_group
            );
        }
        
        // Step 3.2 Prefix sums the number of objects in each chunk
        {
            let mut scope = gpu_profiler.scope("Collision cell prefix sum", encoder);
            self.prefix_sum.execute(wgpu_context, &mut scope, self.collision_cell_buffers.get_chunk_counting().len() as u32);
        }
        
        // Step 3.3 Build the collision cell list
        {
            let mut scope = gpu_profiler.scope("Build collision cells", encoder);
            self.build_collision_cells_shader.dispatch_by_items(&mut scope, (num_chunks, 1, 1), None, &self.bind_resources.bind_group);
        }
    }

    pub fn get_num_counting_chunks(&self) -> u32 {
        Self::calc_num_counting_chunks(self.collision_cell_buffers.get_collision_cells().len() as u32)
    }
    
    fn calc_num_counting_chunks(total_collision_cells: u32) -> u32 {
        (total_collision_cells + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE
    }

    pub fn download_collision_cells(&mut self, wgpu_context: &WgpuContext) -> Vec<u32> {
        self.collision_cell_buffers.download_collision_cells(wgpu_context)
    }
    
    pub fn collision_cells(&self) -> &GpuBuffer<u32> {
        self.collision_cell_buffers.get_collision_cells()
    }
    
    pub fn chunk_obj_count(&self) -> &GpuBuffer<u32> {
        self.collision_cell_buffers.get_chunk_counting()
    }

    pub fn indirect_dispatch_buffer(&self) -> &GpuBuffer<u32> {
        self.collision_cell_buffers.get_indirect_dispatch() 
    }
}