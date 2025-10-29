use glam::{Vec2};
use crate::particles::particle_system::ParticleSystem;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;
use std::num::NonZeroU32;
use wgpu::{BindGroupLayout, BufferAsyncError, CommandEncoder, PushConstantRange};
use wgpu_profiler::GpuProfiler;
use crate::grid::grid_drawer::GridDrawer;
use crate::utils::bind_resources::BindResources;
use crate::utils::radix_sort::radix_sort::{GPUSorter};

/// The value must match in the compute shader.
const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

pub const MAX_CELLS_PER_OBJECT: u32 = 4;

const CELL_SIZE_MULTIPLIER: f32 = 2.2f32;

pub const UNUSED_CELL_ID: u32 = u32::MAX;


pub struct Grid {
    grid_drawer: Option<GridDrawer>,
    should_draw_grid: bool,
    dim: u32,
    grid_buffers: GridBuffers,
    grid_kernels: GridKernels,
    grid_binding_group: BindResources,
    cell_size: f32,
    num_elements: usize,
}

struct GridBuffers{
    cell_ids: GpuBuffer<u32>, // Indicates the cells an object is in. cell_ids[i..i+3] = cell_id_of_object_i
    object_ids: GpuBuffer<u32>, // Need this after sorting to indicate the objects in a cell.
    uniform_buffer: GpuBuffer<UniformData>,
}

struct GridKernels {
    build_cell_ids_shader: ComputeShader,
    gpu_sorter: GPUSorter,
}



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformData {
    num_particles: u32,
    num_collision_cells: u32,
    cell_size: f32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstantsBuildGrid {
    cell_size: f32,
    num_particles: u32,
}

impl Grid {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, particle_system: &ParticleSystem) -> Grid {
        let max_obj_radius = particle_system.get_max_radius();
        let mut grid = Self::new_without_camera(wgpu_context, max_obj_radius, particle_system);
        grid.grid_drawer = Some(GridDrawer::new(wgpu_context, camera, &world_dimensions, grid.cell_size));
        grid
    }

    // No camera needed for tests
    pub fn new_without_camera(wgpu_context: &WgpuContext, max_obj_radius: f32, particle_system: &ParticleSystem) -> Grid{
        let total_particles: usize = particle_system.len();
        let dim: u32 = 2;
        let buffer_len = total_particles * 2usize.pow(dim); // A particle can be in 2**dim different cells
        let cell_size = Self::compute_cell_size(max_obj_radius);
        
        let cell_ids = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; buffer_len],
            wgpu::BufferUsages::STORAGE);
        
        let object_ids = GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_len],
            wgpu::BufferUsages::STORAGE
        );
        
        let uniform_buffer = GpuBuffer::new(
            wgpu_context,
            vec![UniformData {
                num_collision_cells: 0u32,
                num_particles: total_particles as u32,
                cell_size,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        
        let grid_buffers = GridBuffers {
            cell_ids,
            object_ids,
            uniform_buffer,
        };


        let bind_group_layout = Grid::create_binding_group_layout(wgpu_context);

        // Create bind group
        let bind_group = Self::create_binding_group(wgpu_context, &bind_group_layout, &grid_buffers, particle_system);
        
        let grid_binding_group = BindResources{
            bind_group,
            bind_group_layout,
        };
        
        let build_grid_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "build_cell_ids_array",
            &grid_binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
                ("MAX_CELLS_PER_OBJECT", MAX_CELLS_PER_OBJECT as f64)
            ],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<PushConstantsBuildGrid>() as u32,
            }]
        );

        

        let sorter: GPUSorter = GPUSorter::new(wgpu_context, NonZeroU32::new(buffer_len as u32).unwrap(), &grid_buffers.cell_ids, &grid_buffers.object_ids);

        Grid {
            dim,
            should_draw_grid: false,
            grid_drawer: None,
            grid_buffers,
            grid_kernels: GridKernels{build_cell_ids_shader: build_grid_shader, gpu_sorter: sorter},
            grid_binding_group,
            cell_size,
            num_elements: total_particles
        }
    }
    
    pub fn get_total_cells(cell_size: f32, world_dim: &Vec2) -> usize{
        (world_dim.x / cell_size) as usize * (world_dim.y / cell_size) as usize
    }
    
    pub fn toggle_grid_drawing(&mut self){
        self.should_draw_grid = !self.should_draw_grid;
    }

    pub fn compute_cell_size(max_obj_radius: f32) -> f32 {
        max_obj_radius * CELL_SIZE_MULTIPLIER
    }
    
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
    
    fn create_binding_group_layout(wgpu_context: &WgpuContext) -> wgpu::BindGroupLayout{
        let compute_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid compute Bind Group Layout"),
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
                // Uniform data
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Radius
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        wgpu_context.get_device().create_bind_group_layout(&compute_bind_group_layout)
    }
    fn create_binding_group(wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout, grid_buffers: &GridBuffers, particle_system: &ParticleSystem) -> wgpu::BindGroup{
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_system.positions().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grid_buffers.uniform_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_buffers.cell_ids.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grid_buffers.object_ids.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: particle_system.radius().buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }


    /// Refreshes the grid when elements have been added or removed.
    /// This function is called when the particles system is updated.
    pub fn refresh_grid(&mut self, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, particle_system: &ParticleSystem, prev_total_particles: usize){
        self.cell_size = Grid::compute_cell_size(particle_system.get_max_radius());
        self.num_elements = particle_system.len();
        let particles_added = self.num_elements - prev_total_particles;

        // Update the uniform

        let new_uniform: UniformData = UniformData {
            num_particles: self.num_elements as u32,
            num_collision_cells: self.num_elements as u32 * 2u32.pow(self.dim),
            cell_size: self.cell_size,
        };
        self.grid_buffers.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);
        
        
        // Recreate the grid drawer
        self.grid_drawer = Some(GridDrawer::new(wgpu_context, camera, &world_dimensions, self.cell_size));

        let buffer_size = particles_added * 4;
        self.grid_buffers.cell_ids.push_all(&vec![UNUSED_CELL_ID; buffer_size], wgpu_context);
        self.grid_buffers.object_ids.push_all(&vec![0; buffer_size], wgpu_context);
        
        
        // Update the binding group
        self.grid_binding_group.bind_group = Self::create_binding_group(wgpu_context, &self.grid_binding_group.bind_group_layout, &self.grid_buffers, particle_system);
        self.grid_kernels.gpu_sorter.update_sorting_buffers(wgpu_context, NonZeroU32::new(self.grid_buffers.object_ids.len() as u32).unwrap(), &self.grid_buffers.cell_ids, &self.grid_buffers.object_ids);
    }

    /// Step 1: Constructs the map of cell ids to objects.
    /// Key: cell id; Value: Object id
    /// Each particle has a max of 4 cell ids (in 2D space)
    pub fn build_cell_ids(&self, encoder: &mut CommandEncoder){
        self.grid_kernels.build_cell_ids_shader.dispatch_by_items(
            encoder,
            (self.num_elements as u32, 1, 1),
            Some(vec![(0u32, bytemuck::bytes_of(&PushConstantsBuildGrid{
                cell_size: self.cell_size,
                num_particles: self.num_elements as u32,
            }))]),
            &self.grid_binding_group.bind_group
        );
    }

    /// Step 2: Sorts the map of cell ids to objects by cell id.
    /// Key: cell id; Value: Object id
    pub fn sort_map(&mut self, encoder: &mut CommandEncoder){
        self.grid_kernels.gpu_sorter.sort(encoder, None);
    }
    
    pub fn download_cell_ids(&mut self, wgpu_context: &WgpuContext) ->  Result<Vec<u32>, BufferAsyncError>{
        Ok(self.grid_buffers.cell_ids.download(wgpu_context)?.clone())
    }

    pub fn download_object_ids(&mut self, wgpu_context: &WgpuContext) -> Result<Vec<u32>, BufferAsyncError> {
        Ok(self.grid_buffers.object_ids.download(wgpu_context)?.clone())
    }
    
    pub fn update(&mut self, encoder: &mut CommandEncoder, gpu_profiler: &mut GpuProfiler){
        {
            let mut scope = gpu_profiler.scope("Build cell ids", encoder);
            self.build_cell_ids(&mut scope);
        }

        {
            let mut scope = gpu_profiler.scope("Sort map", encoder);
            self.sort_map(&mut scope);
        }
    }
    
    pub fn object_ids(&self) -> &GpuBuffer<u32>{
        &self.grid_buffers.object_ids
    }
    
    pub fn cell_ids(&self) -> &GpuBuffer<u32>{
        &self.grid_buffers.cell_ids
    }

}


impl Renderable for Grid {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        if self.should_draw_grid {
            self.grid_drawer.as_ref().expect("Not drawing grid lines").draw(render_pass, camera);
        }
    }
}