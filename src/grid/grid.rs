use glam::{Vec2, Vec4};
use crate::lines::lines::Lines;
use crate::particles::particle_system::ParticleSystem;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;
use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;
use wgpu::{BindGroupLayout, BufferAsyncError, PushConstantRange};
use crate::utils;
use crate::utils::bind_resources::BindResources;
use crate::utils::gpu_timer::GpuTimer;
use crate::utils::radix_sort::radix_sort::{GPUSorter};
use crate::utils::prefix_sum::prefix_sum::PrefixSum;

/// The value must match in the compute shader.
const COUNTING_CHUNK_SIZE: u32 = 4;
/// The value must match in the compute shader.
const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

const MAX_CELLS_PER_OBJECT: u32 = 4;

const CELL_SIZE_MULTIPLIER: f32 = 2.2f32;

const UNUSED_CELL_ID: u32 = u32::MAX;


pub struct Grid {
    dim: u32,
    render_grid: bool,
    lines: Option<Lines>,
    grid_buffers: GridBuffers,
    grid_kernels: GridKernels,
    grid_binding_group: BindResources,
    cell_size: f32,
    num_elements: usize,
}

struct GridBuffers{
    cell_ids: GpuBuffer<u32>, // Indicates the cells an object is in. cell_ids[i..i+3] = cell_id_of_object_i
    object_ids: GpuBuffer<u32>, // Need this after sorting to indicate the objects in a cell.
    chunk_counting_buffer: GpuBuffer<u32>, // Stores the number of objects in each chunk.
    collision_cells: GpuBuffer<u32>, // Stores the cells that contain more than one object.
    indirect_dispatch_buffer: GpuBuffer<u32>,
    uniform_buffer: GpuBuffer<UniformData>,
}

struct GridKernels {
    build_cell_ids_shader: ComputeShader,
    count_objects_per_chunk_shader: ComputeShader,
    build_collision_cells_shader: ComputeShader,
    collision_solver_shader: ComputeShader,
    gpu_sorter: GPUSorter,
    prefix_sum: PrefixSum,
}



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformData {
    num_particles: u32,
    num_collision_cells: u32,
    cell_size: f32,
    delta_time: f32,
    color: u32,
    num_counting_chunks: u32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstantsBuildGrid {
    cell_size: f32,
    num_particles: u32,
}

impl Grid {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, particle_system: Rc<RefCell<ParticleSystem>>) -> Grid {
        let mut lines = Some(Lines::new(wgpu_context, camera));
        let cell_size = Self::gen_cell_size(max_obj_radius);
        Self::generate_grid_lines(&mut lines, wgpu_context, world_dimensions, cell_size);

        let mut grid = Self::new_without_camera(wgpu_context, max_obj_radius, particle_system.clone());
        grid.lines = lines;
        grid
    }

    // No camera needed for tests
    pub fn new_without_camera(wgpu_context: &WgpuContext, max_obj_radius: f32, particle_system: Rc<RefCell<ParticleSystem>>) -> Grid{
        let total_particles: usize = particle_system.borrow().len();
        let dim: u32 = 2;
        let buffer_len = total_particles * 2usize.pow(dim); // A particle can be in 2**dim different cells
        let cell_size = Self::gen_cell_size(max_obj_radius);

        
        
        let cell_ids = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; buffer_len],
            wgpu::BufferUsages::STORAGE);
        
        let object_ids = GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_len],
            wgpu::BufferUsages::STORAGE
        );
        
        let collision_cells = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; buffer_len],
            wgpu::BufferUsages::STORAGE,       
        );
        
        let chunk_counting_buffer_len: usize = ((buffer_len as u32 + COUNTING_CHUNK_SIZE -1) / COUNTING_CHUNK_SIZE) as usize;
        
        let chunk_counting_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0; chunk_counting_buffer_len],
            wgpu::BufferUsages::STORAGE,      
        );

        
        let uniform_buffer = GpuBuffer::new(
            wgpu_context,
            vec![UniformData {
                num_collision_cells: 0u32,
                num_particles: total_particles as u32,
                cell_size,
                delta_time: 0.0,
                color: 0u32,
                num_counting_chunks: Grid::get_num_counting_chunks(&collision_cells),
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let indirect_dispatch_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0; 3],
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );
        
        let grid_buffers = GridBuffers {
            cell_ids,
            object_ids,
            collision_cells,
            chunk_counting_buffer,
            indirect_dispatch_buffer,
            uniform_buffer,
        };


        let bind_group_layout = Grid::generate_binding_group_layout(wgpu_context);

        // Create bind group
        let bind_group = Self::generate_binding_group(wgpu_context, &bind_group_layout, &grid_buffers, particle_system.clone());
        
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

        let count_objects_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "count_objects_for_each_chunk",
            &grid_binding_group.bind_group_layout,
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
            wgpu::include_wgsl!("grid.wgsl"),
            "build_collision_cells_array",
            &grid_binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
                ("MAX_CELLS_PER_OBJECT", MAX_CELLS_PER_OBJECT as f64),
                ("CHUNK_SIZE", COUNTING_CHUNK_SIZE as f64)
            ],
            &vec![]
        );
        

        let collision_solver_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "solve_collisions",
            &grid_binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![
                ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
                ("MAX_CELLS_PER_OBJECT", MAX_CELLS_PER_OBJECT as f64),
                ("CHUNK_SIZE", COUNTING_CHUNK_SIZE as f64)
            ],
            &vec![]
        );
        let prefix_sum = PrefixSum::new(wgpu_context, &grid_buffers.chunk_counting_buffer);
        let sorter: GPUSorter = GPUSorter::new(wgpu_context, NonZeroU32::new(buffer_len as u32).unwrap(), &grid_buffers.cell_ids, &grid_buffers.object_ids);

        Grid {
            dim,
            render_grid: false,
            lines: None,
            grid_buffers,
            grid_kernels: GridKernels{build_cell_ids_shader: build_grid_shader, count_objects_per_chunk_shader: count_objects_shader, gpu_sorter: sorter, prefix_sum, build_collision_cells_shader, collision_solver_shader },
            grid_binding_group,
            cell_size,
            num_elements: total_particles
        }
    }

    

    pub fn render_grid(&mut self){
        self.render_grid = !self.render_grid;
    }

    pub fn gen_cell_size(max_obj_radius: f32) -> f32 {
        max_obj_radius * CELL_SIZE_MULTIPLIER
    }
    fn generate_grid_lines(lines: &mut Option<Lines>, wgpu_context: &WgpuContext, world_dimensions: Vec2, cell_size: f32){


        let num_vertical_lines = world_dimensions.x / cell_size;
        let mut start;
        let mut end;

        let mut positions: Vec<Vec2> = Vec::new();
        let mut colors: Vec<Vec4> = Vec::new();
        let mut thicknesses: Vec<f32> = Vec::new();

        for i in 0..num_vertical_lines.ceil() as u32{
            start = Vec2::new(i as f32 * cell_size, 0.0);
            end = Vec2::new(i as f32 * cell_size, world_dimensions.y);
            positions.push(start);
            positions.push(end);
            colors.push(Vec4::new(1.0, 1.0, 1.0, 1.0)); // Color for start point
            colors.push(Vec4::new(1.0, 1.0, 1.0, 1.0)); // Color for end point
            thicknesses.push(1.0);                     // Thickness for start point
            thicknesses.push(1.0);                     // Thickness for end point
        }

        let num_horizontal_lines = world_dimensions.y / cell_size;
        for i in 0..num_horizontal_lines.ceil() as u32 {
            start = Vec2::new(0.0, i as f32 * cell_size);
            end = Vec2::new(world_dimensions.x, i as f32 * cell_size);
            positions.push(start);
            positions.push(end);
            colors.push(Vec4::new(1.0, 1.0, 1.0, 1.0)); // Color for start point
            colors.push(Vec4::new(1.0, 1.0, 1.0, 1.0)); // Color for end point
            thicknesses.push(1.0);                     // Thickness for start point
            thicknesses.push(1.0);                     // Thickness for end point
        }

        lines.as_mut().expect("No lines").push_all(wgpu_context, &positions, &colors, &thicknesses);
    }
    
    fn generate_binding_group_layout(wgpu_context: &WgpuContext) -> wgpu::BindGroupLayout{
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
                // Collision cells
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
                // Chunk counting buffer
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
    fn generate_binding_group(wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout, grid_buffers: &GridBuffers, particle_system: Rc<RefCell<ParticleSystem>>) -> wgpu::BindGroup{
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_system.borrow().positions().buffer().as_entire_binding(),
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
                        resource: particle_system.borrow().radius().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: grid_buffers.collision_cells.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: grid_buffers.chunk_counting_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: grid_buffers.indirect_dispatch_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }


    /// Refreshes the grid when elements have been added or removed.
    /// This function is called when the particles system is updated.
    pub fn refresh_grid(&mut self, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, particle_system: Rc<RefCell<ParticleSystem>>, prev_total_particles: usize){
        self.cell_size = Grid::gen_cell_size(particle_system.borrow().get_max_radius());
        self.num_elements = particle_system.borrow().len();
        let particles_added = self.num_elements - prev_total_particles;
        
        // Create a brand new Lines instance
        self.lines = Some(Lines::new(wgpu_context, camera)); 
        Self::generate_grid_lines(&mut self.lines, wgpu_context, world_dimensions, self.cell_size);

        let buffer_size = particles_added * 4;
        self.grid_buffers.cell_ids.push_all(&vec![UNUSED_CELL_ID; buffer_size], wgpu_context);
        self.grid_buffers.object_ids.push_all(&vec![0; buffer_size], wgpu_context);
        self.grid_buffers.chunk_counting_buffer.push_all(&vec![0; ((buffer_size as u32 + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE) as usize], wgpu_context);
        self.grid_buffers.collision_cells.push_all(&vec![UNUSED_CELL_ID; buffer_size], wgpu_context);
        
        
        
        // Update the binding group
        self.grid_binding_group.bind_group = Self::generate_binding_group(wgpu_context, &self.grid_binding_group.bind_group_layout, &self.grid_buffers, particle_system);
        self.grid_kernels.gpu_sorter.update_sorting_buffers(wgpu_context, NonZeroU32::new(self.grid_buffers.object_ids.len() as u32).unwrap(), &self.grid_buffers.cell_ids, &self.grid_buffers.object_ids);
        self.grid_kernels.prefix_sum.update_buffers(wgpu_context, &self.grid_buffers.chunk_counting_buffer);
    }

    /// Step 1: Constructs the map of cell ids to objects.
    /// Key: cell id; Value: Object id
    /// Each particle has a max of 4 cell ids (in 2D space)
    pub fn build_cell_ids(&self, encoder: &mut wgpu::CommandEncoder){
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
    pub fn sort_map(&mut self, encoder: &mut wgpu::CommandEncoder, wgpu_context: &WgpuContext){
        self.grid_kernels.gpu_sorter.sort(encoder, wgpu_context, None);
    }
    
    /// Step 3: Builds the collision cell list.
    /// Key: cell id; Value: Object id
    /// Collision cells are cells that contain more than one object, and therefore they need to be checked for potential collisions 
    pub fn build_collision_cells(&self, wgpu_context: &WgpuContext,  encoder: &mut wgpu::CommandEncoder){
        // Step 3.1 Count the number of objects in each chunk that share the same cell id
        let num_chunks = Grid::get_num_counting_chunks(&self.grid_buffers.collision_cells);
        self.grid_kernels.count_objects_per_chunk_shader.dispatch_by_items(
            encoder,
            (num_chunks, 1, 1),
            None,
            &self.grid_binding_group.bind_group       
        );
        
        // Step 3.2 Prefix sums the number of objects in each chunk
        self.grid_kernels.prefix_sum.execute(wgpu_context, encoder, self.grid_buffers.chunk_counting_buffer.len() as u32);

        // Step 3.3 Build the collision cell list
        self.grid_kernels.build_collision_cells_shader.dispatch_by_items(encoder, (num_chunks, 1, 1), None, &self.grid_binding_group.bind_group);
    }
    
    
    /// Step 4: Solves collisions between objects in the same cell.
    #[cfg(feature = "benchmark")]
    pub fn solve_collisions(&mut self, wgpu_context: &WgpuContext, dt: f32, gpu_timer: &mut GpuTimer){
        
        // Update the uniform if needed
        let old_uniform = self.grid_buffers.uniform_buffer.data()[0];
        
        for color in 1u32..=4u32 {
            let mut encoder = wgpu_context.get_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some(&format!("Collision Encoder Color {}", color)) }
            );

            let scope_label = format!("Solve Collisions - Color {}", color);
            gpu_timer.scope(&scope_label, &mut encoder, |encoder| {
                let new_uniform: UniformData = UniformData {
                    num_particles: old_uniform.num_particles,
                    num_collision_cells: old_uniform.num_collision_cells,
                    cell_size: old_uniform.cell_size,
                    delta_time: dt,
                    color,
                    num_counting_chunks: Grid::get_num_counting_chunks(&self.grid_buffers.collision_cells),
                };
                self.grid_buffers.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);

                self.grid_kernels.collision_solver_shader.indirect_dispatch(
                    encoder,
                    self.grid_buffers.indirect_dispatch_buffer.buffer(),
                    0,
                    None,
                    &self.grid_binding_group.bind_group
                );
            });

            wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        }

    }


    #[cfg(not(feature = "benchmark"))]
    pub fn solve_collisions(&mut self, wgpu_context: &WgpuContext, dt: f32){

        // Update the uniform if needed
        let old_uniform = self.grid_buffers.uniform_buffer.data()[0];

        for color in 1u32..=4u32 {
            let mut encoder = wgpu_context.get_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some(&format!("Collision Encoder Color {}", color)) }
            );
            let new_uniform: UniformData = UniformData {
                num_particles: old_uniform.num_particles,
                num_collision_cells: old_uniform.num_collision_cells,
                cell_size: old_uniform.cell_size,
                delta_time: dt,
                color,
                num_counting_chunks: Grid::get_num_counting_chunks(&self.grid_buffers.collision_cells),
            };
            self.grid_buffers.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);

            self.grid_kernels.collision_solver_shader.indirect_dispatch(
                &mut encoder,
                self.grid_buffers.indirect_dispatch_buffer.buffer(),
                0,
                None,
                &self.grid_binding_group.bind_group
            );
            wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        }

    }

    pub fn download_cell_ids(&mut self, wgpu_context: &WgpuContext) ->  Result<Vec<u32>, BufferAsyncError>{
        Ok(self.grid_buffers.cell_ids.download(wgpu_context)?.clone())
    }

    pub fn download_object_ids(&mut self, wgpu_context: &WgpuContext) -> Result<Vec<u32>, BufferAsyncError> {
        Ok(self.grid_buffers.object_ids.download(wgpu_context)?.clone())
    }

    pub fn download_collision_cells(&mut self, wgpu_context: &WgpuContext) -> Result<Vec<u32>, BufferAsyncError> {
        Ok(self.grid_buffers.collision_cells.download(wgpu_context)?.clone())
    }
    
    fn get_num_counting_chunks(collision_cells: &GpuBuffer<u32>) -> u32 {
        (collision_cells.len() as u32 + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE
    }
}


impl Renderable for Grid {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        if self.render_grid{
            self.lines.as_ref().expect("Not drawing grid lines").draw(render_pass, camera);
        }
    }
    
    #[cfg(feature = "benchmark")]
    fn update(&mut self, delta_time:f32, _world_size: &Vec2, wgpu_context: &WgpuContext, gpu_timer: &mut GpuTimer){
        gpu_timer.begin_frame();
        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        // Update the uniform if needed
        let total_particles = self.num_elements as u32;
        let prev_num_particles = self.grid_buffers.uniform_buffer.data()[0].num_particles;
        if total_particles != prev_num_particles {
            let new_uniform:UniformData = UniformData{
                num_particles: total_particles,
                num_collision_cells: total_particles * 2u32.pow(self.dim),
                cell_size: self.cell_size,
                delta_time,
                color: 0u32,
                num_counting_chunks: Grid::get_num_counting_chunks(&self.grid_buffers.collision_cells),
            };
            self.grid_buffers.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);
        }

        gpu_timer.scope("Build Cell IDs", &mut encoder, |encoder| {
            self.build_cell_ids(encoder);
        });

        gpu_timer.scope("Sort Map", &mut encoder, |encoder| {
            self.sort_map(encoder, wgpu_context);
        });

        gpu_timer.scope("Build Collision Cells", &mut encoder, |encoder| {
            self.build_collision_cells(wgpu_context, encoder);
        });

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

        // Step 4: Perform the collision resolution.
        self.solve_collisions(wgpu_context, delta_time, gpu_timer);

        gpu_timer.end_frame(wgpu_context.get_device(), wgpu_context.get_queue());

    }


    #[cfg(not(feature = "benchmark"))]
    fn update(&mut self, delta_time:f32, _world_size: &Vec2, wgpu_context: &WgpuContext){

        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        // Update the uniform if needed
        let total_particles = self.num_elements as u32;
        let prev_num_particles = self.grid_buffers.uniform_buffer.data()[0].num_particles;
        if total_particles != prev_num_particles {
            let new_uniform:UniformData = UniformData{
                num_particles: total_particles,
                num_collision_cells: total_particles * 2u32.pow(self.dim),
                cell_size: self.cell_size,
                delta_time,
                color: 0u32,
                num_counting_chunks: Grid::get_num_counting_chunks(&self.grid_buffers.collision_cells),
            };
            self.grid_buffers.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);
        }

        self.build_cell_ids(&mut encoder);

        self.sort_map(&mut encoder, wgpu_context);

        self.build_collision_cells(wgpu_context, &mut encoder);

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

        // Step 4: Perform the collision resolution.
        self.solve_collisions(wgpu_context, delta_time);

    }


}