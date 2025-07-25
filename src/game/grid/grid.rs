use glam::{Vec2, Vec4};
use crate::game_data::line::lines::Lines;
use crate::game_data::particle::particle_system::ParticleSystem;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;
use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;
use wgpu::{BindGroupLayout, BufferAsyncError};
use crate::utils;
use crate::utils::radix_sort::radix_sort::GPUSorter;
use crate::utils::prefix_sum::prefix_sum::PrefixSum;

/// The value must match in the compute shader.
const COUNTING_CHUNK_SIZE: u32 = 4;
/// The value must match in the compute shader.
const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

pub struct Grid {
    dim: u32,
    render_grid: bool,
    cell_size: f32,
    world_dimensions: Vec2,
    lines: Option<Lines>,
    cell_ids: Rc<RefCell<GpuBuffer<u32>>>, // Indicates the cells an object is in. cell_ids[i..i+3] = cell_id_of_object_i
    object_ids: Rc<RefCell<GpuBuffer<u32>>>, // Need this after sorting to indicate the objects in a cell.
    chunk_counting_buffer: GpuBuffer<u32>, // Stores the number of objects in each chunk.
    collision_cells: GpuBuffer<u32>, // Stores the cells that contain more than one object.
    elements: Rc<RefCell<ParticleSystem>>,
    uniform_buffer: GpuBuffer<UniformData>,
    grid_kernels: GridKernels,
}

struct GridKernels {
    build_cell_ids_shader: ComputeShader,
    count_objects_per_chunk_shader: ComputeShader,
    build_collision_cells_shader: ComputeShader,
    gpu_sorter: GPUSorter,
    prefix_sum: PrefixSum,
}

impl GridKernels {
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformData {
    num_particles: u32,
    dim: u32,
    cell_size: f32,
    num_cell_ids: u32,
}

const CELL_SCALING_FACTOR: f32 = 2.2;

impl Grid {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, particle_system: Rc<RefCell<ParticleSystem>>) -> Grid {
        let mut lines = Some(Lines::new(wgpu_context, camera));
        let cell_size = Self::gen_cell_size(max_obj_radius);
        Self::generate_grid_lines(&mut lines, wgpu_context, world_dimensions, cell_size);

        let mut grid = Self::new_without_camera(wgpu_context, world_dimensions, max_obj_radius, particle_system.clone());
        grid.lines = lines;
        grid
    }

    // No camera needed for tests
    pub fn new_without_camera(wgpu_context: &WgpuContext, world_dimensions: Vec2, max_obj_radius: f32, particle_system: Rc<RefCell<ParticleSystem>>) -> Grid{
        let total_particles: usize = particle_system.borrow().len();
        let dim: u32 = 2;
        let buffer_len = total_particles * 2usize.pow(dim); // A particle can be in 2**dim different cells
        let cell_size = Self::gen_cell_size(max_obj_radius);

        
        
        let cell_ids = Rc::new(RefCell::new(GpuBuffer::new(
            wgpu_context,
            vec![u32::MAX; GPUSorter::get_required_keys_buffer_size(buffer_len as u32) as usize],
            wgpu::BufferUsages::STORAGE)));
        
        let object_ids = Rc::new(RefCell::new(GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_len],
            wgpu::BufferUsages::STORAGE
        )));
        
        let collision_cells = GpuBuffer::new(
            wgpu_context,
            vec![u32::MAX; buffer_len],
            wgpu::BufferUsages::STORAGE,       
        );
        
        let chunk_counting_buffer_len: usize = ((buffer_len as u32 + COUNTING_CHUNK_SIZE -1) / COUNTING_CHUNK_SIZE) as usize;
        
        let chunk_counting_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0; chunk_counting_buffer_len],
            wgpu::BufferUsages::STORAGE,      
        );

        
        let sorter: GPUSorter = GPUSorter::new(wgpu_context.get_device(), utils::guess_workgroup_size(wgpu_context).unwrap(), NonZeroU32::new(buffer_len as u32).unwrap(), cell_ids.clone(), object_ids.clone());
        
        
        let uniform_data = GpuBuffer::new(
            wgpu_context,
            vec![UniformData {
                num_cell_ids: buffer_len as u32,
                num_particles: total_particles as u32,
                dim,
                cell_size,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );


        // This layout describes what the compute shader can access.
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
            ],
        };

        let bind_group_layout = wgpu_context.get_device().create_bind_group_layout(&compute_bind_group_layout);

        // Create bind group
        let bind_group = wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_system.borrow().instances().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform_data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cell_ids.borrow().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: object_ids.borrow().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: particle_system.borrow().radius().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: collision_cells.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: chunk_counting_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        );

        let build_grid_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "build_cell_ids_array",
            bind_group.clone(),
            bind_group_layout.clone(),
            WORKGROUP_SIZE,
        );

        let count_objects_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "count_objects_for_each_chunk",
            bind_group.clone(),
            bind_group_layout.clone(),
            WORKGROUP_SIZE,
        );
        
        let build_collision_cells_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "build_collision_cells_array",
            bind_group,
            bind_group_layout,
            WORKGROUP_SIZE,
        );
        
        let prefix_sum = PrefixSum::new(wgpu_context, &chunk_counting_buffer);
        
        Grid {
            dim,
            render_grid: false,
            cell_size,
            world_dimensions,
            lines: None,
            cell_ids: cell_ids.clone(),
            object_ids: object_ids.clone(),
            collision_cells,
            elements: particle_system.clone(),
            uniform_buffer: uniform_data,
            chunk_counting_buffer,
            grid_kernels: GridKernels{build_cell_ids_shader: build_grid_shader, count_objects_per_chunk_shader: count_objects_shader, gpu_sorter: sorter, prefix_sum, build_collision_cells_shader },
        }
    }

    

    pub fn render_grid(&mut self){
        self.render_grid = !self.render_grid;
    }

    pub fn gen_cell_size(max_obj_radius: f32) -> f32 {
        max_obj_radius * CELL_SCALING_FACTOR
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
    
    fn get_binding_group(&self, wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout) -> wgpu::BindGroup{
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.elements.borrow().instances().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cell_ids.borrow().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.object_ids.borrow().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.elements.borrow().radius().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.collision_cells.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.chunk_counting_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }


    /// Refreshes the grid when elements have been added or removed.
    /// This function is called when the particle system is updated.
    pub fn refresh_grid(&mut self, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, particles_added: usize){
        let cell_size = max_obj_radius * 2.2;

        self.lines = Some(Lines::new(wgpu_context, camera)); // Create a brand new Lines instance
        Self::generate_grid_lines(&mut self.lines, wgpu_context, world_dimensions, cell_size);

        let buffer_size = particles_added * 4;
        self.cell_ids.borrow_mut().push_all(&vec![0; GPUSorter::get_required_keys_buffer_size(buffer_size as u32) as usize], wgpu_context);
        self.object_ids.borrow_mut().push_all(&vec![0; buffer_size], wgpu_context);
        self.chunk_counting_buffer.push_all(&vec![0; ((buffer_size as u32 + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE) as usize], wgpu_context);
        self.collision_cells.push_all(&vec![0; buffer_size], wgpu_context);
        
        
        
        let binding_group_layout = self.grid_kernels.build_cell_ids_shader.get_bind_group_layout();
        let binding_group = self.get_binding_group(wgpu_context, &binding_group_layout);
        self.grid_kernels.build_cell_ids_shader.update_binding_group(wgpu_context, binding_group.clone());
        self.grid_kernels.count_objects_per_chunk_shader.update_binding_group(wgpu_context, binding_group.clone());
        self.grid_kernels.build_collision_cells_shader.update_binding_group(wgpu_context, binding_group);
        self.grid_kernels.gpu_sorter.update_sorting_buffers(wgpu_context.get_device(), NonZeroU32::new(self.object_ids.borrow().len() as u32).unwrap(), self.cell_ids.clone(), self.object_ids.clone());
        self.grid_kernels.prefix_sum.update_buffers(wgpu_context, &self.chunk_counting_buffer);
    }

    /// Step 1: Constructs the map of cell ids to objects.
    /// Key: cell id; Value: Object id
    /// Each particle would have a max of 4 cell ids (in 2D space)
    pub fn build_cell_ids(&self, encoder: &mut wgpu::CommandEncoder, total_particles: u32){
        self.grid_kernels.build_cell_ids_shader.dispatch_by_items(
            encoder,
            (total_particles, 1, 1),
        );
    }

    /// Step 2: Sorts the map of cell ids to objects by cell id.
    /// Key: cell id; Value: Object id
    pub fn sort_map(&self, encoder: &mut wgpu::CommandEncoder, wgpu_context: &WgpuContext){
        self.grid_kernels.gpu_sorter.sort(encoder, wgpu_context.get_queue(), None);
    }
    
    /// Step 3: Builds the collision cell list.
    /// Key: cell id; Value: Object id
    /// Collision cells are cells that contain more than one object, and therefore they need to be checked for potential collisions 
    pub fn build_collision_cells(&self, encoder: &mut wgpu::CommandEncoder){
        // Step 3.1 Count the number of object in each chunk that share the same cell id
        let num_chunks = (self.collision_cells.len() as u32 + COUNTING_CHUNK_SIZE - 1) / COUNTING_CHUNK_SIZE;
        self.grid_kernels.count_objects_per_chunk_shader.dispatch_by_items(
            encoder,
            (num_chunks, 1, 1),
        );
        
        // Step 3.2 Prefix sums the number of objects in each chunk
        self.grid_kernels.prefix_sum.execute(encoder, self.chunk_counting_buffer.len() as u32);

        // Step 3.3 Build the collision cell list
        self.grid_kernels.build_collision_cells_shader.dispatch_by_items(encoder, (num_chunks, 1, 1));
    }

    pub fn download_cell_ids(&mut self, wgpu_context: &WgpuContext) ->  Result<Vec<u32>, BufferAsyncError>{
        Ok(self.cell_ids.borrow_mut().download(wgpu_context)?.clone())
    }

    pub fn download_object_ids(&mut self, wgpu_context: &WgpuContext) -> Result<Vec<u32>, BufferAsyncError> {
        Ok(self.object_ids.borrow_mut().download(wgpu_context)?.clone())
    }
}


impl Renderable for Grid {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        if self.render_grid{
            self.lines.as_ref().expect("Not drawing grid lines").draw(render_pass, camera);
        }
    }
    fn update(&mut self, delta_time:f32, world_size:&glam::Vec2, wgpu_context: &WgpuContext){
        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        // Update the uniform if needed
        let total_particles = self.elements.borrow().len() as u32;
        let prev_num_particles = self.uniform_buffer.data()[0].num_particles;
        if total_particles != prev_num_particles {
            let new_uniform:UniformData = UniformData{
                num_particles: total_particles,
                num_cell_ids: total_particles * 2u32.pow(self.dim),
                cell_size: Self::gen_cell_size(self.elements.borrow().get_max_radius()),
                dim: self.dim
            };
            self.uniform_buffer.replace_elem(new_uniform, 0, wgpu_context);
        }

        // Step 1: Build the cell IDs array
        self.build_cell_ids(&mut encoder, total_particles);

        // Step 2: Sort the map of cell ids to objects by cell id
        self.sort_map(&mut encoder, wgpu_context);
        
        // Step 3: Build the collision cell list
        self.build_collision_cells(&mut encoder);

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

        println!("{:?}" , self.chunk_counting_buffer.download(wgpu_context).unwrap());
        //println!("Cell ids{:?}", &self.cell_ids.borrow_mut().download(wgpu_context).unwrap().as_slice()[0..total_particles as usize * 4usize]);
        //println!("Object ids{:?}", self.object_ids.borrow_mut().download(wgpu_context).unwrap());
        // self.elements.borrow().instances().download(wgpu_context).unwrap();
        // self.elements.borrow().radius().download(wgpu_context).unwrap();

    }


}