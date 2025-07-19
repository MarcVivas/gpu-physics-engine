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
use wgpu::BufferAsyncError;
use crate::utils;
use crate::utils::radix_sort::radix_sort::GPUSorter;

pub struct Grid {
    dim: u32,
    render_grid: bool,
    cell_size: f32,
    world_dimensions: Vec2,
    lines: Option<Lines>,
    cell_ids: GpuBuffer<u32>, // Indicates the cells an object is in. cell_ids[i..i+3] = cell_id_of_object_i
    object_ids: GpuBuffer<u32>, // Need this after sorting to indicate the objects in a cell.
    build_grid_shader: ComputeShader,
    elements: Rc<RefCell<ParticleSystem>>,
    uniform_buffer: GpuBuffer<UniformData>,
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

        let cell_ids = GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_len],
            wgpu::BufferUsages::STORAGE);
        let object_ids = GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_len],
            wgpu::BufferUsages::STORAGE
        );

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
            ],
        };

        let build_grid_shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("grid.wgsl"),
            "build_cell_ids_array",
            &compute_bind_group_layout,
            &[
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
                    resource: cell_ids.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: object_ids.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: particle_system.borrow().radius().buffer().as_entire_binding(),
                },
            ],
            (64, 1, 1),
        );
        
        /*
        let sorter: GPUSorter = GPUSorter::new(wgpu_context.get_device(), utils::guess_workgroup_size(wgpu_context).unwrap());
        sorter.create_sort_buffers(
            wgpu_context.get_device(),
            NonZeroU32::new(buffer_len as u32).unwrap(),
            cell_ids.buffer(),
            object_ids.buffer(),
        );
        */
        Grid {
            dim,
            render_grid: false,
            cell_size,
            world_dimensions,
            lines: None,
            cell_ids,
            object_ids,
            build_grid_shader,
            elements: particle_system,
            uniform_buffer: uniform_data,
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


    pub fn reset_grid(&mut self, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, particles_added: usize){
        let cell_size = max_obj_radius * 2.2;

        self.lines = Some(Lines::new(wgpu_context, camera)); // Create a brand new Lines instance
        Self::generate_grid_lines(&mut self.lines, wgpu_context, world_dimensions, cell_size);

        let buffer_size = particles_added * 4;
        self.cell_ids.push_all(&vec![0; buffer_size], wgpu_context);
        self.object_ids.push_all(&vec![0; buffer_size], wgpu_context);

        self.build_grid_shader.update_binding_group(wgpu_context, &[
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
                resource: self.cell_ids.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.object_ids.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: self.elements.borrow().radius().buffer().as_entire_binding(),
            },
        ],
        );
    }

    pub fn build_cell_ids(&self, encoder: &mut wgpu::CommandEncoder, total_particles: u32){
        self.build_grid_shader.dispatch_by_items(
            encoder,
            (total_particles, 1, 1),
        );
    }

    pub fn download_cell_ids(&mut self, wgpu_context: &WgpuContext) ->  Result<&Vec<u32>, BufferAsyncError>{
        self.cell_ids.download(wgpu_context)
    }

    pub fn download_object_ids(&mut self, wgpu_context: &WgpuContext) -> Result<&Vec<u32>, BufferAsyncError> {
        self.object_ids.download(wgpu_context)
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
        
        //self.gpu_sorter.sort(&mut encoder, wgpu_context.get_queue(), None);


        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

        //println!("Cell ids{:?}", self.cell_ids.download(wgpu_context).unwrap());
        //println!("Object ids{:?}", self.object_ids.download(wgpu_context).unwrap());
        // self.elements.borrow().instances().download(wgpu_context).unwrap();
        // self.elements.borrow().radius().download(wgpu_context).unwrap();

    }


}
