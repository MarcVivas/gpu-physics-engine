use glam::{Vec2, Vec4};
use crate::game_data::line::lines::Lines;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct Grid {
    render_grid: bool,
    cell_size: f32,
    world_dimensions: Vec2,
    lines: Lines,
    cell_ids: GpuBuffer<u32>, // Indicates the cells an object is in. cell_ids[i..i+3] = cell_id_of_object_i
    object_ids: GpuBuffer<u32>, // Need this after sorting to indicate the objects in a cell.
}



impl Grid {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, total_particles: usize) -> Grid {
        let cell_size = max_obj_radius * 2.2;
        let mut lines = Lines::new(wgpu_context, camera);
        Self::generate_grid_lines(&mut lines, wgpu_context, world_dimensions, cell_size);
        
        let buffer_size = total_particles * 4; // A particle can be in 4 different cells
        
        let cell_ids = GpuBuffer::new(
            wgpu_context,
            vec![0; buffer_size],
            wgpu::BufferUsages::STORAGE);
        let object_ids = GpuBuffer::new(
            wgpu_context, 
            vec![0; buffer_size],
            wgpu::BufferUsages::STORAGE
        );
        
        Grid {
            render_grid: false,
            cell_size,
            world_dimensions,
            lines,
            cell_ids,
            object_ids,       
        }
    }
    
    pub fn render_grid(&mut self){
        self.render_grid = !self.render_grid;
    }

    fn generate_grid_lines(lines: &mut Lines, wgpu_context: &WgpuContext, world_dimensions: Vec2, cell_size: f32){


        let num_vertical_lines = world_dimensions.x / cell_size;
        let mut start = Vec2::new(0.0, 0.0);
        let mut end = Vec2::new(0.0, 0.0);

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
        
        lines.push_all(wgpu_context, &positions, &colors, &thicknesses);
    }
    
    
    pub fn reset_grid(&mut self, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32, particles_added: usize){
        let cell_size = max_obj_radius * 2.2;
        
        self.lines = Lines::new(wgpu_context, camera); // Create a brand new Lines instance
        Self::generate_grid_lines(&mut self.lines, wgpu_context, world_dimensions, cell_size);
        
        let buffer_size = particles_added * 4;
        self.cell_ids.push_all(&vec![0; buffer_size], wgpu_context);
        self.object_ids.push_all(&vec![0; buffer_size], wgpu_context);
    }
}

impl Renderable for Grid {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        if self.render_grid{
            self.lines.draw(render_pass, camera);
        }
    }
    fn update(&self, delta_time:f32, world_size:&glam::Vec2, wgpu_context: &WgpuContext){
        
    }
}