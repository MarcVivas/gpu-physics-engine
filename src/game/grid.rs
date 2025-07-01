use glam::{Vec2, Vec4};
use wasm_bindgen_futures::js_sys::Math::ceil;
use crate::game_data::line::lines::Lines;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;

pub struct Grid {
    cell_size: f32,
    world_dimensions: Vec2,
    lines: Lines,
}

impl Grid {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, max_obj_radius: f32) -> Grid {
        let cell_size = max_obj_radius * 2.2;
        let mut lines = Lines::new(wgpu_context, camera);
        Self::generate_grid_lines(&mut lines, wgpu_context, camera, world_dimensions, cell_size);
        
        Grid {
            cell_size,
            world_dimensions,
            lines,
        }
    }
    
    fn generate_grid_lines(lines: &mut Lines, wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, cell_size: f32){
        
        
        let num_vertical_lines = world_dimensions.x / cell_size;
        let mut start = Vec2::new(0.0, 0.0);
        let mut end = Vec2::new(0.0, 0.0);
        
        for i in 0..num_vertical_lines.ceil() as u32{
            start = Vec2::new(i as f32 * cell_size, 0.0);
            end = Vec2::new(i as f32 * cell_size, world_dimensions.y);
            lines.push(wgpu_context, start, end, Vec4::new(1.0, 1.0, 1.0, 1.0), 1.0);
        }
        
        let num_horizontal_lines = world_dimensions.y / cell_size;
        for i in 0..num_horizontal_lines.ceil() as u32 {
            start = Vec2::new(0.0, i as f32 * cell_size);
            end = Vec2::new(world_dimensions.x, i as f32 * cell_size);
            lines.push(wgpu_context, start, end, Vec4::new(1.0, 1.0, 1.0, 1.0), 1.0);
        }
        
    }
}

impl Renderable for Grid {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        self.lines.draw(render_pass, camera);
    }
    fn update(&self, delta_time:f32, world_size:&glam::Vec2, wgpu_context: &WgpuContext){
        
    }
}