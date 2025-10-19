use glam::{Vec2, Vec4};
use crate::lines::lines::Lines;
use crate::renderer::camera::Camera;
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;

pub struct GridDrawer {
    lines: Lines,

}

impl GridDrawer {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: &Vec2, cell_size: f32) -> Self {
        let lines = Self::create_grid_lines(wgpu_context, camera, world_dimensions.clone(), cell_size);
        Self {
            lines,
        }
    }
    
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera) {
        self.lines.draw(render_pass, camera);       
    }
    
    fn create_grid_lines(wgpu_context: &WgpuContext, camera: &Camera, world_dimensions: Vec2, cell_size: f32) -> Lines {
        let mut lines = Lines::new(wgpu_context, camera);

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

        lines.push_all(wgpu_context, &positions, &colors, &thicknesses);
        lines
    }
    
}