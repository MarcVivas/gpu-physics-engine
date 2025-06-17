use crate::renderer::camera::Camera;

pub trait Renderable {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera);
}
