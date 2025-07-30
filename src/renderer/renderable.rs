use crate::renderer::camera::Camera;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_timer::GpuTimer;

pub trait Renderable {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera);
    fn update(&mut self, delta_time:f32, world_size:&glam::Vec2, wgpu_context: &WgpuContext, gpu_timer: &mut GpuTimer);
}
