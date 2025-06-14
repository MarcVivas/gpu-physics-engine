use glam::Vec3;
use wgpu::util::DeviceExt;
use crate::renderer::camera::{Camera};
use crate::wgpu_context::WgpuContext;
use crate::game_data::particle::particle_system::ParticleSystem;


// Manages multiple render pipelines
pub struct Renderer {
    rendering_pipelines: Vec<wgpu::RenderPipeline>,
    background_color: wgpu::Color,
    camera: Camera,
    particles: ParticleSystem,
}



impl Renderer {
    pub fn new(wgpu_context: &WgpuContext, world_size: &glam::Vec2) -> Option<Self> {
        // 4. Create the camera with the calculated values
        let camera = Camera::new(world_size, &wgpu_context);
        let particles: ParticleSystem = ParticleSystem::new(&wgpu_context, &camera);


        Some(Self {
            rendering_pipelines: vec![

            ],
            background_color: wgpu::Color::BLACK,
            camera,
            particles,
        })
    }

    pub fn add_pipeline(&mut self, pipeline: wgpu::RenderPipeline){
        self.rendering_pipelines.push(pipeline);
    }

    pub fn render(&self, wgpu_context: &WgpuContext) -> Result<(), wgpu::SurfaceError>{
        self.particles.draw(wgpu_context, &self.camera, self.background_color.clone());
        Ok(())
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn update_camera(&mut self, wgpu_context: &WgpuContext){
        self.camera.build_view_projection_matrix(wgpu_context.window_size().width as f32, wgpu_context.window_size().height as f32);
        wgpu_context.get_queue().write_buffer(
            &self.camera.camera_buffer(),
            0, // offset
            bytemuck::cast_slice(&[*self.camera.get_uniform()])
        );
    }

    pub fn background_color(&mut self) -> &mut wgpu::Color {
        &mut self.background_color
    }
}
