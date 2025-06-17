
use crate::renderer::camera::{Camera};
use crate::renderer::renderable::Renderable;
use crate::wgpu_context::WgpuContext;
use crate::game_data::particle::particle_system::ParticleSystem;
use crate::game_data::line::lines::Lines;

// Manages multiple render pipelines
pub struct Renderer {
    rendering_pipelines: Vec<wgpu::RenderPipeline>,
    background_color: wgpu::Color,
    camera: Camera,
    particles: ParticleSystem,
    lines: Lines
}



impl Renderer {
    pub fn new(wgpu_context: &WgpuContext, world_size: &glam::Vec2) -> Option<Self> {
        // 4. Create the camera with the calculated values
        let camera = Camera::new(world_size, &wgpu_context);
        let particles: ParticleSystem = ParticleSystem::new(&wgpu_context, &camera);
        let lines = Lines::new(wgpu_context, &camera);

        Some(Self {
            rendering_pipelines: vec![

            ],
            background_color: wgpu::Color::BLACK,
            camera,
            particles,
            lines
        })
    }

    pub fn add_pipeline(&mut self, pipeline: wgpu::RenderPipeline){
        self.rendering_pipelines.push(pipeline);
    }

    pub fn render(&self, wgpu_context: &WgpuContext) -> Result<(), wgpu::SurfaceError>{
        wgpu_context.get_window().request_redraw();

        // We can't render unless the window is configured
        if !wgpu_context.is_surface_configured() {
            return Ok(());
        }

        // This is where we render
        let output = wgpu_context.get_surface().get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // We need an encoder to create the actual commands to send to the gpu
        let mut encoder = wgpu_context.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Render Encoder"),
        });

        // Use encoder to create a RenderPass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment{
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color),
                            store: wgpu::StoreOp::Store,
                        }
                    })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw particle system
            self.particles.draw(&mut render_pass, &self.camera);
            self.lines.draw(&mut render_pass, &self.camera);
        }

        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn process_events(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.camera.process_events(event)
    }

    pub fn update(&mut self, dt: f32, wgpu_context: &WgpuContext) {
        // Update camera based on input and delta time
        self.camera.update(dt);

        // Update camera matrices and upload to GPU
        self.update_camera_matrices(wgpu_context);
    }

    pub fn update_camera_matrices(&mut self, wgpu_context: &WgpuContext) {
        self.camera.build_view_projection_matrix(
            wgpu_context.window_size().width as f32,
            wgpu_context.window_size().height as f32
        );
        wgpu_context.get_queue().write_buffer(
            &self.camera.camera_buffer(),
            0, // offset
            bytemuck::cast_slice(&[*self.camera.get_uniform()])
        );
    }

    pub fn update_camera(&mut self, wgpu_context: &WgpuContext){
        self.update_camera_matrices(wgpu_context);
    }

    pub fn background_color(&mut self) -> &mut wgpu::Color {
        &mut self.background_color
    }
}
