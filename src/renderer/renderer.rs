use wgpu_profiler::GpuProfiler;
use winit::dpi::PhysicalPosition;
use winit::event::MouseScrollDelta;
use winit::keyboard::{KeyCode};
use crate::renderer::camera::{Camera};
use crate::renderer::renderable::Renderable;
use crate::renderer::wgpu_context::WgpuContext;

// Manages multiple render pipelines
pub struct Renderer {
    background_color: wgpu::Color,
    camera: Camera,
}



impl Renderer {
    pub fn new(wgpu_context: &WgpuContext, world_size: &glam::Vec2) -> Option<Self> {
        // 4. Create the camera with the calculated values
        let camera = Camera::new(world_size, &wgpu_context);

        Some(Self {
            background_color: wgpu::Color::BLACK,
            camera,
        })
    }
    pub fn render(&self, wgpu_context: &WgpuContext, renderables: &[&dyn Renderable], gpu_profiler: &mut GpuProfiler) -> Result<(), wgpu::SurfaceError>{
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
            let mut scope_encoder = gpu_profiler.scope("Render pass", &mut encoder);
            let mut render_pass = scope_encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment{
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw all renderables
            for renderable in renderables {
                renderable.draw(&mut render_pass, &self.camera);
            }
        }
        
        gpu_profiler.resolve_queries(&mut encoder);
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }
    
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool){
        self.camera.move_camera(key, is_pressed);
    }

    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.camera.zoom_camera(mouse_scroll_delta);   
    }

    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        self.camera.set_camera_zoom_position(pos);
    }
    

    // Update renderables
    pub fn update(&mut self, dt: f32, wgpu_context: &WgpuContext, gpu_profiler: &mut GpuProfiler) {
        // Update camera based on input and delta time
        self.camera.update(dt, &wgpu_context.window_size());
        // Update camera matrices and upload to GPU
        self.update_camera_matrices(wgpu_context);
    }

    fn update_camera_matrices(&mut self, wgpu_context: &WgpuContext) {
        self.camera.build_view_projection_matrix(
            &wgpu_context.window_size(),
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
