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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    particles: ParticleSystem,
    num_indices: u32,
}



impl Renderer {
    pub fn new(wgpu_context: &WgpuContext) -> Option<Self> {

        let particles: ParticleSystem = ParticleSystem::new();

        let vertex_buffer = wgpu_context.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex buffer 2"),
            contents: bytemuck::cast_slice(particles.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = wgpu_context.get_device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(particles.indices()),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_vertices = particles.vertices().len() as u32;
        let num_indices = particles.indices().len() as u32;

        // 1. Calculate the bounding box of the vertices
        let (min_x, max_x, min_y, max_y) = particles.vertices().iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y), vertex| {
                (
                    min_x.min(vertex.x),
                    max_x.max(vertex.x),
                    min_y.min(vertex.y),
                    max_y.max(vertex.y),
                )
            },
        );

        // 2. Calculate the center of the bounding box
        let center = Vec3::new(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            0.0
        );

        // 3. Calculate the required zoom to fit the object on screen
        let world_width = max_x - min_x;
        let world_height = max_y - min_y;

        let window_size = wgpu_context.window_size();

        let screen_width = window_size.width as f32;
        let screen_height = window_size.height as f32;

        // Calculate zoom based on width and height, and pick the smaller one to ensure it all fits
        let zoom_x = screen_width / world_width;
        let zoom_y = screen_height / world_height;
        let zoom = zoom_x.min(zoom_y) * 0.9; // Use 90% of the screen for some padding

        // 4. Create the camera with the calculated values
        let camera = Camera::new(center, zoom, &wgpu_context);

        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("../shaders/renderShaders/shader.wgsl"));
        let render_pipeline_layout = wgpu_context.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera.camera_bind_group_layout()],
            push_constant_ranges: &[],
        });

        let render_pipeline = wgpu_context.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label: Some("Render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress, // Size of a Vec2
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![0 => Float32x2], // Vec2 is two 32-bit floats
                            }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState{
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState{
                    format: wgpu_context.get_surface_config().format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            }),

            primitive: wgpu::PrimitiveState{
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,

            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Some(Self {
            rendering_pipelines: vec![
                render_pipeline
            ],
            background_color: wgpu::Color::BLACK,
            camera,
            vertex_buffer,
            index_buffer,
            particles,
            num_indices,
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

        for pipeline in self.rendering_pipelines.iter(){
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

                render_pass.set_pipeline(pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, self.camera.binding_group(), &[]);
                render_pass.draw_indexed(0..self.particles.indices().len() as u32, 0, 0..1);
            }
        }

        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
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
