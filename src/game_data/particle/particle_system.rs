use glam::Vec2;
use rand::Rng;
use crate::{renderer::camera::Camera, utils::gpu_buffer::GpuBuffer, wgpu_context::WgpuContext};

pub struct ParticleSystem {
    vertices: GpuBuffer<glam::Vec2>,
    indices: GpuBuffer<u32>,
    instances: GpuBuffer<glam::Vec2>,
    radiuses: GpuBuffer<f32>,
    colors: GpuBuffer<glam::Vec4>,
    render_pipeline: wgpu::RenderPipeline,
}

impl ParticleSystem {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera) -> Self {
        const NUM_PARTICLES: usize = 100_000;
        const WORLD_WIDTH: f32 = 1920.0;
        const WORLD_HEIGHT: f32 = 1080.0;

        let mut rng = rand::thread_rng();

        let mut instances = Vec::with_capacity(NUM_PARTICLES);
        let mut radiuses = Vec::with_capacity(NUM_PARTICLES);

        for _ in 0..NUM_PARTICLES {
            let x = rng.random_range(0.0..WORLD_WIDTH);
            let y = rng.random_range(0.0..WORLD_HEIGHT);
            instances.push(Vec2::new(x, y));

            let radius = rng.random_range(1.0..4.0);
            radiuses.push(radius);
        }
        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("../../shaders/renderShaders/shader.wgsl"));
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
                buffers: &[
                    wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress, // Size of a Vec2
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![0 => Float32x2], // Vec2 is two 32-bit floats
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![2 => Float32],
                    },

                ],
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


        Self {
            vertices: GpuBuffer::new(
                wgpu_context,
                vec![
                glam::Vec2::new(-0.5, 0.5),
                glam::Vec2::new(0.5, 0.5),
                glam::Vec2::new(0.5, -0.5),
                glam::Vec2::new(-0.5, -0.5),
                ],
                wgpu::BufferUsages::VERTEX
            ),
            indices: GpuBuffer::new(wgpu_context, vec![
                0, 3, 2,
                2, 1, 0
            ],
            wgpu::BufferUsages::INDEX
            ),
            instances: GpuBuffer::new(wgpu_context, instances, wgpu::BufferUsages::VERTEX),
            radiuses: GpuBuffer::new(wgpu_context, radiuses, wgpu::BufferUsages::VERTEX),
            colors: GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX),
            render_pipeline: render_pipeline,
        }
    }

    pub fn draw(&self, wgpu_context: &WgpuContext, camera: &Camera, background_color: wgpu::Color) -> Result<(), wgpu::SurfaceError>{
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
                            load: wgpu::LoadOp::Clear(background_color),
                            store: wgpu::StoreOp::Store,
                        }
                    })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
            render_pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_vertex_buffer(1, self.instances.buffer().slice(..));
            render_pass.set_vertex_buffer(2, self.radiuses.buffer().slice(..));

            render_pass.set_bind_group(0, camera.binding_group(), &[]);
            render_pass.draw_indexed(0..self.indices().len() as u32, 0, 0..self.instances().len() as u32);
        }


        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn vertices(&self) -> &[glam::Vec2] {
        self.vertices.data()
    }

    pub fn indices(&self) -> &[u32] {
        self.indices.data()
    }

    pub fn instances(&self) -> &[glam::Vec2] {
        self.instances.data()
    }

    pub fn radiuses(&self) -> &[f32] {
        self.radiuses.data()
    }

    pub fn color(&self) -> &[glam::Vec4] {
        self.colors.data()
    }
}
