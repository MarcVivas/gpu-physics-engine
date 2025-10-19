use glam::{Vec2, Vec4};
use crate::renderer::renderable::Renderable;
use crate::renderer::camera::Camera;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::renderer::wgpu_context::WgpuContext;

pub struct Lines {
    vertices: GpuBuffer<glam::Vec2>,        // Line endpoints
    colors: GpuBuffer<glam::Vec4>,          // Per-vertex colors
    thicknesses: GpuBuffer<f32>,            // Per-vertex thickness
    render_pipeline: wgpu::RenderPipeline,
}

impl Lines {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera) -> Self {

        let vertices = Vec::new();
        let colors = Vec::new();
        let thicknesses = Vec::new();



        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("line.wgsl"));
        let render_pipeline_layout = wgpu_context.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[&camera.camera_bind_group_layout()],
            push_constant_ranges: &[],
        });

        let render_pipeline = wgpu_context.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label: Some("Line Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Buffer 0: Vertex positions
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vec2>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    // Buffer 1: Colors
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vec4>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4],
                    },
                    // Buffer 2: Thickness
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<f32>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
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
                topology: wgpu::PrimitiveTopology::LineList, // Direct lines rendering
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for lines
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
                vertices,
                wgpu::BufferUsages::VERTEX,
            ),
            colors: GpuBuffer::new(
                wgpu_context,
                colors,
                wgpu::BufferUsages::VERTEX,
            ),
            thicknesses: GpuBuffer::new(
                wgpu_context,
                thicknesses,
                wgpu::BufferUsages::VERTEX,
            ),
            render_pipeline,
        }
    }

    pub fn push(&mut self, wgpu_context: &WgpuContext, start: Vec2, end: Vec2, color: Vec4, thickness: f32) {
        self.colors.push(color, wgpu_context);
        self.colors.push(color, wgpu_context);

        self.thicknesses.push(thickness, wgpu_context);
        self.thicknesses.push(thickness, wgpu_context);

        self.vertices.push(start, wgpu_context);
        self.vertices.push(end, wgpu_context);
    }

    pub fn push_all(&mut self, wgpu_context: &WgpuContext, positions: &[Vec2], color: &[Vec4], thickness: &[f32]) {
        self.colors.push_all(color, wgpu_context);
        self.thicknesses.push_all(thickness, wgpu_context);
        self.vertices.push_all(positions, wgpu_context);
    }

    }

impl Renderable for Lines {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera) {
        if self.vertices.data().len() == 0 {return;}
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_vertex_buffer(1, self.colors.buffer().slice(..));
        render_pass.set_vertex_buffer(2, self.thicknesses.buffer().slice(..));
        render_pass.set_bind_group(0, camera.binding_group(), &[]);
        render_pass.draw(0..self.vertices.data().len() as u32, 0..1);
    }
}
