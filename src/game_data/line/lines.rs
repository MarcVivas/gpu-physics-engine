use glam::Vec2;
use crate::renderer::renderable::Renderable;
use crate::renderer::camera::Camera;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::renderer::wgpu_context::WgpuContext;
use rand::Rng;

pub struct Lines {
    vertices: GpuBuffer<glam::Vec2>,        // Line endpoints
    colors: GpuBuffer<glam::Vec4>,          // Per-vertex colors
    thicknesses: GpuBuffer<f32>,            // Per-vertex thickness
    render_pipeline: wgpu::RenderPipeline,
}

impl Lines {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera) -> Self {
        const TOTAL_LINES: usize = 4;
        let mut vertices = Vec::new();
        let mut colors = Vec::new();
        let mut thicknesses = Vec::new();

        let mut rng = rand::rng();

        // Initialize lines in the for loop
        for i in 0..TOTAL_LINES {
            // Create line endpoints
            let start_x = 100.0 + (i as f32 * 150.0);
            let start_y = 100.0;
            let end_x = start_x + 100.0;
            let end_y = 300.0;

            // Add both endpoints for this line
            vertices.push(glam::Vec2::new(start_x, start_y));
            vertices.push(glam::Vec2::new(end_x, end_y));

            // Random color for this line (same color for both endpoints)
            let line_color = glam::Vec4::new(
                rng.random_range(0.0..1.0), // Red
                rng.random_range(0.0..1.0), // Green
                rng.random_range(0.0..1.0), // Blue
                1.0                         // Alpha
            );
            colors.push(line_color);
            colors.push(line_color); // Same color for both endpoints

            // Random thickness for this line (same thickness for both endpoints)
            let line_thickness = rng.random_range(1.0..5.0);
            thicknesses.push(line_thickness);
            thicknesses.push(line_thickness); // Same thickness for both endpoints
        }

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
                        array_stride: std::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    // Buffer 1: Colors
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<glam::Vec4>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4],
                    },
                    // Buffer 2: Thickness
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
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
                topology: wgpu::PrimitiveTopology::LineList, // Direct line rendering
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
}

impl Renderable for Lines {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_vertex_buffer(1, self.colors.buffer().slice(..));
        render_pass.set_vertex_buffer(2, self.thicknesses.buffer().slice(..));
        render_pass.set_bind_group(0, camera.binding_group(), &[]);
        render_pass.draw(0..self.vertices.data().len() as u32, 0..1);
    }

    fn update(&self, delta_time: f32, world_size: &Vec2, wgpu_context: &WgpuContext) {
        
    }
}
