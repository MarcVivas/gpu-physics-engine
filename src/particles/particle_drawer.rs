use glam::Vec2;
use crate::renderer::camera::Camera;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct ParticleDrawer{
    render_pipeline: Option<wgpu::RenderPipeline>,
    vertices: GpuBuffer<Vec2>,
    indices: GpuBuffer<u32>,
}

impl ParticleDrawer{
    pub fn new(wgpu_context: &WgpuContext, particle_binding_group: &BindResources, camera: &Camera ) -> Self {
        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("particle_drawer.wgsl"));
        let render_pipeline_layout = wgpu_context.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&particle_binding_group.bind_group_layout, &camera.camera_bind_group_layout()],
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
                        array_stride: std::mem::size_of::<glam::Vec2>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
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
        
        
        let vertices =  Self::create_model_vertices(wgpu_context);
        let indices = Self::create_model_indices(wgpu_context);
        
        Self {
            render_pipeline: Some(render_pipeline),
            vertices,
            indices,
        }
        
    }

    fn create_model_vertices(wgpu_context: &WgpuContext) -> GpuBuffer<Vec2>{
        GpuBuffer::new(
            wgpu_context,
            vec![
                Vec2::new(-0.5, 0.5),
                Vec2::new(0.5, 0.5),
                Vec2::new(0.5, -0.5),
                Vec2::new(-0.5, -0.5),
            ],
            wgpu::BufferUsages::VERTEX
        )
    }

    fn create_model_indices(wgpu_context: &WgpuContext) -> GpuBuffer<u32>{
        GpuBuffer::new(wgpu_context, vec![
            0, 3, 2,
            2, 1, 0
        ], wgpu::BufferUsages::INDEX)
    }

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera, particle_binding_group: &BindResources, num_particles: u32){
        render_pass.set_pipeline(self.render_pipeline.as_ref().expect("Render pipeline not set"));
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);

        render_pass.set_bind_group(0, &particle_binding_group.bind_group, &[]);
        render_pass.set_bind_group(1, camera.binding_group(), &[]);
        render_pass.draw_indexed(0..self.get_indices().len() as u32, 0, 0..num_particles);
    }
    
    fn get_indices(&self) -> &Vec<u32>{
        self.indices.data()
    }


}