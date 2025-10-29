use glam::Vec2;
use wgpu::{BindGroup, BindGroupLayout};
use crate::particles::particle_buffers::ParticleBuffers;
use crate::renderer::camera::Camera;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct ParticleDrawer{
    render_pipeline: Option<wgpu::RenderPipeline>,
    vertices: GpuBuffer<Vec2>,
    indices: GpuBuffer<u32>,
    bind_resources: BindResources,
}

impl ParticleDrawer{
    pub fn new(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, camera: &Camera ) -> Self {
        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("particle_drawer.wgsl"));
        let bind_resources = Self::create_binding_resources(wgpu_context, particle_buffers);
        let render_pipeline_layout = wgpu_context.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_resources.bind_group_layout, &camera.camera_bind_group_layout()],
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
            bind_resources,
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

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera, num_particles: u32){
        render_pass.set_pipeline(self.render_pipeline.as_ref().expect("Render pipeline not set"));
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);

        render_pass.set_bind_group(0, &self.bind_resources.bind_group, &[]);
        render_pass.set_bind_group(1, camera.binding_group(), &[]);
        render_pass.draw_indexed(0..self.get_indices().len() as u32, 0, 0..num_particles);
    }
    
    fn get_indices(&self) -> &Vec<u32>{
        self.indices.data()
    }

    fn create_binding_resources(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers) -> BindResources {
        let bind_group_layout = Self::create_binding_group_layout(wgpu_context);
        let bind_group = Self::create_bind_group(wgpu_context, &bind_group_layout, particle_buffers);

        BindResources{
            bind_group_layout,
            bind_group,
        }
    }

    fn create_bind_group(wgpu_context: &WgpuContext, bind_group_layout: &BindGroupLayout, particle_buffers: &ParticleBuffers) -> BindGroup {
        wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers.current_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers.previous_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers.radii.buffer().as_entire_binding(),
                    },
                ],
            }
        )
    }

    fn create_binding_group_layout(wgpu_context: &WgpuContext) -> BindGroupLayout{
        let bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout Descriptor"),
            entries: &[
                // Binding 0: The particles' current positions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: The particles' previous positions
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: The particles' radius
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        wgpu_context.get_device().create_bind_group_layout(&bind_group_layout_descriptor)
    }

    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers) {
        self.bind_resources.bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, particle_buffers);
    }


}