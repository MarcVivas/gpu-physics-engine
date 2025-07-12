use glam::Vec2;
use rand::Rng;
use crate::{renderer::{camera::Camera, renderable::Renderable}, utils::gpu_buffer::GpuBuffer};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;

pub struct ParticleSystem {
    vertices: GpuBuffer<glam::Vec2>,
    indices: GpuBuffer<u32>,
    instances: GpuBuffer<glam::Vec2>,
    velocities: GpuBuffer<glam::Vec2>,
    radius: GpuBuffer<f32>,
    max_radius: f32,
    colors: GpuBuffer<glam::Vec4>,
    render_pipeline: wgpu::RenderPipeline,
    sim_params_buffer: GpuBuffer<SimParams>,
    integration_pass: ComputeShader,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
    _padding: f32,
}

impl ParticleSystem {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera) -> Self {
        const NUM_PARTICLES: usize = 10;
        const WORLD_WIDTH: f32 = 1920.0;
        const WORLD_HEIGHT: f32 = 1080.0;

        let mut rng = rand::rng();

        let mut ins = Vec::with_capacity(NUM_PARTICLES);
        let mut radiuses = Vec::with_capacity(NUM_PARTICLES);
        let mut vels = Vec::with_capacity(NUM_PARTICLES);


        let mut max_radius = f32::MIN;

        for _ in 0..NUM_PARTICLES {
            let x = rng.random_range(0.0..WORLD_WIDTH);
            let y = rng.random_range(0.0..WORLD_HEIGHT);
            let vel_x = rng.random_range(-50.0..50.0); // pixels per second
            let vel_y = rng.random_range(-50.0..50.0);
            ins.push(Vec2::new(x, y));
            vels.push(Vec2::new(vel_x, vel_y));
            let radius = rng.random_range(1.0..4.0);
            if radius > max_radius {
                max_radius = radius;
            }
            radiuses.push(radius);
        }



        let instances = GpuBuffer::new(wgpu_context, ins, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let velocities = GpuBuffer::new(wgpu_context, vels, wgpu::BufferUsages::STORAGE);

        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
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



        let sim_params_buffer = GpuBuffer::new(
            wgpu_context,
            vec![SimParams { delta_time: 0.0, world_width: WORLD_WIDTH, world_height: WORLD_HEIGHT, _padding: 0.0 }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // 3. Create the Compute Pipeline
        let compute_shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        // This layout describes what the compute shader can access.
        let compute_bind_group_layout = wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    // Binding 0: Simulation Parameters (delta_time, etc.)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: The particle positions
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: The particle velocities
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }
        ;

        let integration_pass = ComputeShader::new(
            wgpu_context,
            &compute_shader,
            "cs_main", // Assuming you rename cs_main to be more specific
            &compute_bind_group_layout,
            (64, 1, 1), // The workgroup size from your WGSL
        );




        Self {
            vertices: GpuBuffer::new(
                wgpu_context,
                vec![
                Vec2::new(-0.5, 0.5),
                Vec2::new(0.5, 0.5),
                Vec2::new(0.5, -0.5),
                Vec2::new(-0.5, -0.5),
                ],
                wgpu::BufferUsages::VERTEX
            ),
            indices: GpuBuffer::new(wgpu_context, vec![
                0, 3, 2,
                2, 1, 0
            ],
            wgpu::BufferUsages::INDEX
            ),
            velocities,
            instances,
            radius: GpuBuffer::new(wgpu_context, radiuses, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE),
            max_radius,
            colors: GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX),
            render_pipeline,
            sim_params_buffer,
            integration_pass,
        }
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn vertices(&self) -> &[glam::Vec2] {
        self.vertices.data()
    }

    pub fn indices(&self) -> &[u32] {
        self.indices.data()
    }

    pub fn instances(&self) -> &GpuBuffer<Vec2>{
        &self.instances
    }

    pub fn radius(&self) -> &GpuBuffer<f32> {
        &self.radius
    }

    pub fn color(&self) -> &[glam::Vec4] {
        self.colors.data()
    }

    pub fn get_max_radius(&self) -> f32 {
        self.max_radius
    }
    pub fn add_particles(&mut self, mouse_pos: &Vec2, wgpu_context: &WgpuContext){

        self.instances.push(
            mouse_pos.clone(),
            wgpu_context
        );
        self.velocities.push(
            Vec2::new(rand::random_range(1.0..10.0), rand::random_range(1.0..10.0)),
            wgpu_context
        );

        let rng = rand::random_range(1.0..10.0);
        self.radius.push(
            rng,
            wgpu_context
        );

        self.max_radius = self.max_radius.max(rng);




    }
}

impl Renderable for ParticleSystem {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(1, self.instances.buffer().slice(..));
        render_pass.set_vertex_buffer(2, self.radius.buffer().slice(..));

        render_pass.set_bind_group(0, camera.binding_group(), &[]);
        render_pass.draw_indexed(0..self.indices().len() as u32, 0, 0..self.instances.len() as u32);
    }

    fn update(&mut self, delta_time:f32, world_size:&glam::Vec2, wgpu_context: &WgpuContext) {
        // First, update the delta_time in the uniform buffer
        let sim_params = SimParams {
            delta_time,
            world_width: world_size.x, // Make these constants accessible
            world_height: world_size.y,
            _padding: 0.0,
        };
        wgpu_context.get_queue().write_buffer(
            &self.sim_params_buffer.buffer(),
            0,
            bytemuck::cast_slice(&[sim_params]),
        );

        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        self.integration_pass.dispatch_by_items(
            wgpu_context,
            &mut encoder,
            "Particle Integration Pass",
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.sim_params_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instances.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.velocities.buffer().as_entire_binding(),
                },
            ],
            (self.instances.data().len() as u32, 1, 1),
        );

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
    }
}
