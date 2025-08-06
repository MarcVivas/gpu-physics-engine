use glam::{Vec2, Vec4};
use rand::{random_range, Rng};
use wgpu::hal::DynCommandEncoder;
use winit::event::{ElementState, MouseButton};
use crate::{renderer::{camera::Camera, renderable::Renderable}, utils::gpu_buffer::GpuBuffer};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;


use crate::utils::gpu_timer::GpuTimer;

const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);
pub struct ParticleSystem {
    vertices: GpuBuffer<Vec2>,
    indices: GpuBuffer<u32>,
    current_positions: GpuBuffer<Vec2>,
    previous_positions: GpuBuffer<Vec2>,
    velocities: GpuBuffer<Vec2>,
    radius: GpuBuffer<f32>,
    max_radius: f32,
    colors: GpuBuffer<Vec4>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    sim_params_buffer: GpuBuffer<SimParams>,
    integration_pass: ComputeShader,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    delta_time: f32,
    world_width: f32,
    world_height: f32,
    is_mouse_pressed: u32,
    mouse_pos: Vec2,
}

impl ParticleSystem {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_size: Vec2) -> Self {
        const NUM_PARTICLES: usize = 5000000;
        let world_width: f32 = world_size.x;
        let world_height: f32 = world_size.y;

        let mut rng = rand::rng();

        let mut positions = Vec::with_capacity(NUM_PARTICLES);
        let mut radiuses = Vec::with_capacity(NUM_PARTICLES);
        let mut vels = Vec::with_capacity(NUM_PARTICLES);
        let mut colors = Vec::with_capacity(NUM_PARTICLES);

        let mut max_radius = f32::MIN;

        for _ in 0..NUM_PARTICLES {
            let x = rng.random_range(0.0..world_width);
            let y = rng.random_range(0.0..world_height);
            let vel_x = rng.random_range(-50.0..50.0); // pixels per second
            let vel_y = rng.random_range(-50.0..50.0);
            positions.push(Vec2::new(x, y));
            vels.push(Vec2::new(vel_x, vel_y));
            let radius = rng.random_range(0.5..=0.5) as f32;
            colors.push(glam::vec4(rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), 1.0));
            if radius > max_radius {
                max_radius = radius;
            }
            radiuses.push(radius);
        }



        let current_positions = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let previous_positions = GpuBuffer::new(wgpu_context, positions, wgpu::BufferUsages::STORAGE);
        let velocities = GpuBuffer::new(wgpu_context, vels, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);

        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("particle_system.wgsl"));
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
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vec4>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![4 => Float32x2],
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
            vec![SimParams { delta_time: 0.0, world_width, world_height, is_mouse_pressed: 0, mouse_pos: Vec2::new(0.0, 0.0) }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );


        let radius = GpuBuffer::new(wgpu_context, radiuses, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        
        let integration_pass = Self::gen_integration_pass(wgpu_context, &velocities, &sim_params_buffer, &current_positions, &previous_positions, &radius);

        Self {
            vertices: Self::gen_model_vertices(wgpu_context),
            indices: Self::gen_model_indices(wgpu_context),
            velocities,
            current_positions,
            previous_positions,
            radius,
            max_radius,
            colors: GpuBuffer::new(wgpu_context, colors, wgpu::BufferUsages::VERTEX),
            render_pipeline: Some(render_pipeline),
            sim_params_buffer,
            integration_pass,
        }
    }

    pub fn new_from_buffers(wgpu_context: &WgpuContext, current_positions: GpuBuffer<Vec2>, radii: GpuBuffer<f32>) -> Self {
        let velocities = GpuBuffer::new(
            wgpu_context,
            vec![Vec2::ZERO; current_positions.len()],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        );
        let max_radius: f32 = radii.data().iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap().clone();
        let sim_params_buffer = GpuBuffer::new(wgpu_context, vec![SimParams { delta_time: 0.0, world_width: 1920.0, world_height: 1080.0, is_mouse_pressed: 0, mouse_pos: Vec2::new(0.0, 0.0) }], wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let previous_positions = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::STORAGE);
        let integration_pass = Self::gen_integration_pass(wgpu_context, &velocities, &sim_params_buffer, &current_positions, &previous_positions, &radii);
        Self {
            vertices: Self::gen_model_vertices(wgpu_context),
            indices: Self::gen_model_indices(wgpu_context),
            velocities,
            previous_positions,
            current_positions,
            radius: radii,
            max_radius,
            colors: GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX),
            render_pipeline: None,
            sim_params_buffer,
            integration_pass,
        }
    }

    fn gen_model_vertices(wgpu_context: &WgpuContext) -> GpuBuffer<Vec2>{
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

    fn gen_model_indices(wgpu_context: &WgpuContext) -> GpuBuffer<u32>{
        GpuBuffer::new(wgpu_context, vec![
            0, 3, 2,
            2, 1, 0
        ], wgpu::BufferUsages::INDEX)
    }

    fn gen_integration_pass(wgpu_context: &WgpuContext, velocities: &GpuBuffer<Vec2>, sim_params_buffer: &GpuBuffer<SimParams>, particle_positions: &GpuBuffer<Vec2>, previous_positions: &GpuBuffer<Vec2>, radius: &GpuBuffer<f32>) -> ComputeShader {

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
                // Binding 1: The particles's current positions
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
                // Binding 2: The particles velocities
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
                // Binding 3: The particles's previous positions
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: The particles's radius
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true }, 
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        let bind_group_layout = wgpu_context.get_device().create_bind_group_layout(&compute_bind_group_layout);

        // Create bind group
        let bind_group = wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_params_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: velocities.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: previous_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: radius.buffer().as_entire_binding(),
                    },
                ],
            }
        );
        
        ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("particle_system.wgsl"),
            "verlet_integration", 
            bind_group,
            bind_group_layout,
            WORKGROUP_SIZE, 
        )
    }
    pub fn len(&self) -> usize {
        self.current_positions.len()
    }

    pub fn vertices(&self) -> &[Vec2] {
        self.vertices.data()
    }

    pub fn indices(&self) -> &[u32] {
        self.indices.data()
    }

    pub fn positions(&self) -> &GpuBuffer<Vec2>{
        &self.current_positions
    }

    pub fn radius(&self) -> &GpuBuffer<f32> {
        &self.radius
    }

    pub fn color(&self) -> &[Vec4] {
        self.colors.data()
    }

    pub fn get_max_radius(&self) -> f32 {
        self.max_radius
    }
    pub fn add_particles(&mut self, mouse_pos: &Vec2, wgpu_context: &WgpuContext){
        
        
        for i in 0..100 {
            // Generate a random angle (0 to 2*PI radians)
            let angle = random_range(0.0..std::f32::consts::TAU); // TAU is 2*PI

            // Generate a random radius (from mouse_pos)
            // Start the minimum radius higher to avoid center clumping
            // And potentially make the maximum radius larger or adjust its scaling
            let min_radius = 10.0 ; // Minimum distance from the center
            let max_radius = 50.0 + (i as f32 * 1.5); // Example: Gradually increase max radius
            let radius = random_range(min_radius..=max_radius);


            // Convert polar coordinates to Cartesian (x, y)
            let offset_x = radius * angle.cos();
            let offset_y = radius * angle.sin();

            let pos: Vec2 = mouse_pos + Vec2::new(offset_x, offset_y);

            self.current_positions.push(pos.clone(), wgpu_context);
            self.previous_positions.push(pos, wgpu_context);
            // ... rest of your particles property generation
            self.velocities.push(
                Vec2::new(random_range(1.0..10.0), random_range(1.0..10.0)),
                wgpu_context
            );

            let rng_radius_particle = random_range(1.0..=1.0); // Renamed to avoid conflict
            self.radius.push(
                rng_radius_particle,
                wgpu_context
            );

            self.max_radius = self.max_radius.max(rng_radius_particle);

            self.colors.push(
                glam::vec4(random_range(0.3..1.0), random_range(0.3..1.0), random_range(0.3..1.0), 1.0),
                wgpu_context
            );
        }
        
        self.integration_pass.update_binding_group(wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: self.integration_pass.get_bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.sim_params_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.current_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.velocities.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.previous_positions.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.radius.buffer().as_entire_binding(),
                    },
                ],
            }));
        println!("Total particles: {}", self.current_positions.len());
    }
    pub fn mouse_click_callback(&mut self, wgpu_context: &WgpuContext, mouse_state: &ElementState, position: Vec2){
        let old_sim_params = self.sim_params_buffer.data()[0];

        let sim_params = SimParams {
            delta_time: old_sim_params.delta_time,
            world_width: old_sim_params.world_width,
            world_height: old_sim_params.world_height,
            is_mouse_pressed: mouse_state.is_pressed() as u32,
            mouse_pos: position,
        };
        self.sim_params_buffer.replace_elem(sim_params, 0, wgpu_context);
    }
    pub fn mouse_move_callback(&mut self, wgpu_context: &WgpuContext, position: Vec2){
        let old_sim_params = self.sim_params_buffer.data()[0];
        
        if old_sim_params.is_mouse_pressed == 1 {
            let sim_params = SimParams {
                delta_time: old_sim_params.delta_time,
                world_width: old_sim_params.world_width,
                world_height: old_sim_params.world_height,
                is_mouse_pressed: old_sim_params.is_mouse_pressed,
                mouse_pos: position,
            };
            self.sim_params_buffer.replace_elem(sim_params, 0, wgpu_context);
        }


    }
}



impl Renderable for ParticleSystem {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        render_pass.set_pipeline(self.render_pipeline.as_ref().expect("Render pipeline not set"));
        render_pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        render_pass.set_index_buffer(self.indices.buffer().slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(1, self.current_positions.buffer().slice(..));
        render_pass.set_vertex_buffer(2, self.radius.buffer().slice(..));
        render_pass.set_vertex_buffer(3, self.colors.buffer().slice(..));
        render_pass.set_vertex_buffer(4, self.velocities.buffer().slice(..));

        render_pass.set_bind_group(0, camera.binding_group(), &[]);
        render_pass.draw_indexed(0..self.indices().len() as u32, 0, 0..self.current_positions.len() as u32);
    }

    #[cfg(feature = "benchmark")]
    fn update(&mut self, delta_time:f32, world_size:&Vec2, wgpu_context: &WgpuContext, gpu_timer: &mut GpuTimer) {
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

        gpu_timer.begin_frame();
        gpu_timer.scope("Integration pass", &mut encoder, |encoder| {
            self.integration_pass.dispatch_by_items(
                encoder,
                (self.current_positions.data().len() as u32, 1, 1),
            );
        });
        

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        
        gpu_timer.end_frame(wgpu_context.get_device(), wgpu_context.get_queue());

    }

    #[cfg(not(feature = "benchmark"))]
    fn update(&mut self, delta_time:f32, _world_size:&Vec2, wgpu_context: &WgpuContext) {
        let old_sim_params = self.sim_params_buffer.data()[0];
        
        // First, update the delta_time in the uniform buffer
        let sim_params = SimParams {
            delta_time,
            world_width: old_sim_params.world_width,
            world_height: old_sim_params.world_height,
            is_mouse_pressed: old_sim_params.is_mouse_pressed, 
            mouse_pos: old_sim_params.mouse_pos,
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
            &mut encoder,
            (self.current_positions.data().len() as u32, 1, 1),
        );
        
        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        
    }


}
