use glam::{Vec2, Vec4};
use rand::{random_range, Rng};
use wgpu_profiler::GpuProfiler;
use winit::event::{ElementState};
use crate::{renderer::{camera::Camera, renderable::Renderable}, utils::gpu_buffer::GpuBuffer};
use crate::particles::{particle_kernels::ParticleKernels, particle_buffers::ParticleBuffers};
use crate::particles::particle_drawer::ParticleDrawer;
use crate::particles::particle_push_constants::{SimParams};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;



pub const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

pub struct ParticleSystem {
    particle_buffers: ParticleBuffers,
    particle_binding_group: BindResources,
    particle_drawer: Option<ParticleDrawer>, 
    max_radius: f32,
    sim_params: SimParams,
    particle_kernels: ParticleKernels
}


impl ParticleSystem {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_size: Vec2) -> Self {
        const NUM_PARTICLES: usize = 600000;
        
        let (buffers, max_radius) = Self::generate_initial_particles(wgpu_context, &world_size, NUM_PARTICLES);


        let sim_params = SimParams { delta_time: 0.0, world_width: world_size.x, world_height: world_size.y, is_mouse_pressed: 0, mouse_pos: Vec2::new(0.0, 0.0), num_particles: NUM_PARTICLES as u32 };

        let particle_binding_group = Self::generate_particle_binding_group(&wgpu_context, &buffers);

        let particle_kernels = ParticleKernels::new(wgpu_context, NUM_PARTICLES, &particle_binding_group, &buffers);
       
        let particle_drawer = ParticleDrawer::new(wgpu_context, &particle_binding_group, &camera);

        Self {
            particle_buffers: buffers,
            particle_drawer: Some(particle_drawer),
            max_radius,
            sim_params,
            particle_kernels,
            particle_binding_group,
        }
    }

    pub fn new_from_buffers(wgpu_context: &WgpuContext, current_positions: GpuBuffer<Vec2>, radii: GpuBuffer<f32>) -> Self {
        let total_particles = current_positions.len();
        let max_radius: f32 = radii.data().iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap().clone();
        let sim_params = SimParams { delta_time: 0.0, world_width: 1920.0, world_height: 1080.0, is_mouse_pressed: 0, mouse_pos: Vec2::new(0.0, 0.0), num_particles: total_particles as u32 };
        let previous_positions = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::STORAGE);
        let colors = GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX);
        let current_positions_pong = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::STORAGE);
        let cell_ids = GpuBuffer::new(wgpu_context, vec![0; total_particles], wgpu::BufferUsages::STORAGE);
        let particle_ids = GpuBuffer::new(wgpu_context, vec![0; total_particles], wgpu::BufferUsages::STORAGE);
        
        let buffers = ParticleBuffers{
            current_positions_ping: current_positions,
            current_positions_pong,
            previous_positions, 
            radii,
            colors,
            cell_ids,
            particle_ids
        };

        let particle_binding_group = Self::generate_particle_binding_group(&wgpu_context, &buffers);

        let particle_kernels = ParticleKernels::new(wgpu_context, total_particles, &particle_binding_group, &buffers);
    
        
        Self {
            particle_buffers: buffers,
            particle_drawer: None,
            max_radius,
            sim_params,
            particle_kernels,
            particle_binding_group,
        }
    }

    /// Generates the initial particle data and buffers.
    fn generate_initial_particles(wgpu_context: &WgpuContext, world_size: &Vec2, num_particles: usize) -> (ParticleBuffers, f32){
        let world_width: f32 = world_size.x;
        let world_height: f32 = world_size.y;

        let mut rng = rand::rng();

        let mut positions = Vec::with_capacity(num_particles);
        let mut radiuses = Vec::with_capacity(num_particles);
        let mut colors = Vec::with_capacity(num_particles);

        let mut max_radius = f32::MIN;

        for _ in 0..num_particles {
            let x = rng.random_range(0.0..world_width);
            let y = rng.random_range(0.0..world_height);
            positions.push(Vec2::new(x, y));
            let radius = rng.random_range(0.5..= 0.5) as f32;
            colors.push(glam::vec4(rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), 1.0));
            if radius > max_radius {
                max_radius = radius;
            }
            radiuses.push(radius);
        }


        let particle_ids = GpuBuffer::new(wgpu_context, vec![0; positions.len()], wgpu::BufferUsages::STORAGE);
        let current_positions = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let current_positions_pong = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let cell_ids = GpuBuffer::new(wgpu_context, vec![0; positions.len()], wgpu::BufferUsages::STORAGE);
        let previous_positions = GpuBuffer::new(wgpu_context, positions, wgpu::BufferUsages::STORAGE);
        let radius = GpuBuffer::new(wgpu_context, radiuses, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        (ParticleBuffers{
            current_positions_ping: current_positions,
            current_positions_pong,
            previous_positions, 
            radii: radius,
            colors: GpuBuffer::new(wgpu_context, colors, wgpu::BufferUsages::VERTEX),
            cell_ids,
            particle_ids
        }, max_radius)

    }



    fn generate_particle_binding_group(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers) -> BindResources {
        let bind_group_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout Descriptor"),
            entries: &[
                // Binding 0: The particles' current positions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: The particles' previous positions
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // false means read-write
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

        let bind_group_layout = wgpu_context.get_device().create_bind_group_layout(&bind_group_layout_descriptor);

        // Create bind group
        let bind_group = wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers.current_positions_ping.buffer().as_entire_binding(),
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
        );

        BindResources{
            bind_group_layout,
            bind_group,
        }
    }
    
    pub fn len(&self) -> usize {
        self.particle_buffers.current_positions_ping.len()
    }

    pub fn positions(&self) -> &GpuBuffer<Vec2>{
        &self.particle_buffers.current_positions_ping
    }

    pub fn radius(&self) -> &GpuBuffer<f32> {
        &self.particle_buffers.radii
    }

    pub fn color(&self) -> &[Vec4] {
        self.particle_buffers.colors.data()
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

            self.particle_buffers.current_positions_ping.push(pos.clone(), wgpu_context);
            self.particle_buffers.current_positions_pong.push(pos.clone(), wgpu_context);
            self.particle_buffers.previous_positions.push(pos, wgpu_context);
            self.particle_buffers.cell_ids.push(0, wgpu_context);

            let rng_radius_particle = random_range(1..=3) as f32; 
            self.particle_buffers.radii.push(
                rng_radius_particle,
                wgpu_context
            );

            self.max_radius = self.max_radius.max(rng_radius_particle);

            self.particle_buffers.colors.push(
                glam::vec4(random_range(0.3..1.0), random_range(0.3..1.0), random_range(0.3..1.0), 1.0),
                wgpu_context
            );
        }
        
        self.sim_params.num_particles = self.particle_buffers.current_positions_ping.len() as u32;
        self.particle_binding_group =  Self::generate_particle_binding_group(wgpu_context, &self.particle_buffers);
        
        println!("Total particles: {}", self.particle_buffers.current_positions_ping.len());
    }
    pub fn mouse_click_callback(&mut self, mouse_state: &ElementState, position: Vec2){
        self.sim_params.is_mouse_pressed = mouse_state.is_pressed() as u32;
        self.sim_params.mouse_pos = position;
    }
    pub fn mouse_move_callback(&mut self, position: Vec2){
        if self.sim_params.is_mouse_pressed == 1 {
            self.sim_params.mouse_pos = position;
        }
    }


    pub fn sort_by_cell_id(&mut self, encoder: &mut wgpu::CommandEncoder, cell_size: f32){
        todo!();
        //self.particle_kernels.gpu_sorter.sort(encoder, None);
    }

    pub fn update_positions(&mut self, delta_time:f32, wgpu_context: &WgpuContext, gpu_profiler: &mut GpuProfiler) {
        self.sim_params.delta_time = delta_time;


        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        {
            let mut scope = gpu_profiler.scope("Particle integration pass", &mut encoder);
            self.particle_kernels.integration_pass.dispatch_by_items(
                &mut scope,
                (self.particle_buffers.current_positions_ping.data().len() as u32, 1, 1),
                Some(vec![(0, bytemuck::bytes_of(&self.sim_params))]),
                &self.particle_binding_group.bind_group,
            );
        }
        gpu_profiler.resolve_queries(&mut encoder);

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
    }

}



impl Renderable for ParticleSystem {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        self.particle_drawer.as_ref().expect("Particle drawer null").draw(render_pass, camera, &self.particle_binding_group, self.len() as u32);
    }

}
