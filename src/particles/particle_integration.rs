use glam::Vec2;
use wgpu::{BindGroup, BindGroupLayout, PushConstantRange};
use wgpu_profiler::GpuProfiler;
use winit::event::ElementState;
use crate::particles::particle_buffers::ParticleBuffers;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;


const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);

pub struct ParticleIntegration {
    integration_pass: ComputeShader,
    bind_resources: BindResources,
    sim_params: SimParams,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    pub delta_time: f32,
    pub world_width: f32,
    pub world_height: f32,
    pub is_mouse_pressed: u32,
    pub mouse_pos: Vec2,
    pub num_particles: u32,
}




impl ParticleIntegration {
    pub fn new(wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers, world_size: &Vec2) -> Self {
        let bind_resources = Self::create_binding_resources(&wgpu_context, &particle_buffers);
        let integration_pass = Self::create_integration_pass(wgpu_context, &bind_resources);

        let sim_params = SimParams { 
            delta_time: 0.0, 
            world_width: world_size.x, 
            world_height: world_size.y, 
            is_mouse_pressed: 0, 
            mouse_pos: Vec2::new(0.0, 0.0), 
            num_particles: particle_buffers.current_positions.len() as u32 };


        Self {
            integration_pass,
            bind_resources,
            sim_params,
        }
    }

    /// Creates the integration kernel
    fn create_integration_pass(wgpu_context: &WgpuContext, particle_binding_group: &BindResources) -> ComputeShader {
        ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("particle_integration.wgsl"),
            "verlet_integration",
            &particle_binding_group.bind_group_layout,
            WORKGROUP_SIZE,
            &vec![],
            &vec![
                PushConstantRange{
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..size_of::<SimParams>() as u32
                }
            ]
        )
    }
    
    pub fn update_positions(&mut self, wgpu_context: &WgpuContext, gpu_profiler: &mut GpuProfiler, delta_time: f32){
        self.sim_params.delta_time = delta_time;
        
        // Create a command encoder to build the command buffer
        let mut encoder = wgpu_context.get_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
        );

        {
            let mut scope = gpu_profiler.scope("Particle integration pass", &mut encoder);
            self.integration_pass.dispatch_by_items(
                &mut scope,
                (self.sim_params.num_particles, 1, 1),
                Some(vec![(0, bytemuck::bytes_of(&self.sim_params))]),
                &self.bind_resources.bind_group,
            );
        }
        gpu_profiler.resolve_queries(&mut encoder);

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
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
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: The particles' previous positions
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
                // Binding 2: The particles' radius
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        wgpu_context.get_device().create_bind_group_layout(&bind_group_layout_descriptor)
    }
    
    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_buffers: &ParticleBuffers) {
        self.sim_params.num_particles = particle_buffers.current_positions.len() as u32;
        self.bind_resources.bind_group = Self::create_bind_group(wgpu_context, &self.bind_resources.bind_group_layout, particle_buffers);
    }

    pub fn mouse_click_callback(&mut self, mouse_state: &ElementState, position: Vec2) {
        self.sim_params.is_mouse_pressed = mouse_state.is_pressed() as u32;
        self.sim_params.mouse_pos = position;
    }

    pub fn mouse_move_callback(&mut self, position: Vec2) {
        if self.sim_params.is_mouse_pressed == 1 {
            self.sim_params.mouse_pos = position;
        }
    }
    
}