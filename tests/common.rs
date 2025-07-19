// Not every test file will use every function.
#![allow(dead_code)]

use game_engine::renderer::camera::Camera;
use glam::{Vec2};
use game_engine::game_data::particle::particle_system::ParticleSystem;
use game_engine::renderer::wgpu_context::WgpuContext;
use game_engine::utils::gpu_buffer::GpuBuffer;

// A struct to hold all the common objects for a test.
pub struct TestSetup {
    pub wgpu_context: WgpuContext,
    pub camera: Camera,
}

// The main setup function.
pub async fn setup() -> TestSetup {
    let wgpu_context = WgpuContext::new_for_test().await.unwrap();


    let camera = Camera::new(&Vec2{x: 1920.0, y: 1080.0}, &wgpu_context);

    TestSetup {
        wgpu_context,
        camera,
    }
}

pub fn create_test_particle_system(wgpu_context: &WgpuContext, positions: Vec<Vec2>, radius: Vec<f32>) -> ParticleSystem{
    let p_buffer = GpuBuffer::new(wgpu_context, positions, wgpu::BufferUsages::STORAGE);
    let r_buffer = GpuBuffer::new(wgpu_context, radius, wgpu::BufferUsages::STORAGE);
    ParticleSystem::new_from_buffers(wgpu_context, p_buffer, r_buffer)
}