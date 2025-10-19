use glam::{Vec2, Vec4};
use crate::utils::gpu_buffer::GpuBuffer;

pub struct ParticleBuffers {
    pub current_positions_ping: GpuBuffer<Vec2>,
    pub current_positions_pong: GpuBuffer<Vec2>,
    pub previous_positions: GpuBuffer<Vec2>,
    pub radii: GpuBuffer<f32>,
    pub colors: GpuBuffer<Vec4>,
    pub cell_ids: GpuBuffer<u32>,
    pub particle_ids: GpuBuffer<u32>
}