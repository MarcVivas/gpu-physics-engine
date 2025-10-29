use glam::{Vec2, Vec4};
use crate::utils::gpu_buffer::GpuBuffer;

pub struct ParticleBuffers {
    pub current_positions: GpuBuffer<Vec2>,
    pub previous_positions: GpuBuffer<Vec2>,
    pub radii: GpuBuffer<f32>,
    pub colors: GpuBuffer<Vec4>,
    pub home_cell_ids: GpuBuffer<u32>, // Need this to sort objects by home cell
}