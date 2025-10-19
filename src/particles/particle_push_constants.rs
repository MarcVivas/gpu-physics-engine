use glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SimParams {
    pub delta_time: f32,
    pub world_width: f32,
    pub world_height: f32,
    pub is_mouse_pressed: u32,
    pub mouse_pos: Vec2,
    pub num_particles: u32, 
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstantsCreateCells {
    pub cell_size: f32,
    pub num_particles: u32,
}


