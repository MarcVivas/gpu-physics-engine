use wgpu::{BindGroup, BindGroupLayout};

pub struct BindResources {
    pub bind_group: BindGroup,
    pub bind_group_layout: BindGroupLayout
}

impl BindResources {
    pub fn new(bind_group_layout: BindGroupLayout, bind_group: BindGroup) -> Self {
        Self {
            bind_group,
            bind_group_layout
        }
    }
}
