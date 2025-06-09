use wgpu::Buffer;

pub trait Renderable {
    fn vertex_buffer(&self) -> &Buffer;
    fn num_vertices(&self) -> u32;

}