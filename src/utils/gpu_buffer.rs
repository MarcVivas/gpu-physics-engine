use crate::wgpu_context::{WgpuContext};
use wgpu::{util::DeviceExt, Buffer, Queue};

#[derive(Debug)]
pub struct GpuBuffer<T> {
    data: Vec<T>,
    buffer: wgpu::Buffer,
}

impl<T: bytemuck::Pod> GpuBuffer<T>{
    pub fn new(wgpu_context: &WgpuContext, data: Vec<T>, usage: wgpu::BufferUsages) ->  Self {
        let buffer = wgpu_context.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GpuBuffer"),
                    contents: bytemuck::cast_slice(&data),
                    usage,
                });

        Self { data, buffer }
    }

    // Update the gpu buffer with the data in the vector
    pub fn upload(&self, queue: &Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.data));
    }

    pub fn data(&self) -> &Vec<T>{
        &self.data
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

}
