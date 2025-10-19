use crate::grid::grid::UNUSED_CELL_ID;
use crate::physics::collision_cell_builder::COUNTING_CHUNK_SIZE;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct CollisionCellBuffers{
    chunk_counting_buffer: GpuBuffer<u32>, // Stores the number of objects in each chunk.
    collision_cells: GpuBuffer<u32>, // Stores the cells that contain more than one object.
    indirect_dispatch_buffer: GpuBuffer<u32>,
}


impl CollisionCellBuffers{
    pub fn new(wgpu_context: &WgpuContext, buffer_len: usize) -> Self {
        let chunk_counting_buffer_len: usize = ((buffer_len as u32 + COUNTING_CHUNK_SIZE -1) / COUNTING_CHUNK_SIZE) as usize;

        let chunk_counting_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0; chunk_counting_buffer_len],
            wgpu::BufferUsages::STORAGE,
        );

        let collision_cells = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; buffer_len],
            wgpu::BufferUsages::STORAGE,
        );

        let indirect_dispatch_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0; 3],
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );
        Self{
            chunk_counting_buffer,
            collision_cells,
            indirect_dispatch_buffer,
        }
    }
    
    pub fn get_chunk_counting(&self) -> &GpuBuffer<u32>{
        &self.chunk_counting_buffer
    }
    
    pub fn get_collision_cells(&self) -> &GpuBuffer<u32>{
        &self.collision_cells
    }
    
    pub fn get_indirect_dispatch(&self) -> &GpuBuffer<u32>{
        &self.indirect_dispatch_buffer
    }
    
    pub fn download_collision_cells(&mut self, wgpu_context: &WgpuContext) -> Vec<u32>{
        self.collision_cells.download(wgpu_context).unwrap().clone()
    }
    
    pub fn push_all_to_chunk_counting(&mut self, wgpu_context: &WgpuContext, values: &[u32]) {
        self.chunk_counting_buffer.push_all(values, wgpu_context);
    }
    
    pub fn push_all_to_collision_cells(&mut self, wgpu_context: &WgpuContext, values: &[u32]) {
        self.collision_cells.push_all(values, wgpu_context);
    }
}