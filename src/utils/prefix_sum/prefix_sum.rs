use std::cell::RefCell;
use std::rc::Rc;
use log::__private_api::enabled;
use wgpu::CommandEncoder;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;

const WORKGROUP_SIZE: (u32, u32, u32) = (64, 1, 1);


pub struct PrefixSum {
    shader: ComputeShader,
    intermediate_buffer: GpuBuffer<u32>,
    uniform_data: GpuBuffer<u32>,
}

impl PrefixSum {
    pub fn new(wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) -> Self {
        let intermediate_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0u32; buffer.len()],
            wgpu::BufferUsages::STORAGE,
        );

        let uniform_data = GpuBuffer::new(
            wgpu_context,
            vec![buffer.len() as u32],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );


        let binding_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid compute Bind Group Layout"),
            entries: &[
                // Buffer data
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
                // Uniform data
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Intermediate data
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        };
        
        let binding_group_layout = wgpu_context.get_device().create_bind_group_layout(&binding_group_layout_desc);
        
        let binding_group = wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &binding_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform_data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: intermediate_buffer.buffer().as_entire_binding(),
                    },
                    
                ],
            }
        );
        
        let shader = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "prefix_sum_1",
            binding_group,
            binding_group_layout,
            WORKGROUP_SIZE,
        );
        
     
        
        Self {
            shader,    
            intermediate_buffer,
            uniform_data,
        }
    }
    
    /// Does the prefix sum
    pub fn execute(&self, encoder: &mut CommandEncoder, num_items: u32) {
        self.shader.dispatch_by_items(
            encoder,
            (num_items, 1, 1)
        );
        
    }

    /// Update buffers when resizing the buffer
    pub fn update_buffers(&mut self, wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) {
        let binding_group_layout = self.shader.get_bind_group_layout();
        
        let new_len: u32 = buffer.len() as u32;
        self.uniform_data.replace_elem(new_len, 0, wgpu_context);
        
        let binding_group = wgpu_context.get_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: None,
                layout: &binding_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.intermediate_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        );
        
        self.shader.update_binding_group(
            wgpu_context,
            binding_group
        );
    }
}