use std::cell::RefCell;
use std::rc::Rc;
use log::__private_api::enabled;
use wgpu::CommandEncoder;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::gpu_buffer::GpuBuffer;

const WORKGROUP_SIZE: (u32, u32, u32) = (256, 1, 1);
const ELEMS_PER_THREAD: u32 = 2;
const BLOCK_SIZE: u32 = ELEMS_PER_THREAD * WORKGROUP_SIZE.0;
const LIMIT: u32 = BLOCK_SIZE * BLOCK_SIZE;
pub struct PrefixSum {
    first_pass: ComputeShader,
    second_pass: ComputeShader,
    third_pass: ComputeShader,
    intermediate_buffer: GpuBuffer<u32>,
    uniform_data: GpuBuffer<u32>,
}

impl PrefixSum {
    pub fn new(wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) -> Self {
        if buffer.len() >= LIMIT as usize {
            panic!("Buffer too large for prefix sum");
        }
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
        
        let first_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "prefix_sum_of_each_block",
            binding_group.clone(),
            binding_group_layout.clone(),
            WORKGROUP_SIZE,
        );

        let second_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "prefix_sum_of_the_block_sums",
            binding_group.clone(),
            binding_group_layout.clone(),
            WORKGROUP_SIZE,
        );

        let third_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "add_block_prefix_sums_to_the_buffer",
            binding_group.clone(),
            binding_group_layout.clone(),
            WORKGROUP_SIZE,
        );
        
     
        
        Self {
            first_pass,  
            second_pass,
            third_pass,
            intermediate_buffer,
            uniform_data,
        }
    }
    
    /// Does the prefix sum
    pub fn execute(&self, encoder: &mut CommandEncoder, num_items: u32) {
        self.first_pass.dispatch_by_items(
            encoder,
            (num_items, 1, 1)
        );
        
        
        self.second_pass.dispatch_by_items(
            encoder,
            ((num_items as f32/BLOCK_SIZE as f32).ceil() as u32, 1, 1)
        );
        
        self.third_pass.dispatch_by_items(
            encoder,
            (num_items, 1, 1)
        );
        
    }
    
    pub fn print_buffer(&mut self, wgpu_context: &WgpuContext){
        println!("{:?}", self.intermediate_buffer.download(wgpu_context));
        
    }

    /// Update buffers when resizing the buffer
    pub fn update_buffers(&mut self, wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) {
        let binding_group_layout = self.first_pass.get_bind_group_layout();
        
        let new_len: u32 = buffer.len() as u32;
        self.uniform_data.replace_elem(new_len, 0, wgpu_context);
        
        self.intermediate_buffer.push_all(&vec![0u32; buffer.len()-self.intermediate_buffer.len()], wgpu_context);
        
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
        
        self.first_pass.update_binding_group(
            wgpu_context,
            binding_group.clone()
        );
        
        self.second_pass.update_binding_group(
            wgpu_context,
            binding_group.clone()
        );
        
        self.third_pass.update_binding_group(
            wgpu_context,
            binding_group
        );
    }
}