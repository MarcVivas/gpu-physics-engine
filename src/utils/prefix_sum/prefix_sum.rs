use wgpu::{BindGroup, CommandEncoder, PushConstantRange};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::get_subgroup_size;
use crate::utils::gpu_buffer::GpuBuffer;

const WORKGROUP_SIZE: (u32, u32, u32) = (256, 1, 1);
const LIMIT: u32 = WORKGROUP_SIZE.0 * WORKGROUP_SIZE.0;
pub struct PrefixSum {
    first_pass: ComputeShader,
    second_pass: ComputeShader,
    third_pass: ComputeShader,
    intermediate_buffer: GpuBuffer<u32>,
    block_prefix_sum: Option<Box<PrefixSum>>,
    bind_resources: BindResources,
}

impl PrefixSum {
    pub fn new(wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) -> Self {
        

        let intermediate_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0u32; PrefixSum::get_max_possible_block_sums(buffer)],
            wgpu::BufferUsages::STORAGE,
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
                // Intermediate data
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                        resource: intermediate_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        );
        
        let bind_resources = BindResources::new(binding_group_layout, binding_group);

        let max_subgroup_size = get_subgroup_size(wgpu_context).unwrap();

        let constants = vec![
            ("SUBGROUP_SIZE", max_subgroup_size as f64),
            ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
            ("SHARED_MEMORY_SIZE", ((WORKGROUP_SIZE.0/max_subgroup_size)*2) as f64),
        ];

        let push_constants = vec![
            PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }
        ];

        let first_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "prefix_sum_of_each_block",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &constants,
            &push_constants
        );
        

        let second_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "prefix_sum_of_the_block_sums",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &constants,
            &vec![]
        );

        let third_pass = ComputeShader::new(
            wgpu_context,
            wgpu::include_wgsl!("prefix_sum.wgsl"),
            "add_block_prefix_sums_to_the_buffer",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &constants,
            &push_constants
        );

        
        let mut block_prefix_sum = None;
        if buffer.len() >= LIMIT as usize {
            block_prefix_sum = Some(Box::new(PrefixSum::new(wgpu_context, &intermediate_buffer)));
        }
        
        Self {
            first_pass,  
            second_pass,
            third_pass,
            intermediate_buffer,
            block_prefix_sum,
            bind_resources
        }
    }
    
    /// Performs the prefix sum algorithm
    pub fn execute(&self, wgpu_context: &WgpuContext, encoder: &mut CommandEncoder, num_items: u32) {
        let num_blocks = (num_items as f32 / WORKGROUP_SIZE.0 as f32).ceil() as u32;

        // Pass 1: Dispatch one workgroup per data block.
        self.first_pass.dispatch_by_items(encoder, (num_items, 1, 1), Some((0, &num_items)), &self.bind_resources.bind_group);

        if num_items >= LIMIT {
            self.block_prefix_sum.as_ref().unwrap().execute(wgpu_context, encoder, num_blocks);
        }
        else {
            // Pass 2: Dispatch a single workgroup to scan the block_sums.
            self.second_pass.dispatch::<u32>(encoder, (1, 1, 1), None, &self.bind_resources.bind_group);
        }

        // Pass 3: Dispatch one thread for each number of items to add the block_sums to the buffer.
        self.third_pass.dispatch_by_items(encoder, (num_items, 1, 1), Some((0, &num_items)), &self.bind_resources.bind_group);
        
    }
    
    fn get_max_possible_block_sums(buffer: &GpuBuffer<u32>) -> usize{
        (buffer.len() as f32 / WORKGROUP_SIZE.0 as f32).ceil() as usize
    }
    
    pub fn print_buffer(&mut self, wgpu_context: &WgpuContext){
        println!("{:?}", self.intermediate_buffer.download(wgpu_context));
        
    }

    /// Update buffers when resizing the buffer
    pub fn update_buffers(&mut self, wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) {
        let binding_group_layout = &self.bind_resources.bind_group_layout;
        
        let new_len: u32 = buffer.len() as u32;

        let num_blocks_to_add = PrefixSum::get_max_possible_block_sums(buffer)-self.intermediate_buffer.len();
        self.intermediate_buffer.push_all(&vec![0u32; num_blocks_to_add], wgpu_context);

        if new_len >= LIMIT && self.block_prefix_sum.is_none(){
            self.block_prefix_sum = Some(Box::new(PrefixSum::new(wgpu_context, &self.intermediate_buffer)));
        }
        else if new_len >= LIMIT && self.block_prefix_sum.is_some(){
            self.block_prefix_sum.as_mut().unwrap().update_buffers(wgpu_context, &self.intermediate_buffer);
        }
        
        self.bind_resources.bind_group = wgpu_context.get_device().create_bind_group(
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
                        resource: self.intermediate_buffer.buffer().as_entire_binding(),
                    },
                ],
            }
        );
    }
}