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
    block_prefix_sum: Option<Box<PrefixSum>>,
}

impl PrefixSum {
    pub fn new(wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) -> Self {
        

        let intermediate_buffer = GpuBuffer::new(
            wgpu_context,
            vec![0u32; PrefixSum::get_max_possible_block_sums(buffer)],
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

        
        let mut block_prefix_sum = None;
        if buffer.len() >= LIMIT as usize {
            block_prefix_sum = Some(Box::new(PrefixSum::new(wgpu_context, &intermediate_buffer)));
        }
        
        Self {
            first_pass,  
            second_pass,
            third_pass,
            intermediate_buffer,
            uniform_data,
            block_prefix_sum,
        }
    }
    
    /// Performs the prefix sum algorithm
    pub fn execute(&self, wgpu_context: &WgpuContext, encoder: &mut CommandEncoder, num_items: u32) {
        let num_blocks = (num_items as f32 / BLOCK_SIZE as f32).ceil() as u32;

        // Pass 1: Dispatch one workgroup per data block.
        self.first_pass.dispatch(encoder, (num_blocks, 1, 1));

        if num_items >= LIMIT {
            self.block_prefix_sum.as_ref().unwrap().execute(wgpu_context, encoder, num_blocks);
        }
        else {
            // Pass 2: Dispatch a single workgroup to scan the block_sums.
            self.second_pass.dispatch(encoder, (1, 1, 1));
        }
        


        // Pass 3: Dispatch one thread for each number of items to add the block_sums to the buffer.
        self.third_pass.dispatch_by_items(encoder, (num_items, 1, 1));
        
    }
    
    fn get_max_possible_block_sums(buffer: &GpuBuffer<u32>) -> usize{
        (buffer.len() as f32 / BLOCK_SIZE as f32).ceil() as usize
    }
    
    pub fn print_buffer(&mut self, wgpu_context: &WgpuContext){
        println!("{:?}", self.intermediate_buffer.download(wgpu_context));
        
    }

    /// Update buffers when resizing the buffer
    pub fn update_buffers(&mut self, wgpu_context: &WgpuContext, buffer: &GpuBuffer<u32>) {
        let binding_group_layout = self.first_pass.get_bind_group_layout();
        
        let new_len: u32 = buffer.len() as u32;
        
        self.uniform_data.replace_elem(new_len, 0, wgpu_context);

        let num_blocks_to_add = PrefixSum::get_max_possible_block_sums(buffer)-self.intermediate_buffer.len();
        self.intermediate_buffer.push_all(&vec![0u32; num_blocks_to_add], wgpu_context);

        if new_len >= LIMIT && self.block_prefix_sum.is_none(){
            self.block_prefix_sum = Some(Box::new(PrefixSum::new(wgpu_context, &self.intermediate_buffer)));
        }
        else if new_len >= LIMIT && self.block_prefix_sum.is_some(){
            self.block_prefix_sum.as_mut().unwrap().update_buffers(wgpu_context, &self.intermediate_buffer);
        }
        
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
            binding_group.clone()
        );
        
        self.second_pass.update_binding_group(
            binding_group.clone()
        );
        
        self.third_pass.update_binding_group(
            binding_group
        );
    }
}