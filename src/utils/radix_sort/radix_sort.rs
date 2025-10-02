/*
    This file implements a gpu version of radix sort. 

    Currently, only the sorting for 32-bit key-value pairs is implemented

    All shaders can be found in radix_sort.wgsl
*/

use std::{
    mem,
    num::{NonZeroU32, NonZeroU64},
};

use bytemuck::bytes_of;
use wgpu::{include_wgsl, util::DeviceExt, BufferAsyncError, PushConstantRange};
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::bind_resources::BindResources;
use crate::utils::compute_shader::ComputeShader;
use crate::utils::get_subgroup_size;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::utils::radix_sort::radix_sort;

pub const WORKGROUP_SIZE: (u32, u32, u32) = (256, 1, 1);



// Number of bits processed in one pass
pub const RADIX_SORT_BITS_PER_PASS: u32 = 8;

// 2^(bits processed in one pass)
// In this case 2^8 = 256. Thus, 8 bits are processed in one pass
// This number should be <= WORKGROUP_SIZE
pub const RADIX_SORT_BUCKETS: u32 = 1 << RADIX_SORT_BITS_PER_PASS;

// Number of bits per element
// u32 -> 32 bits
// u64 -> 64 bits
pub const BITS_PER_ELEMENT: u32 = 32;
pub const RADIX_SORT_TOTAL_ITERATIONS: u32 = BITS_PER_ELEMENT / RADIX_SORT_BITS_PER_PASS;

// Each workgroup processes NUM_BLOCKS_PER_WORKGROUP blocks/histograms
pub const NUM_BLOCKS_PER_WORKGROUP: u32 = 45;



pub struct GPUSorter {
    histogram_shader: ComputeShader,
    scatter_shader: ComputeShader,
    sorting_buffers: SortBuffers,
    bind_resources: BindResources
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub num_elements: u32,
    pub current_shift: u32, 
    pub num_workgroups: u32,
    pub num_blocks_per_workgroup: u32,
}

impl GPUSorter {
    pub fn new(wgpu_context: &WgpuContext, length: NonZeroU32, keys: &GpuBuffer<u32>, payload: &GpuBuffer<u32>) -> Self {
        
        let bind_group_layout = Self::create_bind_group_layout(wgpu_context.get_device());

        let sorting_buffers = Self::create_sort_buffers(wgpu_context, length, keys, payload);
        
        let bind_group = sorting_buffers.bind_group_ping.clone();
        
        let bind_resources = BindResources::new(bind_group_layout, bind_group);
        
        
        assert!(WORKGROUP_SIZE.0 <= RADIX_SORT_BUCKETS);
        
        let constants = vec![
            ("WORKGROUP_SIZE", WORKGROUP_SIZE.0 as f64),
            ("RADIX_SORT_BUCKETS", RADIX_SORT_BUCKETS as f64),
            ("SUBGROUP_SIZE", get_subgroup_size(wgpu_context).unwrap() as f64),
        ];

        
        let push_constants = vec![
            PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..size_of::<PushConstants>() as u32,
            }
        ];
        
        let histogram_shader = ComputeShader::new(
            wgpu_context,
            include_wgsl!("radix_sort.wgsl"),
            "build_histogram",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &constants,
            &push_constants,
        );


        let scatter_shader = ComputeShader::new(
            wgpu_context,
            include_wgsl!("radix_sort.wgsl"),
            "scatter_keys",
            &bind_resources.bind_group_layout,
            WORKGROUP_SIZE,
            &constants,
            &push_constants
        );


        Self {
            histogram_shader,
            scatter_shader,
            sorting_buffers,
            bind_resources,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix sort bind group layout"),
            entries: &[
                // Keys buffer
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
                // Histogram buffer
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
                // Payload a
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
                // Keys b 
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Payload b
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }
    
    pub fn build_histogram(&mut self, encoder: &mut wgpu::CommandEncoder, total_threads: (u32, u32, u32), push_constants: &PushConstants, ping_pong: &bool){
        let ping_pong_bind_group = if *ping_pong {&self.sorting_buffers.bind_group_ping} else {&self.sorting_buffers.bind_group_pong};
        self.histogram_shader.dispatch_by_items(
            encoder,
            total_threads,
            Some((0, push_constants)),
            ping_pong_bind_group
        );
    }

    pub fn scatter(&mut self, encoder: &mut wgpu::CommandEncoder, total_threads: (u32, u32, u32), push_constants: &PushConstants, ping_pong: &bool){
        let ping_pong_bind_group = if *ping_pong {&self.sorting_buffers.bind_group_ping} else {&self.sorting_buffers.bind_group_pong};
        self.scatter_shader.dispatch_by_items(
            encoder,
            total_threads,
            Some((0, push_constants)),
            ping_pong_bind_group       
        );
    }
    pub fn sort(&mut self, encoder: &mut wgpu::CommandEncoder, wgpu_context: &WgpuContext, sort_first_n:Option<u32>) {
        let sort_buffers = &self.sorting_buffers;
        
        let num_elements = sort_first_n.unwrap_or(sort_buffers.len());
        let total_threads = ((num_elements + NUM_BLOCKS_PER_WORKGROUP - 1) / NUM_BLOCKS_PER_WORKGROUP, 1, 1);
        let num_workgroups = (total_threads.0 + WORKGROUP_SIZE.0 - 1) / WORKGROUP_SIZE.0; 
        let mut ping_pong: bool = true;
        for i in 0..RADIX_SORT_TOTAL_ITERATIONS{
            let push_constants = PushConstants{
                num_elements,
                current_shift: i * RADIX_SORT_BITS_PER_PASS,
                num_workgroups,
                num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
            };
            self.build_histogram(encoder, total_threads, &push_constants, &ping_pong);
            self.scatter(encoder, total_threads, &push_constants, &ping_pong);
            ping_pong = !ping_pong;
        }
    }

    pub fn get_keys_b(&mut self, wgpu_context: &WgpuContext) -> Result<&Vec<u32>, BufferAsyncError> {
        self.sorting_buffers.keys_b.download(wgpu_context)
    }

    pub fn get_histogram(&mut self, wgpu_context: &WgpuContext) -> Result<&Vec<u32>, BufferAsyncError> {
        self.sorting_buffers.histogram.download(wgpu_context)
    }


    pub fn sort_indirect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sort_buffers: &SortBuffers,
        dispatch_buffer: &wgpu::Buffer,
    ) {
        let bind_group = &sort_buffers.bind_group_ping;

    }

    pub fn update_sorting_buffers(&mut self, wgpu_context: &WgpuContext,
                                  length: NonZeroU32,
                                  keys_a: &GpuBuffer<u32>,
                                  payload_a: &GpuBuffer<u32>){
        self.sorting_buffers = Self::create_sort_buffers(wgpu_context, length, keys_a, payload_a);
    }
    
    /// Creates all buffers necessary for sorting, using user-provided buffers for keys and values.
    ///
    /// # Arguments
    ///
    /// * `wgpu_context` - The wgpu context for creating new buffers.
    /// * `length` - The number of key-value pairs to be sorted.
    /// * `keys_a` - Your buffer containing the keys to be sorted.
    /// * `payload_a` - Your buffer containing the corresponding values (payload).
    fn create_sort_buffers(
        wgpu_context: &WgpuContext,
        length: NonZeroU32,
        keys_a: &GpuBuffer<u32>,
        payload_a: &GpuBuffer<u32>,
    ) -> SortBuffers {
        let length = length.get();
        
        let payload_b = GpuBuffer::new(
            wgpu_context,
            vec![0; length as usize],
            wgpu::BufferUsages::STORAGE
        );

        let keys_b = GpuBuffer::new(
            wgpu_context,
            vec![0; length as usize],
            wgpu::BufferUsages::STORAGE
        );

        let histogram = GpuBuffer::new(
            wgpu_context,
            vec![0; get_histogram_size(length) as usize],   
            wgpu::BufferUsages::STORAGE
        );
        
        let device = wgpu_context.get_device();

        let bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix sort bind group with user buffers"),
            layout: &Self::create_bind_group_layout(device), 
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: keys_a.buffer().as_entire_binding(),
                },
                // Histogram buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram.buffer().as_entire_binding(),
                },
                // Payload a
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: payload_a.buffer().as_entire_binding(),
                },
                // Keys b
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keys_b.buffer().as_entire_binding(),
                },
                // Payload b
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_b.buffer().as_entire_binding(),
                },
            ],
        });

        let bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix sort bind group pong with user buffers"),
            layout: &Self::create_bind_group_layout(device),
            entries: &[
                // Keys b
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: keys_b.buffer().as_entire_binding(),
                },
                // Histogram buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram.buffer().as_entire_binding(),
                },
                // Payload b
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: payload_b.buffer().as_entire_binding(),
                },
                // Keys a
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keys_a.buffer().as_entire_binding(),
                },
                // Payload a
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_a.buffer().as_entire_binding(),
                },
            ],
        });

        SortBuffers {
            histogram,
            keys_b,
            payload_b,
            bind_group_ping,
            bind_group_pong,
            length,
        }
    }
    
   
}

fn get_histogram_size(length: u32) -> u32 {
    let total_threads = ((length + NUM_BLOCKS_PER_WORKGROUP - 1) / NUM_BLOCKS_PER_WORKGROUP, 1, 1);
    let num_workgroups = (total_threads.0 + WORKGROUP_SIZE.0 - 1) / WORKGROUP_SIZE.0;
    RADIX_SORT_BUCKETS * num_workgroups
}

/// Struct containing all buffers necessary for sorting.
/// The key and value buffers can be read and written.
pub struct SortBuffers {
    #[allow(dead_code)]
    histogram: GpuBuffer<u32>,
    
    /// intermediate key buffer for sorting
    #[allow(dead_code)]
    keys_b: GpuBuffer<u32>,

    /// intermediate value buffer for sorting
    #[allow(dead_code)]
    payload_b: GpuBuffer<u32>,


    /// bind group used for sorting
    bind_group_ping: wgpu::BindGroup,
    bind_group_pong: wgpu::BindGroup,

    // number of key-value pairs
    length: u32,
}

impl SortBuffers {
    /// number of key-value pairs that can be stored in this buffer
    pub fn len(&self) -> u32 {
        self.length
    }
    

    
}

