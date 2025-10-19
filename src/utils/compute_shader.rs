// in renderer/compute_shader.rs

use wgpu::{BindGroup, CommandEncoder, PushConstantRange};
use crate::renderer::wgpu_context::WgpuContext;

pub struct ComputeShader {
    pipeline: wgpu::ComputePipeline,
    workgroup_size: (u32, u32, u32),
}

impl ComputeShader {
   pub fn new(
        wgpu_context: &WgpuContext,
        shader_file: wgpu::ShaderModuleDescriptor,
        entry_point: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
        workgroup_size: (u32, u32, u32),
        constants: &Vec<(&str, f64)>,
        push_constants: &Vec<PushConstantRange>,
    ) -> Self {
        let device = wgpu_context.get_device();
        let compute_shader = device.create_shader_module(shader_file);


        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Compute Pipeline Layout for {}", entry_point)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: push_constants.as_slice(),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline for {}", entry_point)),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions{
                constants: constants.as_slice(),
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
       
        Self {
            pipeline,
            workgroup_size,
        }
    }

    /// Dispatches the compute shader.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        dispatch_size: (u32, u32, u32),
        push_constants_data: Option<Vec<(u32, &[u8])>>,
        bind_group: &BindGroup,
    ) {

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });


        compute_pass.set_pipeline(&self.pipeline);

        if let Some(constants) = push_constants_data {
            for (offset, data) in constants {
                compute_pass.set_push_constants(
                    offset,
                    data,
                );
            }
        }

        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_size.0, dispatch_size.1, dispatch_size.2);
    }


    /// A helper function to dispatch based on the total number of items to process.
    pub fn dispatch_by_items(
        &self,
        encoder: &mut CommandEncoder,
        item_count: (u32, u32, u32),
        push_constants_data: Option<Vec<(u32, &[u8])>>,
        bind_group: &BindGroup,
    ) {
        let dispatch_x = (item_count.0 + self.workgroup_size.0 - 1) / self.workgroup_size.0;
        let dispatch_y = (item_count.1 + self.workgroup_size.1 - 1) / self.workgroup_size.1;
        let dispatch_z = (item_count.2 + self.workgroup_size.2 - 1) / self.workgroup_size.2;

        // Pass the context through to the main dispatch method
        self.dispatch(
            encoder,
            (dispatch_x, dispatch_y, dispatch_z),
            push_constants_data,
            bind_group
        );
    }
    
    pub fn indirect_dispatch(
        &self,
        encoder: &mut CommandEncoder,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        push_constants_data: Option<Vec<(u32, &[u8])>>,
        bind_group: &BindGroup,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Solve Pass"), timestamp_writes: None });


        compute_pass.set_pipeline(&self.pipeline);


        if let Some(constants) = push_constants_data {
            for (offset, data) in constants {
                compute_pass.set_push_constants(
                    offset,
                    data,
                );
            }
        }
        
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
    }
}
