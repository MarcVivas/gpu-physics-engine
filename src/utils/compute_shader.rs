// in renderer/compute_shader.rs

use wgpu::BindGroup;
use crate::renderer::wgpu_context::WgpuContext;

pub struct ComputeShader {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    workgroup_size: (u32, u32, u32),
}

impl ComputeShader {
    // The `new` method is unchanged.
    pub fn new(
        wgpu_context: &WgpuContext,
        shader_file: wgpu::ShaderModuleDescriptor,
        entry_point: &str,
        bind_group: BindGroup,
        bind_group_layout: wgpu::BindGroupLayout,
        workgroup_size: (u32, u32, u32),
    ) -> Self {
        let device = wgpu_context.get_device();
        let compute_shader = device.create_shader_module(shader_file);


        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Compute Pipeline Layout for {}", entry_point)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline for {}", entry_point)),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        

        Self {
            pipeline,
            bind_group_layout,
            workgroup_size,
            bind_group
        }
    }

    /// Dispatches the compute shader.
    ///
    /// - `wgpu_context`: Provides access to the `device` for creating resources.
    /// - `encoder`: The command encoder to add the pass to.
    /// - `label`: A debug label for the compute pass.
    /// - `bind_group_entries`: The actual resources (buffers, etc.) to bind for this dispatch.
    /// - `dispatch_size`: The number of workgroups to launch in (x, y, z).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dispatch_size: (u32, u32, u32),
    ) {

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_size.0, dispatch_size.1, dispatch_size.2);
    }


    /// A helper function to dispatch based on the total number of items to process.
    pub fn dispatch_by_items(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        item_count: (u32, u32, u32),
    ) {
        let dispatch_x = (item_count.0 + self.workgroup_size.0 - 1) / self.workgroup_size.0;
        let dispatch_y = (item_count.1 + self.workgroup_size.1 - 1) / self.workgroup_size.1;
        let dispatch_z = (item_count.2 + self.workgroup_size.2 - 1) / self.workgroup_size.2;

        // Pass the context through to the main dispatch method
        self.dispatch(
            encoder,
            (dispatch_x, dispatch_y, dispatch_z)
        );
    }

    /// A helper function to update the binding group with the given entries.
    pub fn update_binding_group(&mut self, wgpu_context: &WgpuContext, bind_group: wgpu::BindGroup) {
        self.bind_group = bind_group;
    }

    pub fn get_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
