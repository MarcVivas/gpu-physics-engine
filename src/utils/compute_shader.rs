// in renderer/compute_shader.rs

use crate::renderer::wgpu_context::WgpuContext;

pub struct ComputeShader {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    workgroup_size: (u32, u32, u32),
}

impl ComputeShader {
    // The `new` method is unchanged.
    pub fn new(
        wgpu_context: &WgpuContext,
        shader_module: &wgpu::ShaderModule,
        entry_point: &str,
        bind_group_layout_descriptor: &wgpu::BindGroupLayoutDescriptor,
        workgroup_size: (u32, u32, u32),
    ) -> Self {
        let device = wgpu_context.get_device();

        let bind_group_layout = device.create_bind_group_layout(bind_group_layout_descriptor);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Compute Pipeline Layout for {}", entry_point)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Compute Pipeline for {}", entry_point)),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            workgroup_size,
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
        wgpu_context: &WgpuContext, // <-- ADDED THIS PARAMETER
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        bind_group_entries: &[wgpu::BindGroupEntry],
        dispatch_size: (u32, u32, u32),
    ) {
        // Create a bind group on-the-fly using the device from the context.
        let bind_group = wgpu_context.get_device().create_bind_group(&wgpu::BindGroupDescriptor { // <-- CORRECTED
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: bind_group_entries,
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_size.0, dispatch_size.1, dispatch_size.2);
    }

    /// A helper function to dispatch based on the total number of items to process.
    pub fn dispatch_by_items(
        &self,
        wgpu_context: &WgpuContext, // <-- ADDED THIS PARAMETER
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        bind_group_entries: &[wgpu::BindGroupEntry],
        item_count: (u32, u32, u32),
    ) {
        let dispatch_x = (item_count.0 + self.workgroup_size.0 - 1) / self.workgroup_size.0;
        let dispatch_y = (item_count.1 + self.workgroup_size.1 - 1) / self.workgroup_size.1;
        let dispatch_z = (item_count.2 + self.workgroup_size.2 - 1) / self.workgroup_size.2;

        // Pass the context through to the main dispatch method
        self.dispatch(
            wgpu_context,
            encoder,
            label,
            bind_group_entries,
            (dispatch_x, dispatch_y, dispatch_z)
        );
    }
}