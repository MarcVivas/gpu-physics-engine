use std::num::NonZeroU32;
use crate::renderer::wgpu_context::WgpuContext;

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    period: f32,
    capacity: u32,
    query_count: u32,
    labels: Vec<String>,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GpuTimer QuerySet"),
            count: capacity * 2,
            ty: wgpu::QueryType::Timestamp,
        });

        let buffer_size = (capacity as u64 * 2) * std::mem::size_of::<u64>() as u64;

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTimer Resolve Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTimer Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            period: queue.get_timestamp_period(),
            capacity,
            query_count: 0,
            labels: Vec::with_capacity(capacity as usize),
        }
    }

    pub fn begin_frame(&mut self) {
        self.query_count = 0;
        self.labels.clear();
    }

    pub fn scope<F>(&mut self, label: impl Into<String>, encoder: &mut wgpu::CommandEncoder, work: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        assert!(self.query_count < self.capacity, "GpuTimer capacity exceeded");
        let start_index = self.query_count * 2;
        let end_index = start_index + 1;

        encoder.write_timestamp(&self.query_set, start_index);
        work(encoder);
        encoder.write_timestamp(&self.query_set, end_index);

        self.labels.push(label.into());
        self.query_count += 1;
    }

    pub fn end_frame(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.query_count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuTimer EndFrame Encoder"),
        });

        encoder.resolve_query_set(
            &self.query_set,
            0..self.query_count * 2,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            self.resolve_buffer.size(),
        );

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn read_results(&self, device: &wgpu::Device) -> Option<Vec<(String, f64)>> {
        if self.query_count == 0 {
            return Some(Vec::new());
        }

        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::MaintainBase::Wait).unwrap();

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            let mut results = Vec::with_capacity(self.query_count as usize);
            for i in 0..self.query_count as usize {
                let start_time = timestamps[i * 2];
                let end_time = timestamps[i * 2 + 1];
                if end_time > start_time {
                    let delta_ticks = end_time - start_time;
                    let delta_ns = delta_ticks as f64 * self.period as f64;
                    let delta_ms = delta_ns / 1_000_000.0;
                    results.push((self.labels[i].clone(), delta_ms));
                }
            }

            drop(data);
            self.staging_buffer.unmap();

            Some(results)
        } else {
            None
        }
    }
    
    
    pub fn print_results(&self, wgpu_context: &WgpuContext) {
        #[cfg(debug_assertions)]
        if let Some(results) = self.read_results(wgpu_context.get_device()) {
            if !results.is_empty() {
                println!("--- GPU Frame Timings ---");
                for (label, time_ms) in results {
                    println!("{:<25}: {:.4} ms", label, time_ms);
                }
            }
        }
    }
}