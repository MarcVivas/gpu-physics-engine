use std::collections::HashMap;
use wgpu::wgt::PollType::Wait;
use crate::renderer::wgpu_context::WgpuContext;

struct ScopeData {
    label: String,
    total_time_ms: f64,
    count: u64,
}

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    period: f32,
    capacity: u32,

    scopes: Vec<ScopeData>,
    scope_map: HashMap<String, usize>,

    query_count: u32,
    frame_scope_indices: Vec<usize>,

    last_frame_query_count: u32,
    last_frame_scope_indices: Vec<usize>,
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
            scopes: Vec::new(),
            scope_map: HashMap::new(),
            query_count: 0,
            frame_scope_indices: Vec::with_capacity(capacity as usize),
            last_frame_query_count: 0,
            last_frame_scope_indices: Vec::with_capacity(capacity as usize),
        }
    }

    pub fn begin_frame(&mut self) {
        self.query_count = 0;
        self.frame_scope_indices.clear();
    }

    pub fn scope<F>(&mut self, label: impl Into<String>, encoder: &mut wgpu::CommandEncoder, work: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        if self.query_count >= self.capacity {
            return;
        }

        let label_str = label.into();
        let scope_index = *self.scope_map.entry(label_str.clone()).or_insert_with(|| {
            let index = self.scopes.len();
            self.scopes.push(ScopeData {
                label: label_str,
                total_time_ms: 0.0,
                count: 0,
            });
            index
        });

        let start_query = self.query_count * 2;
        let end_query = start_query + 1;

        encoder.write_timestamp(&self.query_set, start_query);
        work(encoder);
        encoder.write_timestamp(&self.query_set, end_query);

        self.frame_scope_indices.push(scope_index);
        self.query_count += 1;
    }

    pub fn end_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.process_results(device);

        if self.query_count > 0 {
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

        self.last_frame_query_count = self.query_count;
        self.last_frame_scope_indices.clone_from(&self.frame_scope_indices);
    }

    fn process_results(&mut self, device: &wgpu::Device) {
        if self.last_frame_query_count == 0 {
            return;
        }

        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(Wait).unwrap();

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            for i in 0..self.last_frame_query_count as usize {
                let start_time = timestamps[i * 2];
                let end_time = timestamps[i * 2 + 1];

                if end_time > start_time {
                    let delta_ticks = end_time - start_time;
                    let delta_ns = delta_ticks as f64 * self.period as f64;
                    let delta_ms = delta_ns / 1_000_000.0;

                    let scope_index = self.last_frame_scope_indices[i];
                    self.scopes[scope_index].total_time_ms += delta_ms;
                    self.scopes[scope_index].count += 1;
                }
            }

            drop(data);
            self.staging_buffer.unmap();
        }
    }

    pub fn report(&mut self, wgpu_context: &WgpuContext) {
        {
            self.process_results(wgpu_context.get_device());
            self.last_frame_query_count = 0;

            if !self.scopes.is_empty() {
                println!("\n--- GPU Timings Report (Average) ---");
                for scope in &self.scopes {
                    if scope.count > 0 {
                        let avg_ms = scope.total_time_ms / scope.count as f64;
                        println!("{:<25}: {:.4} ms ({} samples)", scope.label, avg_ms, scope.count);
                    }
                }
            }
        }
    }
}