use std::mem;
use crate::renderer::wgpu_context::{WgpuContext};
use wgpu::{Buffer};
use wgpu::wgt::PollType::Wait;

#[derive(Debug)]
pub struct GpuBuffer<T> {
    data: Vec<T>,
    buffer: Buffer,
    usage: wgpu::BufferUsages,
}

impl<T: bytemuck::Pod> GpuBuffer<T>{
    pub fn new(wgpu_context: &WgpuContext, data: Vec<T>, usage: wgpu::BufferUsages) ->  Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = wgpu_context.get_device().create_buffer(&wgpu::BufferDescriptor  {
                    label: Some("GpuBuffer"),
                    size: (data.capacity() * size_of::<T>().max(1)) as u64,
                    usage,
                    mapped_at_creation: false,
                });
        wgpu_context.get_queue().write_buffer(
            &buffer,
            0,
            bytemuck::cast_slice(&data)
        );

        Self { data, buffer, usage}
    }
    
    pub fn push(&mut self, value: T, wgpu_context: &WgpuContext) {
        self.data.push(value);
        self.upload(wgpu_context, 1usize);
    }

    pub fn push_all(&mut self, values: &[T], wgpu_context: &WgpuContext) {
        self.data.extend_from_slice(values);
        self.upload(wgpu_context, values.len());
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }




    // Update the gpu buffer with the data in the vector
    fn upload(&mut self, wgpu_context: &WgpuContext, total_elems_added: usize) {
        let elem_size = size_of::<T>().max(1) as u64;
        let needed_bytes = (self.data.len() as u64) * elem_size;
        let current_capacity = self.buffer.size();

        if needed_bytes > current_capacity {
            // need a bigger buffer: double the capacity
            let new_capacity_bytes = needed_bytes.max(1) * 2;
            let old_data_len_bytes = ((self.data.len()-total_elems_added) as u64) * elem_size;

            let new_buffer = wgpu_context.get_device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuBuffer (resized)"),
                size: new_capacity_bytes,
                usage: self.usage,
                mapped_at_creation: false,
            });

            // Command a GPU-side copy from the old buffer to the new one.
            let mut encoder = wgpu_context.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuBuffer Resize Copy"),
            });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, old_data_len_bytes);
            wgpu_context.get_queue().submit(Some(encoder.finish()));

            // Replace the old buffer and update capacity.
            self.buffer = new_buffer;
        }

        // small upload: write the new tail
        let slice_start = self.data.len() - total_elems_added;
        let byte_offset = (slice_start * size_of::<T>().max(1)) as u64;
        let slice = &self.data[slice_start..];
        wgpu_context.get_queue().write_buffer(
            &self.buffer,
            byte_offset,
            bytemuck::cast_slice(slice),
        );

    }

    /// Downloads data from the GPU buffer to the CPU-side `Vec`.
    /// This method will overwrite the contents of `self.data`.
    ///
    /// # Returns
    ///
    /// `Ok(&Vec<T>)` if the readback was successful.
    /// `Err(wgpu::BufferAsyncError)` if the buffer mapping fails.
    pub fn download(&mut self, wgpu_context: &WgpuContext) -> Result<&Vec<T>, wgpu::BufferAsyncError> {
        let device = wgpu_context.get_device();
        let queue = wgpu_context.get_queue();

        // We want to read back the number of elements currently tracked by the CPU-side `Vec`.
        // This assumes the `Vec`'s length accurately reflects the amount of valid data on the GPU.
        let size = (self.data.len() * mem::size_of::<T>()) as u64;
        if size == 0 {
            // Nothing to download.
            self.data.clear();
            return Ok(&self.data);
        }

        // 1. Create a "staging" buffer. This is a special buffer that the CPU can read.
        // It needs the `MAP_READ` usage flag. `COPY_DST` is needed because we will
        // copy data *into* it from the main GPU buffer.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer (Download)"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. Create a command encoder to queue the copy command.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        // 3. Command the GPU to copy data from our main buffer to the staging buffer.
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            size,
        );

        // 4. Submit the command to the queue for the GPU to execute.
        queue.submit(Some(encoder.finish()));

        // 5. Map the staging buffer to read its contents from the CPU.
        // `map_async` is an asynchronous operation. We use a channel to wait for it
        // to complete, which makes our `download` function behave synchronously.
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // When the mapping is complete, the result is sent to our channel.
            sender.send(result).unwrap();
        });
        
        
        device.poll(Wait).unwrap();


        // 7. Wait for the mapping result to be received from the callback.
        // `recv()` will block until the result is available.
        match receiver.recv().unwrap() {
            Ok(()) => {
                // 8. The buffer is successfully mapped. Get a view of its contents.
                let mapped_range = buffer_slice.get_mapped_range();

                // 9. The data is a slice of raw bytes (`&[u8]`). We cast it back to `&[T]`.
                let downloaded_data: &[T] = bytemuck::cast_slice(&mapped_range);

                // 10. Update our internal `data` vector with the new data from the GPU.
                self.data.clear();
                self.data.extend_from_slice(downloaded_data);

                // 11. The `mapped_range` is a RAII guard. When it's dropped here,
                // the buffer is automatically unmapped.
                drop(mapped_range);

                Ok(&self.data)
            }
            Err(e) => {
                // The mapping failed. We propagate the error to the caller.
                Err(e)
            }
        }
    }

    /// Downloads just the last element from the GPU buffer.
    ///
    /// This is much more efficient than `download()` if you only need the last value,
    /// as it avoids copying the entire buffer's contents from the GPU to the CPU.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(T))` if the readback was successful and the buffer was not empty.
    /// - `Ok(None)` if the buffer is empty.
    /// - `Err(wgpu::BufferAsyncError)` if the buffer mapping fails.
    pub fn download_last(&self, wgpu_context: &WgpuContext) -> Result<Option<T>, wgpu::BufferAsyncError>
    {
        let device = wgpu_context.get_device();
        let queue = wgpu_context.get_queue();

        let element_size = mem::size_of::<T>() as u64;
        let num_elements = self.data.len();

        // If the buffer is empty, there is no last element to download.
        if num_elements == 0 || element_size == 0 {
            return Ok(None);
        }

        // 1. Calculate the offset of the last element in the source buffer.
        // The offset is (number of elements - 1) * size of each element.
        let source_offset = ((num_elements - 1) * mem::size_of::<T>()) as u64;

        // 2. Create a small staging buffer, just large enough for a single element.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer (Download Last)"),
            size: element_size, // Size of ONE element.
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Create a command encoder to queue the copy command.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Last Encoder"),
        });

        // 4. Command the GPU to copy only the last element.
        // We provide the source offset and a size of just one element.
        encoder.copy_buffer_to_buffer(
            &self.buffer,       // Source GPU buffer
            source_offset,      // Start reading from here
            &staging_buffer,    // Destination staging buffer
            0,                  // Write to the beginning of the staging buffer
            element_size,       // Copy this many bytes
        );

        // 5. Submit the command to the queue.
        queue.submit(Some(encoder.finish()));

        // 6. Map the staging buffer and wait for the result synchronously.
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Poll the device to ensure the GPU work is completed.
        device.poll(Wait).unwrap();

        // 7. Process the result of the mapping.
        match receiver.recv().unwrap() {
            Ok(()) => {
                // 8. Buffer is mapped. Get a view of its contents.
                let mapped_range = buffer_slice.get_mapped_range();

                // 9. The data is a slice of bytes. Cast it to a slice of `T`.
                // This slice will contain exactly one element.
                let data_slice: &[T] = bytemuck::cast_slice(&mapped_range);

                // 10. Copy the single element out. `T: Copy` makes this easy.
                let last_element = data_slice[0];

                // 11. Unmap the buffer by dropping the guard.
                drop(mapped_range);

                Ok(Some(last_element))
            }
            Err(e) => {
                // The mapping failed. Propagate the error.
                Err(e)
            }
        }
    }
    
    pub fn replace_elem(&mut self, new_data: T, index: usize, wgpu_context: &WgpuContext) {
        if index >= self.data.len() {
            panic!("Index out of bounds");
        }
        self.data[index] = new_data;
        wgpu_context.get_queue().write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&self.data),
        );
    }

    pub fn data(&self) -> &Vec<T>{
        &self.data
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

}
