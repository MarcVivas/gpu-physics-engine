use std::{
    num::NonZeroU32,
    ops::{RangeBounds},
};

use wgpu::util::DeviceExt;
use std::sync::mpsc;
use super::radix_sort::GPUSorter;

#[doc(hidden)]
/// only used for testing
/// temporally used for guessing subgroup size
pub fn upload_to_buffer<T: bytemuck::Pod>(
    encoder: &mut wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    values: &[T],
) {
    let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, buffer, 0, staging_buffer.size());
}

#[doc(hidden)]
/// only used for testing
/// temporally used for guessing subgroup size
pub async fn download_buffer<T: Clone + bytemuck::Pod>(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    range: impl RangeBounds<wgpu::BufferAddress>,
) -> Vec<T> {
    // 1. Resolve the byte range requested by the caller.
    let start_bound = match range.start_bound() {
        std::ops::Bound::Included(&n) => n,
        std::ops::Bound::Excluded(&n) => n + 1,
        std::ops::Bound::Unbounded => 0,
    };
    // The end bound for wgpu copies is exclusive.
    let end_bound = match range.end_bound() {
        std::ops::Bound::Included(&n) => n + 1,
        std::ops::Bound::Excluded(&n) => n,
        std::ops::Bound::Unbounded => buffer.size(),
    };
    let size = end_bound - start_bound;

    // A quick check to ensure the requested byte range is valid for the type T.
    assert_eq!(
        size % std::mem::size_of::<T>() as u64,
        0,
        "Download range size must be a multiple of the size of T"
    );

    if size == 0 {
        return Vec::new();
    }

    // 2. Create a "staging" buffer just large enough for the requested range.
    // This buffer is readable by the CPU.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 3. Create a command encoder to queue the copy command.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Download Copy Encoder"),
    });

    // 4. Command the GPU to copy data from the source buffer (at the specified offset)
    // into the beginning of our staging buffer.
    encoder.copy_buffer_to_buffer(
        buffer,          // source
        start_bound,     // source offset
        &staging_buffer, // destination
        0,               // destination offset
        size,            // size
    );

    // 5. Submit the command to the GPU.
    queue.submit(Some(encoder.finish()));

    // 6. Request to map the staging buffer for reading.
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        // This callback will be executed once the buffer is ready.
        // We send the result to our channel. unwrap() is fine here as the receiver won't be dropped.
        sender.send(result).unwrap();
    });

    // 7. Poll the device and block this thread until the GPU has finished all work.
    // This is what makes the function synchronous in practice, despite the `async` keyword.
    device.poll(wgpu::MaintainBase::Wait).unwrap();

    // 8. Block and wait for the result from the `map_async` callback.
    // The first `unwrap()` panics if the channel fails (should not happen).
    // The second `unwrap()` panics if the buffer mapping operation itself returns an error.
    receiver.recv().unwrap().unwrap();

    // 9. Get a mapped view of the data in the staging buffer.
    let data = buffer_slice.get_mapped_range();

    // 10. Cast the raw bytes to our target type `T`, create a Vec, and return it.
    let result = bytemuck::cast_slice(&data).to_vec();

    // 11. The `data` guard is dropped here, which unmaps the buffer.
    // We can also call `unmap` explicitly for clarity.
    drop(data);
    staging_buffer.unmap();

    result
}

async fn test_sort(sorter: &GPUSorter, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    // simply runs a small sort and check if the sorting result is correct
    let n = 8192; // means that 2 workgroups are needed for sorting
    let scrambled_data: Vec<f32> = (0..n).rev().map(|x| x as f32).collect();
    let sorted_data: Vec<f32> = (0..n).map(|x| x as f32).collect();

    let sort_buffers = sorter.create_sort_buffers(device, NonZeroU32::new(n).unwrap());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });
    upload_to_buffer(
        &mut encoder,
        &sort_buffers.keys(),
        device,
        scrambled_data.as_slice(),
    );

    sorter.sort(&mut encoder, queue, &sort_buffers,None);
    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();

    let sorted = download_buffer::<f32>(
        &sort_buffers.keys(),
        device,
        queue,
        0..sort_buffers.keys_valid_size(),
    )
    .await;
    return sorted.into_iter().zip(sorted_data.into_iter()).all(|(a,b)|a==b);
}

/// Function guesses the best subgroup size by testing the sorter with
/// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
pub async fn guess_workgroup_size(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<u32> {
    let mut cur_sorter: GPUSorter;

    log::debug!("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

    let mut best = None;
    for subgroup_size in [1, 8, 16, 32, 64, 128] {
        log::debug!("Checking sorting with subgroupsize {}", subgroup_size);

        cur_sorter = GPUSorter::new(device, subgroup_size);
        let sort_success = test_sort(&cur_sorter, device, queue).await;

        log::debug!("{} worked: {}", subgroup_size, sort_success);

        if !sort_success {
            break;
        } else {
            best = Some(subgroup_size)
        }
    }
    return best;
}
