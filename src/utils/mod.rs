use std::num::NonZeroU32;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;
use crate::utils::radix_sort::radix_sort::GPUSorter;

pub mod gpu_buffer;
pub mod compute_shader;
pub mod radix_sort;


/// Function guesses the best subgroup size by testing the sorter with
/// subgroup sizes 1,8,16,32,64,128 and returning the largest subgroup size that worked.
pub fn guess_workgroup_size(wgpu_context: &WgpuContext) -> Option<u32> {
    let mut current_sorter: GPUSorter;
    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();
    
    log::debug!("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");

    let mut best = None;
    for subgroup_size in [1, 8, 16, 32, 64, 128] {
        log::debug!("Checking sorting with subgroupsize {}", subgroup_size);

        // simply runs a small sort and check if the sorting result is correct
        let n = 8192; // means that 2 workgroups are needed for sorting
        let mut scrambled_data: Vec<u32> = (0..n).rev().collect();
        let required_len = GPUSorter::get_required_keys_buffer_size(scrambled_data.len() as u32);
        scrambled_data.resize(required_len as usize, u32::MAX);
        let mut scrambled_keys_buffer: GpuBuffer<u32> = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);
        let mut scrambled_payload_buffer: GpuBuffer<u32> = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);

        current_sorter = GPUSorter::new(device, subgroup_size, NonZeroU32::new(n).unwrap(), scrambled_keys_buffer.buffer(), scrambled_payload_buffer.buffer());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPURSSorter test_sort"),
        });


        current_sorter.sort(&mut encoder, queue, None);
        let idx = queue.submit([encoder.finish()]);
        device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();


        let sorted_data: Vec<u32> = (0..n).collect();
        
        let sort_success = scrambled_keys_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize] == sorted_data &&
            scrambled_payload_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize] == sorted_data;

        log::debug!("{} worked: {}", subgroup_size, sort_success);

        if !sort_success {
            break;
        } else {
            best = Some(subgroup_size)
        }
    }
    best
}
