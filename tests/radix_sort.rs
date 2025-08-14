use std::num::NonZeroU32;
use wgpu::wgt::PollType::WaitForSubmissionIndex;
use game_engine::utils::gpu_buffer::GpuBuffer;
use game_engine::utils::radix_sort::radix_sort::GPUSorter;
use game_engine::utils::get_subgroup_size;
mod common;
#[test]
fn sort_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    
    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    // simply runs a small sort and check if the sorting result is correct
    let n = 8192; // means that 2 workgroups are needed for sorting
    let mut scrambled_data: Vec<u32> = (0..n).rev().collect();
    let required_len = GPUSorter::get_required_keys_buffer_size(scrambled_data.len() as u32);
    scrambled_data.resize(required_len as usize, u32::MAX);
    let mut scrambled_keys_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);
    let mut scrambled_payload_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);


    let sorter: GPUSorter = GPUSorter::new(device, get_subgroup_size(wgpu_context).unwrap(), NonZeroU32::new(n).unwrap(), &scrambled_keys_buffer, &scrambled_payload_buffer);
    
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });


    sorter.sort(&mut encoder, queue, None);
    let idx = queue.submit([encoder.finish()]);
    device.poll(WaitForSubmissionIndex(idx)).unwrap();


    let sorted_data: Vec<u32> = (0..n).collect();

    assert_eq!(scrambled_keys_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize], sorted_data);
    assert_eq!(scrambled_payload_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize], sorted_data);

}