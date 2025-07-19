use std::num::NonZeroU32;
use game_engine::utils::gpu_buffer::GpuBuffer;
use game_engine::utils::radix_sort::radix_sort::GPUSorter;
use game_engine::utils::guess_workgroup_size;
mod common;
#[test]
fn sort_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    
    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    let sorter: GPUSorter = GPUSorter::new(device, guess_workgroup_size(wgpu_context).unwrap());
    
    
    // simply runs a small sort and check if the sorting result is correct
    let n = 8192; // means that 2 workgroups are needed for sorting
    let mut scrambled_data: Vec<u32> = (0..n).rev().collect();
    let required_len = GPUSorter::get_required_keys_buffer_size(scrambled_data.len() as u32);
    scrambled_data.resize(required_len as usize, u32::MAX);
    
    

    let mut scrambled_keys_buffer: GpuBuffer<u32> = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);
    let mut scrambled_payload_buffer: GpuBuffer<u32> = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);


    let sorted_data: Vec<u32> = (0..n).collect();

    let sort_buffers = sorter.create_sort_buffers(device, NonZeroU32::new(n).unwrap(), scrambled_keys_buffer.buffer(), scrambled_payload_buffer.buffer());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });


    sorter.sort(&mut encoder, queue, &sort_buffers, None);
    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();



    assert_eq!(scrambled_keys_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize], sorted_data);
    assert_eq!(scrambled_payload_buffer.download(wgpu_context).unwrap().as_slice()[0..n as usize], sorted_data);

}