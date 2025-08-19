use std::num::NonZeroU32;
use wgpu::wgt::PollType::WaitForSubmissionIndex;
use game_engine::utils::gpu_buffer::GpuBuffer;
use game_engine::utils::radix_sort::radix_sort::{GPUSorter, PushConstants, NUM_BLOCKS_PER_WORKGROUP, RADIX_SORT_BUCKETS, WORKGROUP_SIZE};
mod common;
#[test]
fn sort_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    
    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    // simply runs a small sort and check if the sorting result is correct
    let n = 25006; // means that 2 workgroups are needed for sorting
    let mut scrambled_data: Vec<u32> = (0..n).rev().collect();
    let required_len = scrambled_data.len();
    let mut scrambled_keys_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);
    let mut scrambled_payload_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);


    let mut sorter: GPUSorter = GPUSorter::new(wgpu_context, NonZeroU32::new(n).unwrap(), &scrambled_keys_buffer, &scrambled_payload_buffer);
    
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });


    sorter.sort(&mut encoder, wgpu_context, None);
    let idx = queue.submit([encoder.finish()]);
    device.poll(WaitForSubmissionIndex(idx)).unwrap();

    let sorted_data: Vec<u32> = (0..n).collect();
    
    
    let keys_result = scrambled_keys_buffer.download(wgpu_context).unwrap();
    let payload_result = scrambled_payload_buffer.download(wgpu_context).unwrap();
    
    println!("{:?}", sorter.get_keys_b(wgpu_context));
    
    assert_eq!(keys_result.len(), required_len);
    assert_eq!(payload_result.len(), required_len);
    
    assert_eq!(keys_result.as_slice(), sorted_data);
    assert_eq!(payload_result.as_slice(), sorted_data);

}


#[test]
fn sort_test_small_sized_array() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    // simply runs a small sort and check if the sorting result is correct
    let mut scrambled_data: Vec<u32> = vec![357_000_000, 90_000, 257, 2, 20_000_000, 1, 30_000, 65611];
    let n = scrambled_data.len() as u32;
    let mut scrambled_keys_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);
    let mut scrambled_payload_buffer = GpuBuffer::new(wgpu_context, scrambled_data.clone(), wgpu::BufferUsages::STORAGE);


    let mut sorter: GPUSorter = GPUSorter::new(wgpu_context, NonZeroU32::new(n).unwrap(), &scrambled_keys_buffer, &scrambled_payload_buffer);




    let num_elements = n;
    let total_threads = ((num_elements + NUM_BLOCKS_PER_WORKGROUP - 1) / NUM_BLOCKS_PER_WORKGROUP, 1, 1);
    let num_workgroups = (total_threads.0 + WORKGROUP_SIZE.0 - 1) / WORKGROUP_SIZE.0;
    // First histogram kernel launch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });
    sorter.build_histogram(
        &mut encoder,
        total_threads,
        &PushConstants{
            num_elements,
            current_shift: 0,
            num_workgroups,
            num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
        },
        &true
    );

    let idx = queue.submit([encoder.finish()]);
    device.poll(WaitForSubmissionIndex(idx)).unwrap();

    let histogram = sorter.get_histogram(wgpu_context).unwrap();
    assert_eq!(histogram.iter().sum::<u32>(), n);
    assert_eq!(histogram.len(), 256);
    let mut expected_histogram = vec![0; 256];
    for elem in scrambled_data.iter() {
        let index = (*elem >> 0) & (RADIX_SORT_BUCKETS - 1u32); 
        expected_histogram[index as usize] += 1;
    }
    assert_eq!(*histogram, expected_histogram);
    
    println!("{:?}", histogram);


    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GPURSSorter test_sort"),
    });
    sorter.scatter(
        &mut encoder,
        total_threads,
        &PushConstants{
            num_elements,
            current_shift: 0,
            num_workgroups,
            num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
        },
        &true
    );

    let idx = queue.submit([encoder.finish()]);
    device.poll(WaitForSubmissionIndex(idx)).unwrap();
    
    // Keys b 
    assert_eq!(*sorter.get_keys_b(wgpu_context).unwrap(), vec![20_000_000, 257, 1, 2, 30_000, 357_000_000, 65611, 90_000]);


}