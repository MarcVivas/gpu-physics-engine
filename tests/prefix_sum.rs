use rand::{random_range, Rng};
use game_engine::utils::gpu_buffer::GpuBuffer;
use game_engine::utils::prefix_sum::prefix_sum::PrefixSum;

mod common;
#[test]
fn inclusive_prefix_sum_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    let n = 81_920;
    let mut original_values: Vec<u32> = (0..n).rev().collect();

    let mut buffer_data = GpuBuffer::new(wgpu_context, original_values.clone(), wgpu::BufferUsages::STORAGE);


    let prefix_sum = PrefixSum::new(
        wgpu_context,
        &buffer_data
    );


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Testing prefix sum"),
    });

    prefix_sum.execute(wgpu_context, &mut encoder, original_values.len() as u32);


    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();


    let result = buffer_data.download(wgpu_context).unwrap();
    let expected_data: Vec<u32> = original_values.iter().scan(0, |sum, i| {
        *sum += *i;
        Some(*sum)
    }).collect();

    assert_eq!(result.len(), expected_data.len());
    assert_eq!(*result, expected_data);
}


#[test]
fn inclusive_prefix_sum_same_values_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    let n = 83090;
    let mut original_values: Vec<u32> = vec![1; n];

    let mut buffer_data = GpuBuffer::new(wgpu_context, original_values.clone(), wgpu::BufferUsages::STORAGE);


    let mut prefix_sum = PrefixSum::new(
        wgpu_context,
        &buffer_data
    );


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Testing prefix sum"),
    });

    prefix_sum.execute(wgpu_context, &mut encoder, original_values.len() as u32);


    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();


    let result = buffer_data.download(wgpu_context).unwrap();
    let expected_data: Vec<u32> = original_values.iter().scan(0, |sum, i| {
        *sum += *i;
        Some(*sum)
    }).collect();

    assert_eq!(result.len(), expected_data.len());
    assert_eq!(*result, expected_data);
}

#[test]
fn inclusive_prefix_sum_all_zero_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    let n = 81920;
    let mut original_values: Vec<u32> = vec![0; n];

    let mut buffer_data = GpuBuffer::new(wgpu_context, original_values.clone(), wgpu::BufferUsages::STORAGE);


    let prefix_sum = PrefixSum::new(
        wgpu_context,
        &buffer_data
    );


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Testing prefix sum"),
    });

    prefix_sum.execute(wgpu_context, &mut encoder, original_values.len() as u32);


    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();


    let result = buffer_data.download(wgpu_context).unwrap();
    let expected_data: Vec<u32> = original_values.iter().scan(0, |sum, i| {
        *sum += *i;
        Some(*sum)
    }).collect();

    assert_eq!(result.len(), expected_data.len());
    assert_eq!(*result, expected_data);
}


#[test]
fn inclusive_prefix_sum_random_test() {
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();

    let n = random_range(10_381_920u32..=14_381_920u32);
    let original_values: Vec<u32> = (0u32..n).map(|_| random_range(0u32..=9u32)).collect();

    let mut buffer_data = GpuBuffer::new(wgpu_context, original_values.clone(), wgpu::BufferUsages::STORAGE);

    let mut prefix_sum = PrefixSum::new(
        wgpu_context,
        &buffer_data
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Testing random prefix sum"),
    });
    

    prefix_sum.execute(wgpu_context, &mut encoder, original_values.len() as u32);

    let idx = queue.submit([encoder.finish()]);
    device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(idx)).unwrap();

    let result = buffer_data.download(wgpu_context).unwrap();
    let expected_data: Vec<u32> = original_values.iter().scan(0, |sum, i| {
        *sum += *i;
        Some(*sum)
    }).collect();

    assert_eq!(result.len(), expected_data.len());
    assert_eq!(*result, expected_data);
}