use glam::Vec2;
use wgpu::wgt::PollType::WaitForSubmissionIndex;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};
use game_engine::grid::grid::Grid;

mod common;

#[test]
fn sort_particles_test(){

    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let device = wgpu_context.get_device();
    let queue = wgpu_context.get_queue();
    
    let max_radius = 10.0;  

    let particle_positions = vec![
        // Particle 0: Crosses into 3 neighbors. -> Home cell id = 2
        Vec2::new(20.0, 42.0),
        // Particle 1: Fully contained within one cell. -> Home cell id = 15
        Vec2::new(77.0, 77.0),
        // Particle 2: Edge case at the origin, fully contained. -> Home cell id = 0
        Vec2::new(5.0, 5.0),
    ];

    // Give each particle a different radius to ensure the radius buffer is read correctly.
    let particle_radii = vec![10.0, 8.0, 1.0];

    let mut particle_system = common::create_test_particle_system(
        wgpu_context,
        particle_positions,
        particle_radii,
    );

    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Particle sort test Encoder") }
    );
    let mut gpu_profiler = GpuProfiler::new(wgpu_context.get_device(), GpuProfilerSettings::default()).unwrap();
    particle_system.sort_by_cell_id(&mut encoder, &mut gpu_profiler, Grid::compute_cell_size(max_radius));
    let idx = queue.submit([encoder.finish()]);
    device.poll(WaitForSubmissionIndex(idx)).unwrap();
    
    let actual_home_cell_ids = particle_system.download_home_cell_ids(wgpu_context);
    let actual_particle_ids = particle_system.download_particle_ids(wgpu_context);
    
    let expected_home_cell_ids = vec![0, 2, 15];
    let expected_particle_ids = vec![2, 0, 1];
    assert_eq!(expected_home_cell_ids, actual_home_cell_ids);
    assert_eq!(expected_particle_ids, actual_particle_ids);
    
    let expected_particle_positions = vec![
        Vec2::new(5.0, 5.0),
        Vec2::new(20.0, 42.0),
        Vec2::new(77.0, 77.0),
    ];
    
    let expected_previous_positions = expected_particle_positions.clone();
    let expected_radii = vec![1.0, 10.0, 8.0];
    
    let actual_particle_buffers = particle_system.download_particle_buffers(wgpu_context);
    
    let actual_particle_positions = actual_particle_buffers.current_positions.data();
    let actual_previous_positions = actual_particle_buffers.previous_positions.data();
    let actual_radii = actual_particle_buffers.radii.data();
    
    assert_eq!(expected_particle_positions, *actual_particle_positions);
    assert_eq!(expected_previous_positions, *actual_previous_positions);
    assert_eq!(expected_radii, *actual_radii);
}