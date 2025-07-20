mod common;

use game_engine::game::grid::grid::Grid;
use glam::Vec2;
use std::rc::Rc;
use std::cell::RefCell;
use game_engine::renderer::renderable::Renderable;
use game_engine::renderer::wgpu_context::WgpuContext;

const UNUSED_CELL_ID: u32 = 0xffffffff;
#[test]
fn test_grid_build_cell_ids_with_multiple_particles() {
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let (mut grid, num_particles) = build_grid_case_1(wgpu_context);

    // DEFINE GOLDEN OUTPUT for all particles.
    let mut expected_cell_ids: Vec<u32> = Vec::new();
    let mut expected_object_ids: Vec<u32> = Vec::new();

    // --- Particle 0 Results ---
    // Pos: (20, 42), Radius: 10.0, Home Cell: (0, 1) -> hash 65536
    // Neighbors: (1,1), (0,2), (1,2)
    expected_cell_ids.extend_from_slice(&[
        65536,  // Home cell (0, 1)
        65537,  // Neighbor (1, 1)
        131072, // Neighbor (0, 2)
        131073, // Neighbor (1, 2)
    ]);
    expected_object_ids.extend_from_slice(&[0, 0, 0, 0]);

    // --- Particle 1 Results ---
    // Pos: (77, 77), Radius: 8.0, Home Cell: (3, 3) -> hash 196611
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        196611,       // Home cell (3, 3)
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
    ]);
    expected_object_ids.extend_from_slice(&[1, 0, 0, 0]);

    // --- Particle 2 Results ---
    // Pos: (5, 5), Radius: 1.0, Home Cell: (0, 0) -> hash 0
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        0,            // Home cell (0, 0)
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
    ]);
    expected_object_ids.extend_from_slice(&[2, 0, 0, 0]);


    // ACT
    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Multi-Particle Test Encoder") }
    );
    grid.build_cell_ids(&mut encoder, num_particles);
    wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));


    // ASSERT
    let binding = grid.download_cell_ids(wgpu_context).unwrap();
    let gpu_cell_ids = &binding.as_slice()[0..expected_cell_ids.len()];
    assert_eq!(*gpu_cell_ids, expected_cell_ids);
    let gpu_object_ids = grid.download_object_ids(wgpu_context).unwrap();
    assert_eq!(*gpu_object_ids, expected_object_ids);
}


fn build_grid_case_1(wgpu_context: &WgpuContext) -> (Grid, u32) {
    // ARRANGE
    let world_dimensions = Vec2::new(200.0, 200.0);
    let max_radius = 10.0; // This implicitly sets cell_size to 22.0

    let particle_positions = vec![
        // Particle 0: Crosses into 3 neighbors.
        Vec2::new(20.0, 42.0),
        // Particle 1: Fully contained within one cell.
        Vec2::new(77.0, 77.0),
        // Particle 2: Edge case at the origin, fully contained.
        Vec2::new(5.0, 5.0),
    ];
    let num_particles = particle_positions.len();
    
    // Give each particle a different radius to ensure the radius buffer is read correctly.
    let particle_radii = vec![10.0, 8.0, 1.0];

    let particle_system = common::create_test_particle_system(
        wgpu_context,
        particle_positions,
        particle_radii,
    );

    (Grid::new_without_camera(
        wgpu_context,
        world_dimensions,
        max_radius,
        Rc::new(RefCell::new(particle_system)),
    ), num_particles as u32)
}
#[test]
pub fn test_grid_build_cell_ids_and_sort(){
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    let (mut grid, num_particles) = build_grid_case_1(wgpu_context);

    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Multi-Particle Test Encoder") }
    );
    
    grid.build_cell_ids(&mut encoder, num_particles);
    grid.sort_map(&mut encoder, wgpu_context);
    
    wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

    let mut expected_cell_ids: Vec<u32> = Vec::new();
    let mut expected_object_ids: Vec<u32> = Vec::new();

    // --- Particle 0 Results ---
    // Pos: (20, 42), Radius: 10.0, Home Cell: (0, 1) -> hash 65536
    // Neighbors: (1,1), (0,2), (1,2)
    expected_cell_ids.extend_from_slice(&[
        65536,  // Home cell (0, 1)
        65537,  // Neighbor (1, 1)
        131072, // Neighbor (0, 2)
        131073, // Neighbor (1, 2)
    ]);
    expected_object_ids.extend_from_slice(&[0, 0, 0, 0]);

    // --- Particle 1 Results ---
    // Pos: (77, 77), Radius: 8.0, Home Cell: (3, 3) -> hash 196611
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        196611,       // Home cell (3, 3)
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
    ]);
    expected_object_ids.extend_from_slice(&[1, 0, 0, 0]);

    // --- Particle 2 Results ---
    // Pos: (5, 5), Radius: 1.0, Home Cell: (0, 0) -> hash 0
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        0,            // Home cell (0, 0)
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
        UNUSED_CELL_ID,
    ]);
    expected_object_ids.extend_from_slice(&[2, 0, 0, 0]);

    let mut expected: Vec<(u32, u32)> = expected_cell_ids.iter()
        .zip(expected_object_ids.iter())
        .map(|(&cell_id, &object_id)| (cell_id, object_id))
        .collect();
    expected.sort();

    let mut gpu_cell_ids = grid.download_cell_ids(wgpu_context).unwrap();
    gpu_cell_ids = Vec::from(&gpu_cell_ids.as_slice()[0..expected_cell_ids.len()]);
    let gpu_object_ids = grid.download_object_ids(wgpu_context).unwrap(); 
    let sorted_result = gpu_cell_ids.iter().zip(gpu_object_ids.iter()).map(|(&cell_id, &object_id)| (cell_id, object_id)).collect::<Vec<(u32, u32)>>();    
    assert_eq!(sorted_result, expected);
    
}