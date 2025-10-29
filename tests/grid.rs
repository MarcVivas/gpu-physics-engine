mod common;

use game_engine::grid::grid::Grid;
use glam::Vec2;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};
use game_engine::particles::particle_system::ParticleSystem;
use game_engine::physics::collision_system::CollisionSystem;
use game_engine::renderer::wgpu_context::WgpuContext;

const UNUSED_CELL_ID: u32 = 0xffffffff;
#[test]
fn test_grid_build_cell_ids_with_multiple_particles() {
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;

    let (mut grid, _) = build_case_1(wgpu_context);

    // DEFINE GOLDEN OUTPUT for all particles.
    let mut expected_cell_ids: Vec<u32> = Vec::new();
    let mut expected_object_ids: Vec<u32> = Vec::new();

    // --- Particle 0 Results ---
    // Pos: (20, 42), Radius: 10.0, Home Cell: (0, 1) -> hash 
    // Neighbors: (1,1), (0,2), (1,2)
    expected_cell_ids.extend_from_slice(&[
        morton_encode(0, 1),  // Home cell (0, 1)
        morton_encode(1, 1),  // Neighbor (1, 1)
        morton_encode(0, 2), // Neighbor (0, 2)
        morton_encode(1, 2), // Neighbor (1, 2)
    ]);
    expected_object_ids.extend_from_slice(&[0, 0, 0, 0]);

    // --- Particle 1 Results ---
    // Pos: (77, 77), Radius: 8.0, Home Cell: (3, 3) -> hash 
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        morton_encode(3, 3),       // Home cell (3, 3)
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
    grid.build_cell_ids(&mut encoder);
    wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));


    // ASSERT
    let binding = grid.download_cell_ids(wgpu_context).unwrap();
    let gpu_cell_ids = &binding.as_slice()[0..expected_cell_ids.len()];
    assert_eq!(*gpu_cell_ids, expected_cell_ids);
    let gpu_object_ids = grid.download_object_ids(wgpu_context).unwrap();
    assert_eq!(*gpu_object_ids, expected_object_ids);
}


/// Spreads the lower 16 bits of an integer to every other bit.
/// Example (2-bit): n = 3 (binary 11) becomes 5 (binary 0101).
fn split_by_bits(n: u32) -> u32 {
    let mut x = n & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Encodes 2D coordinates (16-bit max) into a 1D Morton index.
/// Example: (x=3, y=3) -> (binary 11, 11) -> interleaved 1111 -> 15.
fn morton_encode(x: u32, y: u32) -> u32 {
    split_by_bits(x) | (split_by_bits(y) << 1)
}

/// Compacts bits from every other position to the lower 16 bits.
/// This is the inverse of `split_by_bits`.
/// Example (2-bit): n = 5 (binary 0101) becomes 3 (binary 11).
fn unsplit_by_bits(n: u32) -> u32 {
    let mut x = n & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    x
}



fn build_case_1(wgpu_context: &WgpuContext) -> (Grid, ParticleSystem) {
    // ARRANGE
    let max_radius = 10.0; // This implicitly sets cell_size to 22.0

    let particle_positions = vec![
        // Particle 0: Crosses into 3 neighbors.
        Vec2::new(20.0, 42.0),
        // Particle 1: Fully contained within one cell.
        Vec2::new(77.0, 77.0),
        // Particle 2: Edge case at the origin, fully contained.
        Vec2::new(5.0, 5.0),
    ];
    
    // Give each particle a different radius to ensure the radius buffer is read correctly.
    let particle_radii = vec![10.0, 8.0, 1.0];

    let particle_system = common::create_test_particle_system(
        wgpu_context,
        particle_positions,
        particle_radii,
    );

    (Grid::new_without_camera(
        wgpu_context,
        max_radius,
        &particle_system,
    ), particle_system)
}
#[test]
pub fn test_grid_build_cell_ids_and_sort(){
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    let (mut grid, _) = build_case_1(wgpu_context);

    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Multi-Particle Test Encoder") }
    );
    
    grid.build_cell_ids(&mut encoder);
    grid.sort_map(&mut encoder);
    
    wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));

    let mut expected_cell_ids: Vec<u32> = Vec::new();
    let mut expected_object_ids: Vec<u32> = Vec::new();

    // --- Particle 0 Results ---
    // Pos: (20, 42), Radius: 10.0, Home Cell: (0, 1) -> hash 65536
    // Neighbors: (1,1), (0,2), (1,2)
    expected_cell_ids.extend_from_slice(&[
        morton_encode(0, 1),  // Home cell (0, 1)
        morton_encode(1, 1),  // Neighbor (1, 1)
        morton_encode(0, 2), // Neighbor (0, 2)
        morton_encode(1, 2), // Neighbor (1, 2)
    ]);
    expected_object_ids.extend_from_slice(&[0, 0, 0, 0]);

    // --- Particle 1 Results ---
    // Pos: (77, 77), Radius: 8.0, Home Cell: (3, 3) -> hash 196611
    // No neighbors touched.
    expected_cell_ids.extend_from_slice(&[
        morton_encode(3, 3),       // Home cell (3, 3)
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




#[test]
pub fn test_grid_build_cell_ids_sort_and_build_empty_collision_cells_list(){
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;
    let (mut grid, particles) = build_case_1(wgpu_context);

    let mut collision_system = CollisionSystem::new(wgpu_context, 2, &particles, &grid);
    
    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Multi-Particle Test Encoder") }
    );

    
    grid.build_cell_ids(&mut encoder);
    grid.sort_map(&mut encoder);
    collision_system.solve_collisions(wgpu_context, encoder, &mut GpuProfiler::new(wgpu_context.get_device(), GpuProfilerSettings::default()).unwrap());
    

    let expected_collision_cells: Vec<u32> = vec![UNUSED_CELL_ID; grid.download_object_ids(wgpu_context).unwrap().len()];
    let actual_collision_cells = collision_system.download_collision_cells(wgpu_context);

    assert_eq!(actual_collision_cells, expected_collision_cells);

}


fn build_case_2(wgpu_context: &WgpuContext, positions: Vec<Vec2>) -> (Grid, ParticleSystem) {
    // ARRANGE
    let max_radius = 10.0; // This implicitly sets cell_size to 22.0


    let num_particles = positions.len();

    // Give each particles a different radius to ensure the radius buffer is read correctly.
    let particle_radii = vec![10.0; num_particles];

    let particle_system = common::create_test_particle_system(
        wgpu_context,
        positions,
        particle_radii,
    );

    (
        Grid::new_without_camera(
        wgpu_context,
        max_radius,
        &particle_system,
        )
        , particle_system
    )
}

fn gen_case_2_particles() -> Vec<Vec2> {
    // 546 particles at the same position and therefore, cells
     vec![
        // Particle 0: Crosses into 3 neighbors.
        Vec2::new(20.0, 42.0);
        546
        
    ]
}
#[test]
pub fn test_grid_build_cell_ids_sort_and_build_collision_cells_list(){
    // SETUP
    let setup = pollster::block_on(common::setup());
    let wgpu_context = &setup.wgpu_context;


    let positions = gen_case_2_particles();
    let (mut grid, particles) = build_case_2(wgpu_context, positions.clone());
    let num_particles = positions.len();
    
    let mut collision_system = CollisionSystem::new(wgpu_context, 2, &particles, &grid);
    
    let mut encoder = wgpu_context.get_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("Multi-Particle Test Encoder") }
    );

    grid.build_cell_ids(&mut encoder);
    grid.sort_map(&mut encoder);
    collision_system.solve_collisions(wgpu_context, encoder, &mut GpuProfiler::new(wgpu_context.get_device(), GpuProfilerSettings::default()).unwrap());
    

    let mut expected_collision_cells: Vec<u32> = (0u32..4u32).map(|i| i * num_particles as u32).collect();
    expected_collision_cells.resize(grid.download_object_ids(wgpu_context).unwrap().len(), UNUSED_CELL_ID);
    let actual_collision_cells = collision_system.download_collision_cells(wgpu_context);

    assert_eq!(actual_collision_cells, expected_collision_cells);

}