use wgpu::CommandEncoder;
use wgpu_profiler::GpuProfiler;
use crate::grid::grid::Grid;
use crate::particles::particle_system::ParticleSystem;
use crate::physics::collision_cell_builder::CollisionCellBuilder;
use crate::physics::collision_solver::CollisionSolver;
use crate::renderer::wgpu_context::WgpuContext;
use crate::utils::gpu_buffer::GpuBuffer;

pub struct CollisionSystem {
    collision_cell_builder: CollisionCellBuilder,
    collision_solver: CollisionSolver,
}
impl CollisionSystem {
    pub fn new(wgpu_context: &WgpuContext, dim: u32, particle_system: &ParticleSystem, grid: &Grid) -> Self {
        let collision_cell_builder = CollisionCellBuilder::new(wgpu_context, particle_system.len(), dim, grid);
        let collision_solver = CollisionSolver::new(wgpu_context, particle_system, grid, &collision_cell_builder);
        
        Self {
            collision_solver,
            collision_cell_builder,
        }
    }
    
    pub fn refresh(&mut self, wgpu_context: &WgpuContext, particle_system: &ParticleSystem, grid: &Grid, particles_added: usize){
        let new_buffer_size = particles_added * 4; 
        self.collision_cell_builder.refresh_buffers(wgpu_context, new_buffer_size, grid);
        self.collision_solver.refresh_buffers(wgpu_context, particle_system, grid, &self.collision_cell_builder);
    }
    
    pub fn solve_collisions(&mut self, wgpu_context: &WgpuContext, dt: f32, mut encoder: CommandEncoder, gpu_profiler: &mut GpuProfiler){
        self.collision_cell_builder.build_collision_cells(wgpu_context, &mut encoder, gpu_profiler);
        gpu_profiler.resolve_queries(&mut encoder);

        // Submit the commands to the GPU
        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        
        let indirect_dispatch_buffer = self.collision_cell_builder.indirect_dispatch_buffer(); 
        self.collision_solver.solve_collisions(wgpu_context, gpu_profiler, indirect_dispatch_buffer);
    }
    
    pub fn download_collision_cells(&mut self, wgpu_context: &WgpuContext) -> Vec<u32>{
        self.collision_cell_builder.download_collision_cells(wgpu_context)
    }
    
}
