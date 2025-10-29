use std::time::{Duration, Instant};
use glam::{Vec2, Vec4};
use rand::{random_range, Rng};
use wgpu_profiler::GpuProfiler;
use winit::event::{ElementState};
use crate::{renderer::{camera::Camera, renderable::Renderable}, utils::gpu_buffer::GpuBuffer};
use crate::grid::grid::UNUSED_CELL_ID;
use crate::particles::{particle_integration::ParticleIntegration, particle_buffers::ParticleBuffers};
use crate::particles::particle_drawer::ParticleDrawer;
use crate::particles::particle_sort::ParticleSort;
use crate::renderer::wgpu_context::WgpuContext;

const SORT_INTERVAL_SECONDS: u64 = 4;
const SORT_INTERVAL: Duration = Duration::from_millis(SORT_INTERVAL_SECONDS * 1000); 

pub struct ParticleSystem {
    particle_buffers: ParticleBuffers,
    particle_buffers_copy: ParticleBuffers,
    particle_drawer: Option<ParticleDrawer>, 
    max_radius: f32,
    particle_integration: ParticleIntegration,
    particle_sort: ParticleSort,
    last_sort_time: Instant,
}

impl ParticleSystem {
    pub fn new(wgpu_context: &WgpuContext, camera: &Camera, world_size: Vec2) -> Self {
        const NUM_PARTICLES: usize = 1_000_000;
        
        let ((buffers, buffers_copy), max_radius) = Self::generate_initial_particles(wgpu_context, &world_size, NUM_PARTICLES);
        
        let particle_integration = ParticleIntegration::new(wgpu_context, &buffers, &world_size);
       
        let particle_drawer = ParticleDrawer::new(wgpu_context, &buffers, &camera);
        
        let particle_sort = ParticleSort::new(wgpu_context, &buffers, &buffers_copy);

        Self {
            particle_buffers: buffers,
            particle_buffers_copy: buffers_copy,
            particle_drawer: Some(particle_drawer),
            particle_sort,
            max_radius,
            particle_integration,
            last_sort_time: Instant::now() - SORT_INTERVAL,
        }
    }

    pub fn new_from_buffers(wgpu_context: &WgpuContext, current_positions: GpuBuffer<Vec2>, radii: GpuBuffer<f32>) -> Self {
        let total_particles = current_positions.len();
        let max_radius: f32 = radii.data().iter().max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap()).unwrap().clone();
        
        let previous_positions_pong = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::STORAGE);
        let current_positions_pong = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let radii_pong = GpuBuffer::new(wgpu_context, radii.data().clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let colors_pong = GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX);
        let home_cell_ids_buffer = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; total_particles],
            wgpu::BufferUsages::STORAGE);


        let buffers_pong = ParticleBuffers {
            home_cell_ids: home_cell_ids_buffer,
            previous_positions: previous_positions_pong,
            current_positions: current_positions_pong,
            radii: radii_pong,
            colors: colors_pong,
        };
        
        let previous_positions = GpuBuffer::new(wgpu_context, current_positions.data().clone(), wgpu::BufferUsages::STORAGE);
        let colors = GpuBuffer::new(wgpu_context, vec![glam::vec4(0.1, 0.4, 0.5, 1.0)], wgpu::BufferUsages::VERTEX);
        let home_cell_ids_copy = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; total_particles],
            wgpu::BufferUsages::STORAGE);
        
        let buffers_ping = ParticleBuffers{
            home_cell_ids: home_cell_ids_copy,
            current_positions,
            previous_positions, 
            radii,
            colors,
        };

        let particle_kernels = ParticleIntegration::new(wgpu_context, &buffers_ping, &Vec2::new(1920.0, 1080.0));
        
        let particle_sort = ParticleSort::new(wgpu_context, &buffers_ping, &buffers_pong);
        
        Self {
            particle_buffers: buffers_ping,
            particle_buffers_copy: buffers_pong,
            particle_drawer: None,
            particle_sort,
            max_radius,
            particle_integration: particle_kernels,
            last_sort_time: Instant::now() - SORT_INTERVAL,
        }
    }

    /// Generates the initial particle data and buffers.
    fn generate_initial_particles(wgpu_context: &WgpuContext, world_size: &Vec2, num_particles: usize) -> ((ParticleBuffers, ParticleBuffers), f32){
        let world_width: f32 = world_size.x;
        let world_height: f32 = world_size.y;

        let mut rng = rand::rng();

        let mut positions = Vec::with_capacity(num_particles);
        let mut radii = Vec::with_capacity(num_particles);
        let mut colors = Vec::with_capacity(num_particles);
        let mut max_radius = f32::MIN;

        for _ in 0..num_particles as u32 {
            let x = rng.random_range(0.0..world_width);
            let y = rng.random_range(0.0..world_height);
            positions.push(Vec2::new(x, y));
            let radius = rng.random_range(0.5..= 0.5) as f32;
            colors.push(glam::vec4(rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), rng.random_range(0.3..0.8), 1.0));
            if radius > max_radius {
                max_radius = radius;
            }
            radii.push(radius);
        }


        let current_positions = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let previous_positions = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::STORAGE);
        let radius = GpuBuffer::new(wgpu_context, radii.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let home_cell_ids_buffer = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; num_particles],
            wgpu::BufferUsages::STORAGE);
        
        
        let buffers = ParticleBuffers{
            home_cell_ids: home_cell_ids_buffer,
            current_positions,
            previous_positions,
            radii: radius,
            colors: GpuBuffer::new(wgpu_context, colors.clone(), wgpu::BufferUsages::VERTEX),
        };
                
        
        let current_positions_copy = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let previous_positions_copy = GpuBuffer::new(wgpu_context, positions.clone(), wgpu::BufferUsages::STORAGE);
        let radius_copy = GpuBuffer::new(wgpu_context, radii.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let colors_copy = GpuBuffer::new(wgpu_context, colors.clone(), wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE);
        let home_cell_ids_copy = GpuBuffer::new(
            wgpu_context,
            vec![UNUSED_CELL_ID; num_particles],
            wgpu::BufferUsages::STORAGE);
        let buffers_copy = ParticleBuffers {
            home_cell_ids: home_cell_ids_copy,
            current_positions: current_positions_copy,
            previous_positions: previous_positions_copy,
            radii: radius_copy, 
            colors: colors_copy,
        };
        
        ((buffers, buffers_copy), max_radius)
    }

    pub fn add_particles(&mut self, mouse_pos: &Vec2, wgpu_context: &WgpuContext){
        
        
        for i in 0..100 {
            // Generate a random angle (0 to 2*PI radians)
            let angle = random_range(0.0..std::f32::consts::TAU); // TAU is 2*PI

            // Generate a random radius (from mouse_pos)
            // Start the minimum radius higher to avoid center clumping
            // And potentially make the maximum radius larger or adjust its scaling
            let min_radius = 10.0 ; // Minimum distance from the center
            let max_radius = 50.0 + (i as f32 * 1.5); // Example: Gradually increase max radius
            let radius = random_range(min_radius..=max_radius);


            // Convert polar coordinates to Cartesian (x, y)
            let offset_x = radius * angle.cos();
            let offset_y = radius * angle.sin();

            let pos: Vec2 = mouse_pos + Vec2::new(offset_x, offset_y);

            self.particle_buffers.current_positions.push(pos.clone(), wgpu_context);
            self.particle_buffers_copy.current_positions.push(pos.clone(), wgpu_context);
            self.particle_buffers.previous_positions.push(pos, wgpu_context);
            self.particle_buffers_copy.previous_positions.push(pos, wgpu_context);

            let rng_radius_particle = random_range(1..=3) as f32; 
            self.particle_buffers.radii.push(
                rng_radius_particle,
                wgpu_context
            );
            self.particle_buffers_copy.radii.push(
                rng_radius_particle,
                wgpu_context
            );

            self.max_radius = self.max_radius.max(rng_radius_particle);

            self.particle_buffers.colors.push(
                glam::vec4(random_range(0.3..1.0), random_range(0.3..1.0), random_range(0.3..1.0), 1.0),
                wgpu_context
            );
            self.particle_buffers_copy.colors.push(
                glam::vec4(random_range(0.3..1.0), random_range(0.3..1.0), random_range(0.3..1.0), 1.0),
                wgpu_context
            );
            
            self.particle_buffers.home_cell_ids.push(UNUSED_CELL_ID, wgpu_context);
            self.particle_buffers_copy.home_cell_ids.push(UNUSED_CELL_ID, wgpu_context);
            
        }
        
        self.particle_sort.refresh(wgpu_context, &self.particle_buffers, &self.particle_buffers_copy);
        self.particle_integration.refresh(wgpu_context, &self.particle_buffers);
        self.particle_drawer.as_mut().expect("Particle drawer null").refresh(wgpu_context, &self.particle_buffers);
        
        println!("Total particles: {}", self.len());
    }
    pub fn mouse_click_callback(&mut self, mouse_state: &ElementState, position: Vec2){
        self.particle_integration.mouse_click_callback(mouse_state, position);

    }
    pub fn mouse_move_callback(&mut self, position: Vec2){
        self.particle_integration.mouse_move_callback(position);
    }
    
    pub fn is_it_time_to_sort(&self) -> bool {
        self.last_sort_time.elapsed() >= SORT_INTERVAL
    }
    
    pub fn reset_last_sort_time(&mut self) {
        self.last_sort_time = Instant::now();
    }
    pub fn sort_by_cell_id(&self, encoder: &mut wgpu::CommandEncoder, gpu_profiler: &mut GpuProfiler, cell_size: f32){
        self.particle_sort.sort(
            encoder,
            gpu_profiler,
            self,
            cell_size,
        );
    }

    pub fn update_positions(&mut self, delta_time:f32, wgpu_context: &WgpuContext, gpu_profiler: &mut GpuProfiler) {
        self.particle_integration.update_positions(wgpu_context, gpu_profiler, delta_time);
    }
    
    
    pub fn download_home_cell_ids(&mut self, wgpu_context: &WgpuContext) -> Vec<u32>{
        self.particle_buffers.home_cell_ids.download(wgpu_context).unwrap().clone()
    }

    pub fn download_particle_ids(&mut self, wgpu_context: &WgpuContext) -> Vec<u32>{
        self.particle_sort.download_particle_ids(wgpu_context).clone()
    }

    pub fn download_particle_buffers(&mut self, wgpu_context: &WgpuContext) -> &ParticleBuffers{
        let _ = self.particle_buffers.current_positions.download(wgpu_context);
        let _ = self.particle_buffers.radii.download(wgpu_context);
        let _ = self.particle_buffers.previous_positions.download(wgpu_context);
        let _ = self.particle_buffers.colors.download(wgpu_context);
        let _ = self.particle_buffers.home_cell_ids.download(wgpu_context);
        &self.particle_buffers
    }

    pub fn buffers(&self) -> &ParticleBuffers {
        &self.particle_buffers
    }

    pub fn copy_buffers(&self) -> &ParticleBuffers {
        &self.particle_buffers_copy
    }

    pub fn len(&self) -> usize {
        self.buffers().current_positions.len()
    }

    pub fn positions(&self) -> &GpuBuffer<Vec2>{
        &self.buffers().current_positions
    }

    pub fn radius(&self) -> &GpuBuffer<f32> {
        &self.buffers().radii
    }

    pub fn color(&self) -> &[Vec4] {
        self.buffers().colors.data()
    }

    pub fn get_max_radius(&self) -> f32 {
        self.max_radius
    }

}



impl Renderable for ParticleSystem {
    fn draw(&self, render_pass: &mut wgpu::RenderPass, camera: &Camera){
        self.particle_drawer.as_ref().expect("Particle drawer null").draw(render_pass, camera, self.len() as u32);
    }

}
