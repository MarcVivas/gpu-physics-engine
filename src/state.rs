use std::sync::{Arc};
use glam::Vec2;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};
use winit::dpi;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;
use crate::particles::particle_system::ParticleSystem;
use crate::utils::input_manager::InputManager;
use crate::utils::render_timer::RenderTimer;
use crate::renderer::renderer::Renderer;
use crate::renderer::wgpu_context::WgpuContext;
use crate::grid::grid::Grid;
use crate::physics::collision_system::CollisionSystem;
use crate::renderer::renderable::Renderable;

const DIMENSION: u32 = 2; 

// This will store the state of the program
pub struct State {
    world_size: Vec2,
    wgpu_context: WgpuContext,
    render_timer: RenderTimer,
    renderer: Renderer,
    particles: ParticleSystem,
    grid: Grid,
    collision_system: CollisionSystem,
    mouse_position: Option<dpi::PhysicalPosition<f64>>,
    gpu_profiler: GpuProfiler,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let world_size = Vec2::new(3048.0, 1048.0);
        let wgpu_context = WgpuContext::new(window).await?;
        let renderer = Renderer::new(&wgpu_context, &world_size).unwrap();

        let particles = ParticleSystem::new(&wgpu_context, renderer.camera(), world_size);
        let grid =  Grid::new(&wgpu_context, renderer.camera(), world_size, &particles);

        let render_timer = RenderTimer::new();

        let mouse_position = None;
        
        
        #[cfg(feature = "benchmark")]
        let gpu_profiler = GpuProfiler::new(wgpu_context.get_device(), GpuProfilerSettings::default())?;
        #[cfg(not(feature = "benchmark"))]
        let gpu_profiler = GpuProfiler::new(wgpu_context.get_device(), GpuProfilerSettings{
            enable_timer_queries: false,
            enable_debug_groups: false,
            max_num_pending_frames: 1
        })?;
        
        let collision_system = CollisionSystem::new(&wgpu_context, DIMENSION, &particles, &grid);
        
        Ok(Self {
            world_size,
            wgpu_context,
            render_timer,
            particles,
            renderer,
            mouse_position,
            grid,
            gpu_profiler,
            collision_system
        })

    }

    
    /// This function is called every frame
    pub fn render_loop(&mut self, event: &WindowEvent, event_loop: &ActiveEventLoop){
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size ) => self.wgpu_context.resize(size.width, size.height),
            WindowEvent::RedrawRequested => self.update_and_redraw(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => InputManager::process_keyboard_input(self, event_loop, code, key_state),
            WindowEvent::CursorMoved { position, .. } => InputManager::process_cursor_moved(self, position),
            WindowEvent::MouseInput {state: mouse_state, button: mouse_button, ..} => InputManager::process_mouse_input(self, mouse_state, mouse_button),
            WindowEvent::MouseWheel { delta, .. } => InputManager::process_mouse_wheel(self, *delta), 
            _ => {}
        }
    }
    
    fn update_and_redraw(&mut self) {
        self.update();
        match self.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = self.wgpu_context.window_size();
                self.wgpu_context.resize(size.x as u32, size.y as u32);
            }
            Err(e) => {
                log::error!("Unable to render: {:?}", e);
            }
        }

        self.gpu_profiler.end_frame().unwrap();
        #[cfg(feature = "benchmark")]
        if let Some(profiling_data) = self.gpu_profiler.process_finished_frame(self.wgpu_context.get_queue().get_timestamp_period()) {
            wgpu_profiler::chrometrace::write_chrometrace(std::path::Path::new("benchmark.json"), &profiling_data).unwrap();
        }
    }
    
    fn update(&mut self){
        let dt = self.render_timer.get_delta().as_secs_f32();
        
        {
            let mut encoder = self.wgpu_context.get_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") }
            );
            if self.particles.is_it_time_to_sort(){
                self.particles.sort_by_cell_id(&mut encoder, &mut self.gpu_profiler, self.grid.cell_size());
                self.particles.reset_last_sort_time();                
            }
            self.grid.update(&mut encoder, &mut self.gpu_profiler);
            self.collision_system.solve_collisions(&self.wgpu_context, encoder, &mut self.gpu_profiler);
        }
        
        self.particles.update_positions(dt, &self.wgpu_context, &mut self.gpu_profiler);
        
        // Update renderer with delta time (includes camera update)
        self.renderer.update(dt, &self.wgpu_context, &mut self.gpu_profiler);
    }
    
    fn render(&mut self)  -> anyhow::Result<(), wgpu::SurfaceError>{
        let renderables: Vec<&dyn Renderable> = vec![&self.particles, &self.grid,];
        self.renderer.render(&self.wgpu_context, &renderables, &mut self.gpu_profiler)?;
        Ok(())
    }
}

impl State {
    pub fn get_mouse_position(&self) -> Option<dpi::PhysicalPosition<f64>> {
        self.mouse_position
    }
    pub fn get_wgpu_context(&self) -> &WgpuContext {
        &self.wgpu_context
    }
    
    pub fn get_renderer(&self) -> &Renderer {
        &self.renderer
    }
    
    pub fn get_world_size(&self) -> Vec2 {
        self.world_size
    }
}

impl State {
    pub fn get_mouse_world_position(&self) -> Vec2 {
        self.get_renderer().camera().screen_to_world(&self.get_wgpu_context().window_size(), &Vec2::new(self.get_mouse_position().unwrap().x as f32, self.get_mouse_position().unwrap().y as f32))
    }
    pub fn set_mouse_position(&mut self, position: Option<dpi::PhysicalPosition<f64>>) {
        self.mouse_position = position;
        self.renderer.set_camera_zoom_position(position);
        let world_position = self.get_mouse_world_position();
        self.particles.mouse_move_callback(world_position);
    }
}

impl State {
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool){
        self.renderer.move_camera(key, is_pressed);
    }
    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta){
        self.renderer.zoom_camera(mouse_scroll_delta);
    }
    
    pub fn mouse_click_callback(&mut self, mouse_state: &ElementState, button: &MouseButton){
        if button == &MouseButton::Left {
            let position = self.get_mouse_world_position();
            self.particles.mouse_click_callback(mouse_state, position);
        }
    }

    pub fn add_particles(&mut self){
        let mouse_world_pos = self.get_mouse_world_position();
        let prev_num_particles = self.particles.positions().len();
        self.particles.add_particles(
            &mouse_world_pos,
            &self.wgpu_context
        );
        
        let camera = self.renderer.camera();
        let world_size = self.get_world_size();
        self.grid.refresh_grid(&self.wgpu_context, camera, world_size, &self.particles, prev_num_particles);
        let particles_added = self.particles.positions().len() - prev_num_particles;
        self.collision_system.refresh(&self.wgpu_context, &self.particles, &self.grid, particles_added); 
    }
    
    pub fn toggle_grid_drawing(&mut self){
        self.grid.toggle_grid_drawing();
    }
}

