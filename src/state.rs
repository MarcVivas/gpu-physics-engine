use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc};
use glam::Vec2;
use winit::dpi;
use winit::event::{KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;
use crate::particles::particle_system::ParticleSystem;
use crate::utils::input_manager::InputManager;
use crate::utils::render_timer::RenderTimer;
use crate::renderer::renderer::Renderer;
use crate::renderer::wgpu_context::WgpuContext;
use crate::grid::grid::Grid;
use crate::utils::gpu_timer::GpuTimer;

// This will store the state of the program
pub struct State {
    world_size: Vec2,
    wgpu_context: WgpuContext,
    render_timer: RenderTimer,
    renderer: Renderer,
    particles: Rc<RefCell<ParticleSystem>>,
    grid: Rc<RefCell<Grid>>,
    mouse_position: Option<dpi::PhysicalPosition<f64>>,
    gpu_timer: GpuTimer,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let world_size = Vec2::new(4920.0, 2080.0);
        let wgpu_context = WgpuContext::new(window).await?;
        let mut renderer = Renderer::new(&wgpu_context, &world_size).unwrap();

        let particles = Rc::new(RefCell::new(ParticleSystem::new(&wgpu_context, renderer.camera(), world_size)));
        let grid =  Rc::new(RefCell::new(Grid::new(&wgpu_context, renderer.camera(), world_size, particles.borrow().get_max_radius(), particles.clone())));

        renderer.add_renderable(particles.clone());
        renderer.add_renderable(grid.clone());

        let render_timer = RenderTimer::new();

        let mouse_position = None;
        
        let gpu_timer = GpuTimer::new(wgpu_context.get_device(), wgpu_context.get_queue(), 10);
        
        Ok(Self {
            world_size,
            wgpu_context,
            render_timer,
            particles,
            renderer,
            mouse_position,
            grid,
            gpu_timer,
        })

    }

    
    
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
                self.wgpu_context.resize(size.width, size.height);
            }
            Err(e) => {
                log::error!("Unable to render: {:?}", e);
            }
        }
    }
    
    #[cfg(feature = "benchmark")]
    fn update(&mut self){
        let dt = self.render_timer.get_delta().as_secs_f32();
        // Update renderer with delta time (includes camera update)
        self.renderer.update(dt, &self.wgpu_context, &self.world_size, &mut self.gpu_timer);
    }

    #[cfg(not(feature = "benchmark"))]
    fn update(&mut self){
        let dt = self.render_timer.get_delta().as_secs_f32();
        // Update renderer with delta time (includes camera update)
        self.renderer.update(dt, &self.wgpu_context, &self.world_size);
    }

    fn render(&mut self)  -> Result<(), wgpu::SurfaceError>{
        self.renderer.render(&self.wgpu_context)?;
        Ok(())
    }
}

impl State {
    pub fn get_particles(&self) -> Rc<RefCell<ParticleSystem>> {
        self.particles.clone()
    }
    pub fn get_grid(&self) -> Rc<RefCell<Grid>> {
        self.grid.clone()
    }
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
    pub fn set_mouse_position(&mut self, position: Option<dpi::PhysicalPosition<f64>>) {
        self.mouse_position = position;
        self.renderer.set_camera_zoom_position(position);
    }
}

impl State {
    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool){
        self.renderer.move_camera(key, is_pressed);
    }
    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta){
        self.renderer.zoom_camera(mouse_scroll_delta);
    }
}

#[cfg(feature = "benchmark")]
impl Drop for State {
    fn drop(&mut self) {
        self.gpu_timer.report(&self.wgpu_context);
    }
}
