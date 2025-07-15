use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc};
use glam::Vec2;
use winit::dpi;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;
use crate::game_data::line::lines::Lines;
use crate::game_data::particle::particle_system::ParticleSystem;
use crate::game::input_manager::InputManager;
use crate::renderer::render_timer::RenderTimer;
use crate::renderer::renderer::Renderer;
use crate::renderer::wgpu_context::WgpuContext;
use crate::game::grid::grid::Grid;
// This will store the state of the game
pub struct State {
    world_size: Vec2,
    wgpu_context: WgpuContext,
    render_timer: RenderTimer,
    input_manager: InputManager,
    renderer: Renderer,
    particles: Rc<RefCell<ParticleSystem>>,
    grid: Rc<RefCell<Grid>>,
    mouse_position: Option<dpi::PhysicalPosition<f64>>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let world_size = glam::Vec2::new(1920.0, 1080.0);
        let wgpu_context = WgpuContext::new(window).await?;
        let mut renderer = Renderer::new(&wgpu_context, &world_size).unwrap();

        let particles = Rc::new(RefCell::new(ParticleSystem::new(&wgpu_context, renderer.camera())));
        let grid =  Rc::new(RefCell::new(Grid::new(&wgpu_context, renderer.camera(), world_size, particles.borrow().get_max_radius(), particles.clone())));

        renderer.add_renderable(particles.clone());
        renderer.add_renderable(grid.clone());

        let render_timer = RenderTimer::new();
        let input_manager = InputManager::new();

        let mouse_position = None;

        Ok(Self {
            world_size,
            wgpu_context,
            render_timer,
            input_manager,
            particles,
            renderer,
            mouse_position,
            grid,
        })

    }


    pub fn render_loop(&mut self, event: &WindowEvent, event_loop: &ActiveEventLoop){
        self.renderer.process_events(event);

        match event {
            WindowEvent::Resized(size ) => self.wgpu_context.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
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
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                match (code, key_state.is_pressed()) {
                    (KeyCode::Escape, true) => event_loop.exit(),
                    (KeyCode::KeyP, true) => {
                        let prev_num_particles = self.particles.borrow().instances().len();
                        self.particles.borrow_mut().add_particles(
                            &self.renderer.camera().screen_to_world(Vec2::new(self.mouse_position.unwrap().x as f32, self.mouse_position.unwrap().y as f32)),
                            &self.wgpu_context
                        );
                        let particles_added = self.particles.borrow().instances().len() - prev_num_particles;
                        self.grid.borrow_mut().reset_grid(&self.wgpu_context, self.renderer.camera(), self.world_size, self.particles.borrow().get_max_radius(), particles_added);
                    },
                    (KeyCode::KeyG, true) => {
                        self.grid.borrow_mut().render_grid();
                    }

                    _ => {}
                }

            },
            WindowEvent::CursorMoved { position, .. } => {
                // Update the stored mouse position
                self.mouse_position = Some(*position);
            },
            _ => {
                // Handle global input through InputManager (no renderer needed)
                self.input_manager.manage_input(event, event_loop);
            }
        }

    }

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
