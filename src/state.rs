use std::sync::Arc;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;
use crate::game_data::particle::particle_system::ParticleSystem;
use crate::input_manager::InputManager;
use crate::render_timer::RenderTimer;
use crate::renderer::renderer::Renderer;
use crate::wgpu_context::WgpuContext;

// This will store the state of our game
pub struct State {
    world_size: glam::Vec2,
    wgpu_context: WgpuContext,
    particles: ParticleSystem,
    render_timer: RenderTimer,
    input_manager: InputManager,
    renderer: Renderer,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let world_size = glam::Vec2::new(1920.0, 1080.0);
        let wgpu_context = WgpuContext::new(window).await?;
        let renderer = Renderer::new(&wgpu_context, &world_size).unwrap();

        let particles: ParticleSystem = ParticleSystem::new(&wgpu_context, renderer.camera());

        //renderer.add_pipeline(render_pipeline);
        let render_timer = RenderTimer::new();
        let input_manager = InputManager::new();

        Ok(Self {
            world_size,
            wgpu_context,
            render_timer,
            input_manager,
            renderer,
            particles

        })

    }


    pub fn render_loop(&mut self, event: &WindowEvent, event_loop: &ActiveEventLoop){
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
            }
            _ => {self.input_manager.manage_input(event, event_loop, self.renderer.background_color());}
        }

    }

    fn update(&mut self){
        self.render_timer.get_delta();
        // Recalculate the matrix
        self.renderer.update_camera(&self.wgpu_context);
    }

    fn render(&mut self)  -> Result<(), wgpu::SurfaceError>{
        self.renderer.render(&self.wgpu_context)?;
        Ok(())
    }
}
