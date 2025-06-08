use std::sync::Arc;
use wgpu::Adapter;
use winit::dpi;
use winit::window::Window;

pub struct SurfaceManager {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub is_surface_configured: bool,
    pub config: wgpu::SurfaceConfiguration,

}

impl SurfaceManager {
    pub fn new(window: Arc<Window>, instance: &wgpu::Instance, adapter: &Adapter) -> Self {
        let surface = instance.create_surface(window.clone()).unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        Self { window, surface, is_surface_configured: false, config }
    }
    
    pub fn window_size(&self) -> dpi::PhysicalSize<u32> {
        self.window.inner_size() 
    }

    pub fn resize(&mut self, _width: u32, _height: u32, device: &wgpu::Device){
        if _width > 0 && _height > 0 {
            self.config.width = _width;
            self.config.height = _height;
            self.surface.configure(&device, &self.config);
            self.is_surface_configured = true;
        }
    }
}