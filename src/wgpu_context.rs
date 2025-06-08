use std::sync::Arc;
use winit::window::Window;
use crate::surface_manager::SurfaceManager;

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_manager: SurfaceManager,
}

impl WgpuContext {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {


        // The instance is a handle to our GPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;


        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions{
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await?;

        let surface_manager: SurfaceManager = SurfaceManager::new(window, &instance, &adapter);
                
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor{
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }).await?;



        Ok(Self {
            device,
            queue,
            surface_manager
        })
    }
    
    pub fn window_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.surface_manager.window_size()
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_manager.resize(width, height, &self.device);
    }
}