use std::sync::Arc;
use wgpu::Adapter;
use winit::window::Window;

use crate::renderer::surface_manager::SurfaceManager;

pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_manager: Option<SurfaceManager>,
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
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await?;

        let surface_manager: Option<SurfaceManager> = Some(SurfaceManager::new(window, &instance, &adapter));

      
        

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor{
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: WgpuContext::get_limits(&adapter),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }).await?;



        Ok(Self {
            device,
            queue,
            surface_manager
        })
    }
    
    fn get_limits(adapter: &Adapter) -> wgpu::Limits {
        let limits;
        if cfg!(target_arch = "wasm32") {
            // When on web, request the browser's supported limits
            limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        } else {
            // For native, use the adapter's reported limits
            limits = adapter.limits();
        }
        
        limits
    }
    pub async fn new_for_test() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // <-- NO SURFACE
                force_fallback_adapter: false,
            })
            .await?;
            

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Test Device"),
                    required_features: wgpu::Features::empty(), // Add features if your shaders need them
                    required_limits: WgpuContext::get_limits(&adapter),
                    ..Default::default()
                },
            )
            .await?;

        Ok(Self {
            device,
            queue,
            surface_manager: None, // <-- NO SURFACE MANAGER
        })
    }
    
    pub fn window_size(&self) -> winit::dpi::PhysicalSize<u32> {
        if self.surface_manager.is_none() {
            return winit::dpi::PhysicalSize::new(0, 0);
        }
        self.surface_manager.as_ref().expect("No surface in this context").window_size()
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_manager.as_mut().expect("No surface in this context").resize(width, height, &self.device);
    }
    
    pub fn get_window(&self) -> &Arc<Window> {
        self.surface_manager.as_ref().expect("No surface in this context").get_window()
    }
    
    pub fn get_surface(&self) -> &wgpu::Surface<'static> {
        self.surface_manager.as_ref().expect("No surface manager in this context").get_surface()
    }
    pub fn is_surface_configured(&self) -> bool {
        self.surface_manager.as_ref().expect("No surface in this context").is_surface_configured()
    }
    
    pub fn get_device(&self) -> &wgpu::Device {
        &self.device
    }
    
    pub fn get_queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    
    pub fn get_surface_config(&self) -> &wgpu::SurfaceConfiguration{
        &self.surface_manager.as_ref().expect("No surface in this context").get_config()
    }
}