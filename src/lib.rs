mod camera;
use camera::{Camera, CameraUniform};

mod render_timer;
use render_timer::{RenderTimer};

mod input_manager;
use input_manager::{InputManager};

mod surface_manager;
use surface_manager::{SurfaceManager};

mod wgpu_context;
use wgpu_context::{WgpuContext};

use std::sync::Arc;
use glam::Vec3;
use winit::{
    event::*,
    event_loop::{EventLoop, ActiveEventLoop},
    keyboard::{KeyCode, PhysicalKey},
    application::ApplicationHandler,
    window::Window
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

// This will store the state of our game
pub struct State {
    wgpu_context: WgpuContext,
    background_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    render_timer: RenderTimer,
    input_manager: InputManager,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}
const VERTICES: &[Vertex] = &[
    Vertex { position: [-5.0868241, 7.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    Vertex { position: [-0.49513406, 10.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    Vertex { position: [-2.21918549, -2.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    Vertex { position: [4.44147372, 3.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];
impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let wgpu_context = WgpuContext::new(window).await?;
      



        let vertex_buffer = wgpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = wgpu_context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let num_vertices = VERTICES.len() as u32;
        let num_indices = INDICES.len() as u32;

        // 1. Calculate the bounding box of the vertices
        let (min_x, max_x, min_y, max_y) = VERTICES.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y), vertex| {
                (
                    min_x.min(vertex.position[0]),
                    max_x.max(vertex.position[0]),
                    min_y.min(vertex.position[1]),
                    max_y.max(vertex.position[1]),
                )
            },
        );

        // 2. Calculate the center of the bounding box
        let center = Vec3::new(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            0.0
        );

        // 3. Calculate the required zoom to fit the object on screen
        let world_width = max_x - min_x;
        let world_height = max_y - min_y;
        
        let window_size = wgpu_context.window_size();

        let screen_width = window_size.width as f32;
        let screen_height = window_size.height as f32;

        // Calculate zoom based on width and height, and pick the smaller one to ensure it all fits
        let zoom_x = screen_width / world_width;
        let zoom_y = screen_height / world_height;
        let zoom = zoom_x.min(zoom_y) * 0.9; // Use 90% of the screen for some padding

        // 4. Create the camera with the calculated values
        let camera = Camera::new(center, zoom);

        // 1. Create the Camera controller and the initial uniform data
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, window_size.width as f32, window_size.height as f32);

        // 2. Create the wgpu::Buffer
        let camera_buffer = wgpu_context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                // COPY_DST allows us to update the buffer later.
                // UNIFORM tells wgpu we'll use it in a bind group.
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        // 3. Create the Bind Group Layout (the "template")
        let camera_bind_group_layout = wgpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // This binding number must match the shader e.g. `@binding(0)`
                    binding: 0,
                    // We only need the camera matrix in the vertex shader.
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("Camera Bind Group Layout"),
        });

        // 4. Create the Bind Group (the "instance")
        let camera_bind_group = wgpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("Camera Bind Group"),
        });


        let shader = wgpu_context.device.create_shader_module(wgpu::include_wgsl!("shaders/renderShaders/shader.wgsl"));
        let render_pipeline_layout = wgpu_context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = wgpu_context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label: Some("Render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState{
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState{
                    format: wgpu_context.surface_manager.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default()
            }),

            primitive: wgpu::PrimitiveState{
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,

            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        let render_timer = RenderTimer::new();
        
        let input_manager = InputManager::new();

        Ok(Self {
            wgpu_context,
            background_color: wgpu::Color {r:0.3, g:0.1, b:0.3, a:1.0},
            render_pipeline,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
            camera,
            camera_buffer,
            camera_uniform,
            camera_bind_group,
            render_timer,
            input_manager,
            
        })
        
    }
    

    
   

   
    
    fn input(&mut self, event: &WindowEvent, event_loop: &ActiveEventLoop){
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
            _ => {self.input_manager.manage_input(event, event_loop, &mut self.background_color);}
        }
        
    }
    
    fn update(&mut self){
        let delta = self.render_timer.get_delta();
        println!("{:?}", delta.as_secs_f32() * 1000.0);
        // Recalculate the matrix
        self.camera_uniform.update_view_proj(&self.camera, self.wgpu_context.window_size().width as f32, self.wgpu_context.window_size().height as f32);
        self.wgpu_context.queue.write_buffer(
            &self.camera_buffer,
            0, // offset
            bytemuck::cast_slice(&[self.camera_uniform])
        );
    }
    
    fn render(&mut self)  -> Result<(), wgpu::SurfaceError>{
        self.wgpu_context.surface_manager.window.request_redraw();
        
        // We can't render unless the window is configured
        if !self.wgpu_context.surface_manager.is_surface_configured {
            return Ok(());
        }
        
        // This is where we render
        let output = self.wgpu_context.surface_manager.surface.get_current_texture()?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // We need an encoder to create the actual commands to send to the gpu
        let mut encoder = self.wgpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Render Encoder"),
        });
        
        // Use encoder to create a RenderPass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment{
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color),
                            store: wgpu::StoreOp::Store,
                        }
                    })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
        
        
        self.wgpu_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
}


pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();


        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy
                        .send_event(
                            State::new(window)
                                .await
                                .expect("Unable to create canvas!!!")
                        )
                        .is_ok())
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, event_loop: &ActiveEventLoop, mut event: State) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };
        
        state.input(&event, &event_loop);
        
    }
}


pub fn run() -> anyhow::Result<()>{
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }
    
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    
    event_loop.run_app(&mut app)?;
    
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
