use std::sync::Arc;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;
use crate::input_manager::InputManager;
use crate::render_timer::RenderTimer;
use crate::renderer::camera::Camera;
use crate::renderer::renderer::Renderer;
use crate::wgpu_context::WgpuContext;

// This will store the state of our game
pub struct State {
    wgpu_context: WgpuContext,
    background_color: wgpu::Color,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    render_timer: RenderTimer,
    input_manager: InputManager,
    renderer: Renderer,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
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
        let mut renderer = Renderer::new(&wgpu_context).unwrap();

        let vertex_buffer = wgpu_context.get_device().create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = wgpu_context.get_device().create_buffer_init(
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
        let camera = Camera::new(center, zoom, &wgpu_context);


        let shader = wgpu_context.get_device().create_shader_module(wgpu::include_wgsl!("shaders/renderShaders/shader.wgsl"));
        let render_pipeline_layout = wgpu_context.get_device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera.camera_bind_group_layout()],
            push_constant_ranges: &[],
        });

        let render_pipeline = wgpu_context.get_device().create_render_pipeline(&wgpu::RenderPipelineDescriptor{
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
                    format: wgpu_context.get_surface_config().format,
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

        renderer.add_pipeline(render_pipeline);
        let render_timer = RenderTimer::new();
        let input_manager = InputManager::new();

        Ok(Self {
            wgpu_context,
            background_color: wgpu::Color {r:0.3, g:0.1, b:0.3, a:1.0},
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
            render_timer,
            input_manager,
            renderer

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
            _ => {self.input_manager.manage_input(event, event_loop, &mut self.background_color);}
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