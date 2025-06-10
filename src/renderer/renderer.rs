use glam::Vec3;
use wgpu::util::DeviceExt;
use crate::renderer::camera::{Camera};
use crate::wgpu_context::WgpuContext;


const VERTICES: &[crate::state::Vertex] = &[
    crate::state::Vertex { position: [-5.0868241, 7.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    crate::state::Vertex { position: [-0.49513406, 10.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    crate::state::Vertex { position: [-2.21918549, -2.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    crate::state::Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    crate::state::Vertex { position: [4.44147372, 3.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

// Manages multiple render pipelines
pub struct Renderer {
    rendering_pipelines: Vec<wgpu::RenderPipeline>,
    background_color: wgpu::Color,
    camera: Camera,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl Renderer {
    pub fn new(wgpu_context: &WgpuContext) -> Option<Self> {
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

        
        Some(Self {
            rendering_pipelines: Vec::new(),
            background_color: wgpu::Color::BLACK,  
            camera,
            vertex_buffer,
            index_buffer,
            num_indices,       
        })
    }
    
    pub fn add_pipeline(&mut self, pipeline: wgpu::RenderPipeline){
        self.rendering_pipelines.push(pipeline);   
    }

    pub fn render(&self, wgpu_context: &WgpuContext) -> Result<(), wgpu::SurfaceError>{
        wgpu_context.get_window().request_redraw();

        // We can't render unless the window is configured
        if !wgpu_context.is_surface_configured() {
            return Ok(());
        }

        // This is where we render
        let output = wgpu_context.get_surface().get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // We need an encoder to create the actual commands to send to the gpu
        let mut encoder = wgpu_context.get_device().create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Render Encoder"),
        });
        
        for pipeline in self.rendering_pipelines.iter(){
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

                render_pass.set_pipeline(pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.set_bind_group(0, self.camera.binding_group(), &[]);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        wgpu_context.get_queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
    
    pub fn update_camera(&mut self, wgpu_context: &WgpuContext){
        self.camera.build_view_projection_matrix(wgpu_context.window_size().width as f32, wgpu_context.window_size().height as f32);
        wgpu_context.get_queue().write_buffer(
            &self.camera.camera_buffer(),
            0, // offset
            bytemuck::cast_slice(&[*self.camera.get_uniform()])
        );
    }
}
