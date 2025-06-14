    use glam::{Mat4, Vec3};
    use wgpu::util::DeviceExt;

    pub struct Camera {
        pub position: Vec3,
        pub zoom: f32,
        camera_controller: CameraController,
        camera_uniform: CameraUniform,
        camera_buffer: wgpu::Buffer,
        camera_bind_group: wgpu::BindGroup,
        camera_bind_group_layout: wgpu::BindGroupLayout,
    }

    impl Camera {
        pub fn new(world_size: &glam::Vec2, wgpu_context: &WgpuContext) -> Self {


            let min_x = 0.0;
            let max_x = world_size.x;
            let min_y = 0.0;
            let max_y = world_size.y;

            let position = Vec3::new(
                (min_x + max_x) / 4.0,
                (min_y + max_y) / 4.0,
                0.0
            );

            // 3. Calculate the required zoom to fit the object on screen
            let world_width = world_size.x;
            let world_height = world_size.y;

            let window_size = wgpu_context.window_size();

            let screen_width = window_size.width as f32;
            let screen_height = window_size.height as f32;

            // Calculate zoom based on width and height, and pick the smaller one to ensure it all fits
            let zoom_x = screen_width / world_width;
            let zoom_y = screen_height / world_height;
            let zoom = zoom_x.min(zoom_y) * 0.9; // Use 90% of the screen for some padding

            // 1. Create the Camera controller and the initial uniform data
            let camera_uniform = CameraUniform::new();

            // 2. Create the wgpu::Buffer
            let camera_buffer = wgpu_context.get_device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Buffer"),
                    contents: bytemuck::cast_slice(&[camera_uniform]),
                    // COPY_DST allows us to update the buffer later.
                    // UNIFORM tells wgpu we'll use it in a bind group.
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }
            );

            // 3. Create the Bind Group Layout (the "template")
            let camera_bind_group_layout = wgpu_context.get_device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            let camera_bind_group = wgpu_context.get_device().create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }
                ],
                label: Some("Camera Bind Group"),
            });
            Self {
                position,
                zoom,
                camera_controller: CameraController::new(10.0, 0.1),
                camera_uniform,
                camera_buffer,
                camera_bind_group,
                camera_bind_group_layout,
            }
        }

        pub fn build_view_projection_matrix(&mut self, screen_width: f32, screen_height: f32) -> Mat4 {
            let view = Mat4::from_translation(self.position);

            let projection = Mat4::orthographic_rh(
                screen_width / 2.0 / self.zoom,    // left
                screen_width / self.zoom,     // right
                screen_height / 2.0 / self.zoom,   // bottom
                screen_height / self.zoom,    // top
                -1.0, // near
                1.0, // far
            );

            let view_proj = projection * view;
            self.camera_uniform.update_view_proj(&view_proj);
            view_proj
        }

        pub fn process_events(&mut self, event: &winit::event::WindowEvent) {
            self.camera_controller.process_events(event);
        }

        pub fn binding_group(&self) -> &wgpu::BindGroup {
            &self.camera_bind_group
        }

        pub fn get_uniform(&self) -> &CameraUniform {
            &self.camera_uniform
        }

        pub fn camera_buffer(&self) -> &wgpu::Buffer {
            &self.camera_buffer
        }

        pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
            &self.camera_bind_group_layout

        }

    }

    #[repr(C)]
    // `bytemuck` is used to easily convert the struct to a byte slice for the GPU buffer.
    #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct CameraUniform {
        // We can't use glam::Mat4 directly, so we use a 4x4 array of f32s.
        view_proj: [[f32; 4]; 4],
    }

    impl CameraUniform {
        pub fn new() -> Self {
            Self {
                // Initialize with the identity matrix.
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            }
        }

        pub fn update_view_proj(&mut self, m: &Mat4) {
            self.view_proj = m.to_cols_array_2d();
        }
    }

    use winit::event::{WindowEvent};
    use crate::wgpu_context::WgpuContext;

    #[derive(Debug)]
    pub struct CameraController {
        is_up_pressed: bool,
        is_down_pressed: bool,
        is_left_pressed: bool,
        is_right_pressed: bool,
        speed: f32,
        zoom_sensitivity: f32,
        scroll_delta: f32, // New field to store scroll amount
    }

    impl CameraController {
        fn new(speed: f32, zoom_sensitivity: f32) -> Self {
            Self {
                is_up_pressed: false,
                is_down_pressed: false,
                is_left_pressed: false,
                is_right_pressed: false,
                speed,
                zoom_sensitivity,
                scroll_delta: 0.0,
            }
        }

        fn process_events(&mut self, event: &WindowEvent) -> bool {
            return true;
        }

        fn update_camera(&mut self, camera: &mut super::camera::Camera) {
            let move_speed = self.speed / camera.zoom;

            if self.is_up_pressed { camera.position.y += move_speed; }
            if self.is_down_pressed { camera.position.y -= move_speed; }
            if self.is_right_pressed { camera.position.x += move_speed; }
            if self.is_left_pressed { camera.position.x -= move_speed; }

            if self.scroll_delta != 0.0 {
                let zoom_factor = 1.0 - (self.scroll_delta * self.zoom_sensitivity);
                camera.zoom *= zoom_factor;
                camera.zoom = camera.zoom.max(0.01);
            }

            // Reset scroll delta after applying it
            self.scroll_delta = 0.0;
        }
    }
