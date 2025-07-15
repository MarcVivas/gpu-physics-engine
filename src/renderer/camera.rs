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
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
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
                camera_controller: CameraController::new(250.0, 0.1),
                camera_uniform,
                camera_buffer,
                camera_bind_group,
                camera_bind_group_layout,
            }
        }

        pub fn build_view_projection_matrix(&mut self, screen_width: f32, screen_height: f32) -> Mat4 {
            // Update screen size in controller for zoom-to-cursor calculations
            self.camera_controller.screen_size = glam::Vec2::new(screen_width, screen_height);

            // Create view matrix - invert camera position to move the world opposite to camera
            let view = Mat4::from_translation(-self.position);

            // Create symmetric orthographic projection centered around origin
            let half_width = screen_width / (2.0 * self.zoom);
            let half_height = screen_height / (2.0 * self.zoom);

            let projection = Mat4::orthographic_rh(
                -half_width,  // left
                half_width,   // right
                -half_height, // bottom
                half_height,  // top
                -1.0, // near
                1.0,  // far
            );

            let view_proj = projection * view;
            self.camera_uniform.update_view_proj(&view_proj);
            view_proj
        }

        pub fn process_events(&mut self, event: &winit::event::WindowEvent) -> bool {
            self.camera_controller.process_events(event)
        }

        pub fn update(&mut self, dt: f32) {
            let move_speed = self.camera_controller.speed * dt / self.zoom;

            if self.camera_controller.is_up_pressed { self.position.y += move_speed; }
            if self.camera_controller.is_down_pressed { self.position.y -= move_speed; }
            if self.camera_controller.is_right_pressed { self.position.x += move_speed; }
            if self.camera_controller.is_left_pressed { self.position.x -= move_speed; }

            if self.camera_controller.scroll_delta != 0.0 {
                // 1. Get the world coordinates of the mouse before zooming
                let mouse_world_pos_before_zoom = self.screen_to_world(self.camera_controller.mouse_position);

                // 2. Calculate the new zoom level
                let zoom_factor = 1.0 + (self.camera_controller.scroll_delta * self.camera_controller.zoom_sensitivity);
                self.zoom *= zoom_factor;
                // Clamp the zoom to prevent it from becoming too small or large
                self.zoom = self.zoom.clamp(0.1, 100.0);

                // 3. Get the world coordinates of the mouse after zooming
                let mouse_world_pos_after_zoom = self.screen_to_world(self.camera_controller.mouse_position);

                // 4. Calculate the difference (how much the world shifted under the cursor)
                let world_delta = mouse_world_pos_before_zoom - mouse_world_pos_after_zoom;

                // 5. Adjust the camera position to counteract the shift
                self.position += glam::Vec3::new(world_delta.x, world_delta.y, 0.0);

                // 6. Reset the scroll delta
                self.camera_controller.scroll_delta = 0.0;
            }
        }

        pub fn screen_to_world(&self, screen_pos: glam::Vec2) -> glam::Vec2 {
            let screen_size = self.camera_controller.screen_size;

            // Convert screen coordinates to normalized device coordinates (-1 to 1)
            let ndc_x = (screen_pos.x / screen_size.x) * 2.0 - 1.0;
            let ndc_y = 1.0 - (screen_pos.y / screen_size.y) * 2.0; // Flip Y axis

            // Convert NDC to world coordinates
            let half_width = screen_size.x / (2.0 * self.zoom);
            let half_height = screen_size.y / (2.0 * self.zoom);

            let world_x = self.position.x + ndc_x * half_width;
            let world_y = self.position.y + ndc_y * half_height;

            glam::Vec2::new(world_x, world_y)
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

    use winit::event::{WindowEvent, KeyEvent, MouseScrollDelta};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use crate::renderer::wgpu_context::WgpuContext;

    #[derive(Debug)]
    pub struct CameraController {
        is_up_pressed: bool,
        is_down_pressed: bool,
        is_left_pressed: bool,
        is_right_pressed: bool,
        speed: f32,
        zoom_sensitivity: f32,
        scroll_delta: f32, // New field to store scroll amount
        mouse_position: glam::Vec2, // Track mouse position for zoom-to-cursor
        screen_size: glam::Vec2, // Track screen size for coordinate conversion
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
                mouse_position: glam::Vec2::ZERO,
                screen_size: glam::Vec2::new(800.0, 600.0), // Default size
            }
        }

        fn process_events(&mut self, event: &WindowEvent) -> bool {
            match event {
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                    let is_pressed = *state == winit::event::ElementState::Pressed;
                    match physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.is_up_pressed = is_pressed;
                            true
                        }
                        PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.is_down_pressed = is_pressed;
                            true
                        }
                        PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.is_left_pressed = is_pressed;
                            true
                        }
                        PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.is_right_pressed = is_pressed;
                            true
                        }
                        _ => false,
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    self.scroll_delta += match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    true
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.mouse_position = glam::Vec2::new(position.x as f32, position.y as f32);
                    false // Don't consume this event
                }
                _ => false,
            }
        }


    }
