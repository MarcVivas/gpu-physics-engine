use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub zoom: f32,
    camera_controller: CameraController
}

impl Camera {
    pub fn new(position: Vec3, zoom: f32) -> Self {
        Self { position, zoom, camera_controller: CameraController::new(10.0, 0.1) }
    }

    pub fn build_view_projection_matrix(&self, screen_width: f32, screen_height: f32) -> Mat4 {
        let view = Mat4::from_translation(self.position);

        let projection = Mat4::orthographic_rh(
            -screen_width / 2.0 / self.zoom,    // left
            screen_width / 2.0 / self.zoom,     // right
            -screen_height / 2.0 / self.zoom,   // bottom
            screen_height / 2.0 / self.zoom,    // top
            -1.0, // near
            1.0, // far
        );

        projection * view
    }
    
    pub fn process_events(&mut self, event: &winit::event::WindowEvent) {
        self.camera_controller.process_events(event);
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

    pub fn update_view_proj(&mut self, camera: &Camera, screen_width: f32, screen_height: f32) {
        self.view_proj = camera.build_view_projection_matrix(screen_width, screen_height).to_cols_array_2d();
    }
}

use winit::event::{ElementState, WindowEvent, MouseScrollDelta};
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
        use glam::Vec2;

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
