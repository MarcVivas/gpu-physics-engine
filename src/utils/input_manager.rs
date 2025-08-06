use glam::Vec2;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::state::State;

pub struct InputManager {}

impl InputManager {
    
    /// Manages keyboard inputs from the user
    pub fn process_keyboard_input(state: &mut State, event_loop: &ActiveEventLoop, code: &KeyCode, key_state: &ElementState) {
        match (code, key_state.is_pressed()) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyP, true) => {
                let prev_num_particles = state.get_particles().borrow().positions().len();
                state.get_particles().borrow_mut().add_particles(
                    &state.get_renderer().camera().screen_to_world(&state.get_wgpu_context().window_size(), &Vec2::new(state.get_mouse_position().unwrap().x as f32, state.get_mouse_position().unwrap().y as f32)),
                    state.get_wgpu_context()
                );
                let particles_added = state.get_particles().borrow().positions().len() - prev_num_particles;
                state.get_grid().borrow_mut().refresh_grid(state.get_wgpu_context(), state.get_renderer().camera(), state.get_world_size(), state.get_particles().borrow().get_max_radius(), particles_added);
            },
            (KeyCode::KeyG, true) => {
                state.get_grid().borrow_mut().render_grid();
            },
            (KeyCode::KeyW | KeyCode::ArrowUp, true) => {
                state.move_camera(KeyCode::KeyW, true);
            },
            (KeyCode::KeyW | KeyCode::ArrowUp, false) => {
                state.move_camera(KeyCode::KeyW, false);
            },
            (KeyCode::KeyS | KeyCode::ArrowDown, true) => {
                state.move_camera(KeyCode::KeyS, true);
            },
            (KeyCode::KeyS | KeyCode::ArrowDown, false) => {
                state.move_camera(KeyCode::KeyS, false);
            },
            (KeyCode::KeyA | KeyCode::ArrowLeft, true) => {
                state.move_camera(KeyCode::KeyA, true);
            },
            (KeyCode::KeyA | KeyCode::ArrowLeft, false) => {
                state.move_camera(KeyCode::KeyA, false);
            },
            (KeyCode::KeyD | KeyCode::ArrowRight, true) => {
                state.move_camera(KeyCode::KeyD, true);
            },
            (KeyCode::KeyD | KeyCode::ArrowRight, false) => {
                state.move_camera(KeyCode::KeyD, false);
            },
            _ => {}
        }
    }
    
    /// Manages mouse movement
    pub fn process_cursor_moved(state: &mut State, position: &PhysicalPosition<f64>){
        // Update the stored mouse position
        state.set_mouse_position(Some(*position));
    }
    
    /// Manages mouse button inputs from the user
    pub fn process_mouse_input(state: &mut State, mouse_state: &ElementState, button: &MouseButton){
        state.mouse_click_callback(mouse_state, button);
    }
    
    /// Manages mouse wheel inputs from the user
    pub fn process_mouse_wheel(state: &mut State, delta: MouseScrollDelta){
        state.zoom_camera(delta);
    }
}
