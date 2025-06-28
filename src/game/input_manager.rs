use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::game::state::State;

pub struct InputManager {

}

impl InputManager {
    pub fn new() -> InputManager {
        InputManager {}
    }

    pub fn manage_input(&self, event: &WindowEvent, event_loop: &ActiveEventLoop) {
        match event {
            WindowEvent::CloseRequested => {event_loop.exit();}

            WindowEvent::KeyboardInput {
                event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => {
                // Handle global input only
                self.handle_key(event_loop, code, key_state.is_pressed());
            },
            WindowEvent::MouseWheel { delta, .. } => {
                // Mouse wheel is handled directly by renderer in State
            }
            _ => {}
        }
    }

    fn handle_key(&self, event_loop: &ActiveEventLoop, code: &KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }
}
