use std::time::{Instant, Duration};

pub struct RenderTimer {
    last_render_time: Instant,
}

impl RenderTimer {
    pub fn new() -> Self {
        Self {
            last_render_time: Instant::now(),
        }
    }
    
    pub fn get_delta(&mut self) -> Duration {
        let now = Instant::now();
        let delta_time = now - self.last_render_time;
        self.last_render_time = now;
        delta_time
    }
}