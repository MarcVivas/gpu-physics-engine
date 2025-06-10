use std::time::{Instant, Duration};

pub struct RenderTimer {
    last_render_time: Instant,
    total_render_time: Duration,
    frame_count: u64,
}
impl RenderTimer {
    pub fn new() -> Self {
        Self {
            last_render_time: Instant::now(),
            total_render_time: Duration::new(0, 0),
            frame_count: 0u64,
        }
    }
    
    pub fn get_delta(&mut self) -> Duration {
        let now = Instant::now();
        let delta_time = now - self.last_render_time;
        self.last_render_time = now;
        self.total_render_time += delta_time;
        self.frame_count += 1;
        delta_time
    }
    
    fn get_average_render_time(&self) -> f64{
        self.total_render_time.as_secs_f64() / self.frame_count as f64 * 1000.0f64
    }
}

// Destructor equivalent from C++
impl Drop for RenderTimer {
    fn drop(&mut self) {
        println!("Average render time: {:?} ms", self.get_average_render_time());
        println!("Frame count: {}", self.frame_count);
        println!("Total render time: {:?} s", self.total_render_time.as_secs_f64());
    }   
}