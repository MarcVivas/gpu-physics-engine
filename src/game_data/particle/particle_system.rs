use glam::Vec2;

pub struct ParticleSystem {
    vertices: Vec<glam::Vec2>,
    indices: Vec<u32>,
    radiuses: Vec<f32>,
    color: Vec<wgpu::Color>
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            vertices: vec![
                glam::Vec2::new(-0.5, 0.5),  // 0: Top-left
                glam::Vec2::new(0.5, 0.5),   // 1: Top-right
                glam::Vec2::new(0.5, -0.5),  // 2: Bottom-right
                glam::Vec2::new(-0.5, -0.5), // 3: Bottom-left
            ],
            indices: vec![
                0, 1, 2,
                0, 2, 3
            ],
            radiuses: vec![4.0],
            color: vec![wgpu::Color{r: 0.4, g:0.4, b:0.5, a:1.0}],
        }
    }

    pub fn vertices(&self) -> &[glam::Vec2] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}
