use glam::Vec2;
use rand::Rng;

pub struct ParticleSystem {
    vertices: Vec<glam::Vec2>,
    indices: Vec<u32>,
    instances: Vec<glam::Vec2>,
    radiuses: Vec<f32>,
    colors: Vec<wgpu::Color>
}

impl ParticleSystem {
    pub fn new() -> Self {
        const NUM_PARTICLES: usize = 2_000_000;
        const WORLD_WIDTH: f32 = 1920.0;
        const WORLD_HEIGHT: f32 = 1080.0;

        let mut rng = rand::thread_rng();

        let mut instances = Vec::with_capacity(NUM_PARTICLES);
        let mut radiuses = Vec::with_capacity(NUM_PARTICLES);

        for _ in 0..NUM_PARTICLES {
            let x = rng.random_range(-WORLD_WIDTH / 2.0..WORLD_WIDTH / 2.0);
            let y = rng.random_range(-WORLD_HEIGHT / 2.0..WORLD_HEIGHT / 2.0);
            instances.push(Vec2::new(x, y));

            let radius = rng.random_range(1.0..4.0);
            radiuses.push(radius);
        }

        Self {
            vertices: vec![
                glam::Vec2::new(-0.5, 0.5),
                glam::Vec2::new(0.5, 0.5),
                glam::Vec2::new(0.5, -0.5),
                glam::Vec2::new(-0.5, -0.5),
            ],
            indices: vec![
                0, 3, 2,
                2, 1, 0
            ],
            instances,
            radiuses,
            colors: vec![wgpu::Color { r: 0.4, g: 0.4, b: 0.5, a: 1.0 }],
        }
    }

    pub fn vertices(&self) -> &[glam::Vec2] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn instances(&self) -> &[glam::Vec2] {
        &self.instances
    }

    pub fn radiuses(&self) -> &[f32] {
        &self.radiuses
    }

    pub fn color(&self) -> &[wgpu::Color] {
        &self.colors
    }
}
