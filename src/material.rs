use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub albedo: Vec3,
    pub emissive: Vec3,
    pub alpha: f32,
}

impl Material {
    pub fn new(albedo: Vec3) -> Self {
        Self {
            albedo,
            ..Self::default()
        }
    }

    pub fn with_emissive(mut self, emissive: Vec3) -> Self {
        self.emissive = emissive;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: Vec3::splat(0.8),
            emissive: Vec3::ZERO,
            alpha: 1.0,
        }
    }
}
