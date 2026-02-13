use glam::Vec3;

use crate::texture::TextureHandle;

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub ns: f32,
    pub ke: Vec3,
    pub alpha: f32,
    pub map_kd_path: Option<String>,
    pub map_kd: Option<TextureHandle>,
}

impl Default for Material {
    fn default() -> Self {
        let kd = Vec3::splat(0.8);
        Self {
            ka: Vec3::ZERO,
            kd,
            ks: Vec3::ZERO,
            ns: 0.0,
            ke: Vec3::ZERO,
            alpha: 1.0,
            map_kd_path: None,
            map_kd: None,
        }
    }
}

impl Material {
    pub fn new(kd: Vec3) -> Self {
        Self { kd, ..Self::default() }
    }

    pub fn with_emissive(mut self, ke: Vec3) -> Self {
        self.ke = ke;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}
