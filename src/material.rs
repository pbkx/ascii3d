use glam::{Mat3, Vec3};

use crate::texture::TextureHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub ns: f32,
    pub ke: Vec3,
    pub alpha: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub map_kd_path: Option<String>,
    pub map_kd: Option<TextureHandle>,
    pub map_kd_uv_transform: Mat3,
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
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: true,
            map_kd_path: None,
            map_kd: None,
            map_kd_uv_transform: Mat3::IDENTITY,
        }
    }
}

impl Material {
    pub fn new(kd: Vec3) -> Self {
        Self {
            kd,
            ..Self::default()
        }
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
