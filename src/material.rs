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
    pub map_kd_texcoord_set: usize,
    pub map_kd_uv_transform: Mat3,
    pub map_normal: Option<TextureHandle>,
    pub map_normal_texcoord_set: usize,
    pub map_normal_uv_transform: Mat3,
    pub map_normal_scale: f32,
    pub map_occlusion: Option<TextureHandle>,
    pub map_occlusion_texcoord_set: usize,
    pub map_occlusion_uv_transform: Mat3,
    pub map_occlusion_strength: f32,
    pub map_emissive: Option<TextureHandle>,
    pub map_emissive_texcoord_set: usize,
    pub map_emissive_uv_transform: Mat3,
    pub map_metallic_roughness: Option<TextureHandle>,
    pub map_metallic_roughness_texcoord_set: usize,
    pub map_metallic_roughness_uv_transform: Mat3,
    pub metallic: f32,
    pub roughness: f32,
    pub pbr_metallic_roughness: bool,
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
            map_kd_texcoord_set: 0,
            map_kd_uv_transform: Mat3::IDENTITY,
            map_normal: None,
            map_normal_texcoord_set: 0,
            map_normal_uv_transform: Mat3::IDENTITY,
            map_normal_scale: 1.0,
            map_occlusion: None,
            map_occlusion_texcoord_set: 0,
            map_occlusion_uv_transform: Mat3::IDENTITY,
            map_occlusion_strength: 1.0,
            map_emissive: None,
            map_emissive_texcoord_set: 0,
            map_emissive_uv_transform: Mat3::IDENTITY,
            map_metallic_roughness: None,
            map_metallic_roughness_texcoord_set: 0,
            map_metallic_roughness_uv_transform: Mat3::IDENTITY,
            metallic: 0.0,
            roughness: 1.0,
            pbr_metallic_roughness: false,
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
