use glam::Vec3;

/// Simple material model.
///
/// This is intentionally minimal and CPU-friendly: the renderer mostly consumes
/// `kd` and `ke` today, while we keep additional MTL fields around so that
/// later shading work can use them without changing the asset pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Ambient reflectance (MTL: `Ka`).
    pub ka: Vec3,
    /// Diffuse reflectance / base color (MTL: `Kd`).
    pub kd: Vec3,
    /// Specular reflectance (MTL: `Ks`).
    pub ks: Vec3,
    /// Shininess exponent (MTL: `Ns`).
    pub ns: f32,
    /// Emissive color (MTL: `Ke`).
    pub ke: Vec3,
    /// Alpha in [0, 1]. (MTL: `d` or `Tr`.)
    pub alpha: f32,
}

impl Default for Material {
    fn default() -> Self {
        // Defaults are chosen to preserve existing deterministic render hashes:
        // keep the old diffuse default, keep specular disabled by default, and
        // assume opaque materials unless specified.
        let kd = Vec3::splat(0.8);
        Self {
            ka: Vec3::ZERO,
            kd,
            ks: Vec3::ZERO,
            ns: 0.0,
            ke: Vec3::ZERO,
            alpha: 1.0,
        }
    }
}

impl Material {
    /// Create a material with the given diffuse/base color (`Kd`).
    pub fn new(kd: Vec3) -> Self {
        Self { kd, ..Self::default() }
    }

    /// Convenience for setting emissive color (`Ke`).
    pub fn with_emissive(mut self, ke: Vec3) -> Self {
        self.ke = ke;
        self
    }

    /// Convenience for setting alpha.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
}
