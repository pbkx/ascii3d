use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct ShadeSample {
    pub intensity: f32,
    pub rgb: Vec3,
}

pub trait Shader {
    fn shade(&self, depth: f32, normal: Vec3, albedo: Vec3) -> ShadeSample;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ShaderId {
    #[default]
    Lambert,
    Unlit,
}

#[derive(Clone, Copy, Debug)]
pub struct LambertShader {
    pub light_dir: Vec3,
    pub ambient: f32,
}

impl Default for LambertShader {
    fn default() -> Self {
        
        
        let light_dir = Vec3::new(0.2, 0.4, 1.0).normalize();
        LambertShader { light_dir, ambient: 0.15 }
    }
}

impl Shader for LambertShader {
    fn shade(&self, _depth: f32, normal: Vec3, albedo: Vec3) -> ShadeSample {
        let n = normal.normalize_or_zero();
        let l = self.light_dir.normalize_or_zero();
        let ndotl = n.dot(l).clamp(0.0, 1.0);
        let t = (self.ambient + (1.0 - self.ambient) * ndotl).clamp(0.0, 1.0);
        ShadeSample { intensity: t, rgb: albedo * t }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnlitShader;

impl Shader for UnlitShader {
    fn shade(&self, _depth: f32, _normal: Vec3, albedo: Vec3) -> ShadeSample {
        ShadeSample { intensity: 1.0, rgb: albedo }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BuiltinShader {
    Lambert(LambertShader),
    Unlit(UnlitShader),
}

impl BuiltinShader {
    pub fn from_id(id: ShaderId) -> Self {
        match id {
            ShaderId::Lambert => BuiltinShader::Lambert(LambertShader::default()),
            ShaderId::Unlit => BuiltinShader::Unlit(UnlitShader),
        }
    }
}

impl Shader for BuiltinShader {
    fn shade(&self, depth: f32, normal: Vec3, albedo: Vec3) -> ShadeSample {
        match *self {
            BuiltinShader::Lambert(s) => s.shade(depth, normal, albedo),
            BuiltinShader::Unlit(s) => s.shade(depth, normal, albedo),
        }
    }
}
