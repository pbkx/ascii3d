use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct ShadeSample {
    pub intensity: f32,
    pub rgb: Vec3,
}

pub trait Shader {
    fn shade_scalar(&self, depth: f32, normal: Vec3, kd: Vec3, ks: Vec3, ns: f32, ke: Vec3) -> f32;
    fn shade_rgb(&self, depth: f32, normal: Vec3, kd: Vec3, ks: Vec3, ns: f32, ke: Vec3) -> Vec3;

    fn shade(
        &self,
        depth: f32,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> ShadeSample {
        ShadeSample {
            intensity: self.shade_scalar(depth, normal, kd, ks, ns, ke),
            rgb: self.shade_rgb(depth, normal, kd, ks, ns, ke),
        }
    }
}

fn luma(rgb: Vec3) -> f32 {
    // Rec.709 luma coefficients
    0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z
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
        LambertShader {
            light_dir,
            ambient: 0.15,
        }
    }
}

impl Shader for LambertShader {
    fn shade_scalar(
        &self,
        _depth: f32,
        normal: Vec3,
        _kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> f32 {
        let n = normal.normalize_or_zero();
        let l = self.light_dir.normalize_or_zero();
        let ndotl = n.dot(l).clamp(0.0, 1.0);

        // Keep the historic behavior for the base "lighting intensity" so existing
        // golden hashes remain stable when Ks/Ke are zero.
        let base = (self.ambient + (1.0 - self.ambient) * ndotl).clamp(0.0, 1.0);

        let mut t = base;

        // Specular highlight (Blinn–Phong) from Ks/Ns.
        if ndotl > 0.0 && ns > 0.0 && ks.length_squared() > 0.0 {
            let v = Vec3::new(0.0, 0.0, 1.0);
            let h = (l + v).normalize_or_zero();
            let ndoth = n.dot(h).clamp(0.0, 1.0);
            let shin = ns.max(1.0);
            let spec = ndoth.powf(shin);
            t += luma(ks) * spec;
        }

        // Emissive adds directly.
        if ke.length_squared() > 0.0 {
            t += luma(ke);
        }

        t.clamp(0.0, 1.0)
    }

    fn shade_rgb(&self, _depth: f32, normal: Vec3, kd: Vec3, ks: Vec3, ns: f32, ke: Vec3) -> Vec3 {
        let n = normal.normalize_or_zero();
        let l = self.light_dir.normalize_or_zero();
        let ndotl = n.dot(l).clamp(0.0, 1.0);
        let base = (self.ambient + (1.0 - self.ambient) * ndotl).clamp(0.0, 1.0);

        let mut out = kd * base;

        if ndotl > 0.0 && ns > 0.0 && ks.length_squared() > 0.0 {
            let v = Vec3::new(0.0, 0.0, 1.0);
            let h = (l + v).normalize_or_zero();
            let ndoth = n.dot(h).clamp(0.0, 1.0);
            let shin = ns.max(1.0);
            let spec = ndoth.powf(shin);
            out += ks * spec;
        }

        out + ke
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnlitShader;

impl Shader for UnlitShader {
    fn shade_scalar(
        &self,
        _depth: f32,
        _normal: Vec3,
        _kd: Vec3,
        _ks: Vec3,
        _ns: f32,
        ke: Vec3,
    ) -> f32 {
        (1.0 + luma(ke)).clamp(0.0, 1.0)
    }

    fn shade_rgb(
        &self,
        _depth: f32,
        _normal: Vec3,
        kd: Vec3,
        _ks: Vec3,
        _ns: f32,
        ke: Vec3,
    ) -> Vec3 {
        kd + ke
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

    pub fn id(&self) -> ShaderId {
        match self {
            BuiltinShader::Lambert(_) => ShaderId::Lambert,
            BuiltinShader::Unlit(_) => ShaderId::Unlit,
        }
    }
}

impl Shader for BuiltinShader {
    fn shade_scalar(&self, depth: f32, normal: Vec3, kd: Vec3, ks: Vec3, ns: f32, ke: Vec3) -> f32 {
        match *self {
            BuiltinShader::Lambert(s) => s.shade_scalar(depth, normal, kd, ks, ns, ke),
            BuiltinShader::Unlit(s) => s.shade_scalar(depth, normal, kd, ks, ns, ke),
        }
    }

    fn shade_rgb(&self, depth: f32, normal: Vec3, kd: Vec3, ks: Vec3, ns: f32, ke: Vec3) -> Vec3 {
        match *self {
            BuiltinShader::Lambert(s) => s.shade_rgb(depth, normal, kd, ks, ns, ke),
            BuiltinShader::Unlit(s) => s.shade_rgb(depth, normal, kd, ks, ns, ke),
        }
    }
}
