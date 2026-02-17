use glam::Vec3;

use crate::Light;

#[derive(Clone, Copy, Debug)]
pub struct ShadeSample {
    pub intensity: f32,
    pub rgb: Vec3,
}

pub trait Shader {
    fn shade_scalar(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> f32;
    fn shade_rgb(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> Vec3;

    fn shade(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> ShadeSample {
        ShadeSample {
            intensity: self.shade_scalar(depth, view_pos, normal, kd, ks, ns, ke),
            rgb: self.shade_rgb(depth, view_pos, normal, kd, ks, ns, ke),
        }
    }
}

fn luma(rgb: Vec3) -> f32 {
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

fn view_dir_from_view_pos(view_pos: Vec3) -> Vec3 {
    let v = (-view_pos).normalize_or_zero();
    if v.length_squared() > 0.0 {
        v
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

fn light_vector_and_radiance(light: Light, view_pos: Vec3) -> Option<(Vec3, Vec3)> {
    match light {
        Light::Directional {
            direction,
            color,
            intensity,
        } => {
            let l = (-direction).normalize_or_zero();
            let radiance = color.max(Vec3::ZERO) * intensity.max(0.0);
            if l.length_squared() > 0.0 && radiance.length_squared() > 0.0 {
                Some((l, radiance))
            } else {
                None
            }
        }
        Light::Point {
            position,
            color,
            intensity,
        } => {
            let to_light = position - view_pos;
            let dist_sq = to_light.length_squared();
            if dist_sq <= 1e-6 {
                return None;
            }
            let l = to_light * dist_sq.sqrt().recip();
            let attenuation = dist_sq.max(1e-4).recip();
            let radiance = color.max(Vec3::ZERO) * intensity.max(0.0) * attenuation;
            if radiance.length_squared() > 0.0 {
                Some((l, radiance))
            } else {
                None
            }
        }
    }
}

impl LambertShader {
    fn shade_scalar_with_lights(
        &self,
        _depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        _kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
        lights: &[Light],
    ) -> f32 {
        let n = normal.normalize_or_zero();
        let v = view_dir_from_view_pos(view_pos);

        if lights.is_empty() {
            let l = self.light_dir.normalize_or_zero();
            let ndotl = n.dot(l).clamp(0.0, 1.0);

            let base = (self.ambient + (1.0 - self.ambient) * ndotl).clamp(0.0, 1.0);
            let mut t = base;

            if ndotl > 0.0 && ns > 0.0 && ks.length_squared() > 0.0 {
                let h = (l + v).normalize_or_zero();
                let ndoth = n.dot(h).clamp(0.0, 1.0);
                let shin = ns.max(1.0);
                let spec = ndoth.powf(shin);
                t += luma(ks) * spec;
            }

            if ke.length_squared() > 0.0 {
                t += luma(ke);
            }

            return t.clamp(0.0, 1.0);
        }

        let diffuse_weight = (1.0 - self.ambient).clamp(0.0, 1.0);
        let mut t = self.ambient.clamp(0.0, 1.0);

        for &light in lights {
            let Some((l, radiance)) = light_vector_and_radiance(light, view_pos) else {
                continue;
            };
            let ndotl = n.dot(l).clamp(0.0, 1.0);
            if ndotl <= 0.0 {
                continue;
            }
            t += diffuse_weight * ndotl * luma(radiance);

            if ns > 0.0 && ks.length_squared() > 0.0 {
                let h = (l + v).normalize_or_zero();
                let ndoth = n.dot(h).clamp(0.0, 1.0);
                let shin = ns.max(1.0);
                let spec = ndoth.powf(shin);
                t += luma(ks * spec * radiance);
            }
        }

        if ke.length_squared() > 0.0 {
            t += luma(ke);
        }

        t.clamp(0.0, 1.0)
    }

    fn shade_rgb_with_lights(
        &self,
        _depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
        lights: &[Light],
    ) -> Vec3 {
        let n = normal.normalize_or_zero();
        let v = view_dir_from_view_pos(view_pos);

        if lights.is_empty() {
            let l = self.light_dir.normalize_or_zero();
            let ndotl = n.dot(l).clamp(0.0, 1.0);
            let base = (self.ambient + (1.0 - self.ambient) * ndotl).clamp(0.0, 1.0);

            let mut out = kd * base;
            if ndotl > 0.0 && ns > 0.0 && ks.length_squared() > 0.0 {
                let h = (l + v).normalize_or_zero();
                let ndoth = n.dot(h).clamp(0.0, 1.0);
                let shin = ns.max(1.0);
                let spec = ndoth.powf(shin);
                out += ks * spec;
            }

            return out + ke;
        }

        let diffuse_weight = (1.0 - self.ambient).clamp(0.0, 1.0);
        let mut out = kd * self.ambient.clamp(0.0, 1.0);

        for &light in lights {
            let Some((l, radiance)) = light_vector_and_radiance(light, view_pos) else {
                continue;
            };
            let ndotl = n.dot(l).clamp(0.0, 1.0);
            if ndotl <= 0.0 {
                continue;
            }
            out += kd * (diffuse_weight * ndotl) * radiance;

            if ns > 0.0 && ks.length_squared() > 0.0 {
                let h = (l + v).normalize_or_zero();
                let ndoth = n.dot(h).clamp(0.0, 1.0);
                let shin = ns.max(1.0);
                let spec = ndoth.powf(shin);
                out += ks * spec * radiance;
            }
        }

        out + ke
    }
}

impl Shader for LambertShader {
    fn shade_scalar(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> f32 {
        self.shade_scalar_with_lights(depth, view_pos, normal, kd, ks, ns, ke, &[])
    }

    fn shade_rgb(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> Vec3 {
        self.shade_rgb_with_lights(depth, view_pos, normal, kd, ks, ns, ke, &[])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnlitShader;

impl Shader for UnlitShader {
    fn shade_scalar(
        &self,
        _depth: f32,
        _view_pos: Vec3,
        _normal: Vec3,
        kd: Vec3,
        _ks: Vec3,
        _ns: f32,
        ke: Vec3,
    ) -> f32 {
        luma(kd + ke).clamp(0.0, 1.0)
    }

    fn shade_rgb(
        &self,
        _depth: f32,
        _view_pos: Vec3,
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

    pub fn shade_scalar_with_lights(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
        lights: &[Light],
    ) -> f32 {
        match *self {
            BuiltinShader::Lambert(s) => {
                s.shade_scalar_with_lights(depth, view_pos, normal, kd, ks, ns, ke, lights)
            }
            BuiltinShader::Unlit(s) => s.shade_scalar(depth, view_pos, normal, kd, ks, ns, ke),
        }
    }

    pub fn shade_rgb_with_lights(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
        lights: &[Light],
    ) -> Vec3 {
        match *self {
            BuiltinShader::Lambert(s) => {
                s.shade_rgb_with_lights(depth, view_pos, normal, kd, ks, ns, ke, lights)
            }
            BuiltinShader::Unlit(s) => s.shade_rgb(depth, view_pos, normal, kd, ks, ns, ke),
        }
    }
}

impl Shader for BuiltinShader {
    fn shade_scalar(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> f32 {
        match *self {
            BuiltinShader::Lambert(s) => s.shade_scalar(depth, view_pos, normal, kd, ks, ns, ke),
            BuiltinShader::Unlit(s) => s.shade_scalar(depth, view_pos, normal, kd, ks, ns, ke),
        }
    }

    fn shade_rgb(
        &self,
        depth: f32,
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> Vec3 {
        match *self {
            BuiltinShader::Lambert(s) => s.shade_rgb(depth, view_pos, normal, kd, ks, ns, ke),
            BuiltinShader::Unlit(s) => s.shade_rgb(depth, view_pos, normal, kd, ks, ns, ke),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Shader, UnlitShader};
    use glam::Vec3;

    #[test]
    fn unlit_scalar_uses_albedo_luma() {
        let shader = UnlitShader;
        let dark = shader.shade_scalar(
            0.0,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
        );
        let bright = shader.shade_scalar(
            0.0,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
        );
        assert!(bright > dark);
    }
}
