use glam::Vec3;

use crate::shader::{BuiltinShader, Shader};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DebugView {
    #[default]
    Final,
    Depth,
    Normals,
    Albedo,
}

impl DebugView {
    pub fn as_str(self) -> &'static str {
        match self {
            DebugView::Final => "final",
            DebugView::Depth => "depth",
            DebugView::Normals => "normals",
            DebugView::Albedo => "albedo",
        }
    }

    pub fn parse(s: &str) -> Option<DebugView> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "final" => DebugView::Final,
            "depth" => DebugView::Depth,
            "normals" | "normal" => DebugView::Normals,
            "albedo" | "kd" => DebugView::Albedo,
            _ => return None,
        })
    }
}

fn luma(rgb: Vec3) -> f32 {
    (0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z).clamp(0.0, 1.0)
}

fn ndc_depth_to_unit(depth: f32) -> f32 {
    (depth * 0.5 + 0.5).clamp(0.0, 1.0)
}

pub fn scalar_for_view(
    view: DebugView,
    shader: &BuiltinShader,
    depth: f32,
    view_pos: Vec3,
    normal: Vec3,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
) -> f32 {
    match view {
        DebugView::Final => shader.shade_scalar(depth, view_pos, normal, kd, ks, ns, ke),
        DebugView::Depth => ndc_depth_to_unit(depth),
        DebugView::Normals => (0.5 + 0.5 * normal.z).clamp(0.0, 1.0),
        DebugView::Albedo => luma(kd),
    }
}

pub fn rgb_for_view(
    view: DebugView,
    shader: &BuiltinShader,
    depth: f32,
    view_pos: Vec3,
    normal: Vec3,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
) -> Vec3 {
    match view {
        DebugView::Final => shader.shade_rgb(depth, view_pos, normal, kd, ks, ns, ke),
        DebugView::Depth => Vec3::splat(ndc_depth_to_unit(depth)),
        DebugView::Normals => (normal + Vec3::ONE) * 0.5,
        DebugView::Albedo => kd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{BuiltinShader, ShaderId};

    #[test]
    fn depth_view_remaps_ndc_to_unit() {
        let shader = BuiltinShader::from_id(ShaderId::Unlit);
        let n = Vec3::Z;
        let view_pos = Vec3::new(0.0, 0.0, -1.0);
        let kd = Vec3::ONE;
        let ks = Vec3::ZERO;
        let ns = 0.0;
        let ke = Vec3::ZERO;

        let s_near = scalar_for_view(DebugView::Depth, &shader, -1.0, view_pos, n, kd, ks, ns, ke);
        let s_mid = scalar_for_view(DebugView::Depth, &shader, 0.0, view_pos, n, kd, ks, ns, ke);
        let s_far = scalar_for_view(DebugView::Depth, &shader, 1.0, view_pos, n, kd, ks, ns, ke);
        assert!((s_near - 0.0).abs() < 1e-6);
        assert!((s_mid - 0.5).abs() < 1e-6);
        assert!((s_far - 1.0).abs() < 1e-6);

        let r = rgb_for_view(DebugView::Depth, &shader, 0.0, view_pos, n, kd, ks, ns, ke);
        assert!((r.x - 0.5).abs() < 1e-6);
        assert!((r.y - 0.5).abs() < 1e-6);
        assert!((r.z - 0.5).abs() < 1e-6);
    }
}
