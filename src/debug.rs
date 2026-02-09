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

pub fn scalar_for_view(
    view: DebugView,
    shader: &BuiltinShader,
    depth: f32,
    normal: Vec3,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
) -> f32 {
    match view {
        DebugView::Final => shader.shade_scalar(depth, normal, kd, ks, ns, ke),
        DebugView::Depth => depth,
        DebugView::Normals => (0.5 + 0.5 * normal.z).clamp(0.0, 1.0),
        DebugView::Albedo => luma(kd),
    }
}

pub fn rgb_for_view(
    view: DebugView,
    shader: &BuiltinShader,
    depth: f32,
    normal: Vec3,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
) -> Vec3 {
    match view {
        DebugView::Final => shader.shade_rgb(depth, normal, kd, ks, ns, ke),
        DebugView::Depth => Vec3::splat(depth),
        DebugView::Normals => (normal + Vec3::ONE) * 0.5,
        DebugView::Albedo => kd,
    }
}
