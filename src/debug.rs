use glam::Vec3;

use crate::shader::{BuiltinShader, Shader};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
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

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "final" | "f" => Some(DebugView::Final),
            "depth" | "d" => Some(DebugView::Depth),
            "normals" | "n" => Some(DebugView::Normals),
            "albedo" | "a" => Some(DebugView::Albedo),
            _ => None,
        }
    }
}

pub fn scalar_for_view(view: DebugView, shader: &BuiltinShader, depth: f32, normal: Vec3, albedo: Vec3) -> f32 {
    match view {
        DebugView::Final => shader.shade(depth, normal, albedo).intensity,
        DebugView::Depth => (1.0 - depth / 10.0).clamp(0.0, 1.0),
        DebugView::Normals => (normal.dot(Vec3::new(0.0, 0.0, 1.0)) * 0.5 + 0.5).clamp(0.0, 1.0),
        DebugView::Albedo => (0.2126 * albedo.x + 0.7152 * albedo.y + 0.0722 * albedo.z).clamp(0.0, 1.0),
    }
}

pub fn rgb_for_view(view: DebugView, shader: &BuiltinShader, depth: f32, normal: Vec3, albedo: Vec3) -> Vec3 {
    match view {
        DebugView::Final => shader.shade(depth, normal, albedo).rgb.clamp(Vec3::ZERO, Vec3::ONE),
        DebugView::Depth => {
            let t = scalar_for_view(DebugView::Depth, shader, depth, normal, albedo);
            Vec3::splat(t)
        }
        DebugView::Normals => (normal * 0.5 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE),
        DebugView::Albedo => albedo.clamp(Vec3::ZERO, Vec3::ONE),
    }
}
