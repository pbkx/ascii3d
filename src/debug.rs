use glam::Vec3;

use crate::shader::{BuiltinShader, Shader};
use crate::targets::buffer::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DebugView {
    #[default]
    Final,
    Depth,
    Normals,
    Albedo,
}

impl DebugView {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "final" => Some(DebugView::Final),
            "depth" => Some(DebugView::Depth),
            "normals" | "normal" => Some(DebugView::Normals),
            "albedo" => Some(DebugView::Albedo),
            _ => None,
        }
    }
}

#[must_use]
pub fn cell_for_view(view: DebugView, shader: &BuiltinShader, ramp: &[char], depth: f32, normal: Vec3, albedo: Vec3) -> Cell {
    match view {
        DebugView::Final => {
            let s = shader.shade(depth, normal, albedo);
            let t = s.intensity.clamp(0.0, 1.0);
            let ch = pick_ramp_char(ramp, t);
            Cell::new(ch, 255, 0, depth)
        }
        DebugView::Depth => {
            let (ch, _t) = map_depth_char(ramp, depth);
            Cell::new(ch, 180, 0, depth)
        }
        DebugView::Normals => {
            let (ch, fg) = map_normals_char(ramp, normal);
            Cell::new(ch, fg, 0, depth)
        }
        DebugView::Albedo => {
            let a = albedo.clamp(Vec3::ZERO, Vec3::ONE);
            let t = ((a.x + a.y + a.z) * (1.0 / 3.0)).clamp(0.0, 1.0);
            let ch = pick_ramp_char(ramp, t);
            let fg = (t * 255.0).round() as u8;
            Cell::new(ch, fg, 0, depth)
        }
    }
}

#[must_use]
pub fn rgb_for_view(view: DebugView, shader: &BuiltinShader, depth: f32, normal: Vec3, albedo: Vec3) -> Vec3 {
    match view {
        DebugView::Final => shader.shade(depth, normal, albedo).rgb.clamp(Vec3::ZERO, Vec3::ONE),
        DebugView::Depth => {
            let t = map_depth_t(depth);
            Vec3::splat(t)
        }
        DebugView::Normals => {
            let n = normal.normalize_or_zero();
            (n * 0.5 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE)
        }
        DebugView::Albedo => albedo.clamp(Vec3::ZERO, Vec3::ONE),
    }
}

#[must_use]
fn pick_ramp_char(ramp: &[char], t: f32) -> char {
    if ramp.is_empty() {
        return '#';
    }
    let t = t.clamp(0.0, 1.0);
    let idx = (t * (ramp.len().saturating_sub(1) as f32)).round() as usize;
    ramp[idx.min(ramp.len().saturating_sub(1))]
}

#[must_use]
fn map_depth_t(depth: f32) -> f32 {
    if !depth.is_finite() {
        return 0.0;
    }
    // Depth is stored in clip/NDC space (-1..1). Map to "near bright".
    (1.0 - (depth + 1.0) * 0.5).clamp(0.0, 1.0)
}

#[must_use]
fn map_depth_char(ramp: &[char], depth: f32) -> (char, f32) {
    let t = map_depth_t(depth);
    (pick_ramp_char(ramp, t), t)
}

#[must_use]
fn map_normals_char(ramp: &[char], normal: Vec3) -> (char, u8) {
    let n = normal.normalize_or_zero();
    let t = (n.z * 0.5 + 0.5).clamp(0.0, 1.0);
    let c = (((n.x.abs() + n.y.abs() + n.z.abs()) * (1.0 / 3.0)).clamp(0.0, 1.0) * 255.0).round() as u8;
    (pick_ramp_char(ramp, t), c)
}
