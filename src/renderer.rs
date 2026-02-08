use crate::{
    debug::{rgb_for_view, scalar_for_view, DebugView},
    dither::{Dither, DitherMode},
    glyph::{AsciiRamp, GlyphMode},
    gbuffer::GBuffer,
    raster,
    scene::Scene,
    shader::{BuiltinShader, ShaderId},
    temporal::{TemporalConfig, TemporalQuantizer, TemporalState},
    targets::{BufferTarget, Cell, ImageTarget},
    types::Rgb8,
};

use glam::Vec3;
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct RendererConfig {
    width: usize,
    height: usize,
    shader: BuiltinShader,
    debug_view: DebugView,
    glyph_mode: GlyphMode,
    dither_mode: DitherMode,
    temporal: TemporalConfig,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 40,
            shader: BuiltinShader::from_id(ShaderId::Lambert),
            debug_view: DebugView::Final,
            glyph_mode: GlyphMode::default(),
            dither_mode: DitherMode::None,
            temporal: TemporalConfig::default(),
        }
    }
}

impl RendererConfig {
    pub fn new(width: usize, height: usize) -> Self {
        RendererConfig::default().with_size(width, height)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn shader(&self) -> &BuiltinShader {
        &self.shader
    }

    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    pub fn glyph_mode(&self) -> &GlyphMode {
        &self.glyph_mode
    }

    pub fn dither_mode(&self) -> DitherMode {
        self.dither_mode
    }

    pub fn temporal(&self) -> &TemporalConfig {
        &self.temporal
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_shader_id(mut self, id: ShaderId) -> Self {
        self.shader = BuiltinShader::from_id(id);
        self
    }

    pub fn with_debug_view(mut self, view: DebugView) -> Self {
        self.debug_view = view;
        self
    }

    pub fn with_glyph_mode(mut self, mode: GlyphMode) -> Self {
        self.glyph_mode = mode;
        self
    }

    pub fn with_dither_mode(mut self, mode: DitherMode) -> Self {
        self.dither_mode = mode;
        self
    }

    pub fn with_temporal_config(mut self, config: TemporalConfig) -> Self {
        self.temporal = config;
        self
    }

    pub fn with_temporal_enabled(mut self, enabled: bool) -> Self {
        self.temporal.enabled = enabled;
        self
    }

    pub fn with_temporal_ema_alpha(mut self, alpha: f32) -> Self {
        self.temporal.ema_alpha = alpha;
        self
    }

    pub fn with_temporal_hysteresis(mut self, hysteresis: f32) -> Self {
        self.temporal.hysteresis = hysteresis;
        self
    }

    pub fn with_temporal_anchored_dither(mut self, anchored: bool) -> Self {
        self.temporal.anchored_dither = anchored;
        self
    }

    pub fn with_ascii_ramp(self, ramp: AsciiRamp) -> Self {
        self.with_glyph_mode(GlyphMode::AsciiRamp(ramp))
    }

    pub fn with_ramp_name(mut self, name: &str) -> Self {
        if let Some(r) = AsciiRamp::from_name(name) {
            self.glyph_mode = GlyphMode::AsciiRamp(r);
        }
        self
    }

    pub fn with_ramp_chars(mut self, chars: &str) -> Self {
        self.glyph_mode = GlyphMode::AsciiRamp(AsciiRamp::new(chars));
        self
    }
}

pub struct Renderer {
    config: RendererConfig,
    temporal: RefCell<TemporalState>,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            temporal: RefCell::new(TemporalState::new()),
        }
    }

    pub fn render(&self, scene: &Scene, target: &mut BufferTarget) {
        if target.width() != self.config.width() || target.height() != self.config.height() {
            *target = BufferTarget::new(self.config.width(), self.config.height());
        }
        target.clear(Cell::new(' ', Rgb8::BLACK, Rgb8::BLACK, f32::INFINITY));
        let mut gbuf = match self.config.glyph_mode() {
            GlyphMode::HalfBlock => GBuffer::new(self.config.width(), self.config.height().saturating_mul(2)),
            _ => GBuffer::new(self.config.width(), self.config.height()),
        };
        raster::render_to_gbuffer(scene, &mut gbuf);
        match self.config.glyph_mode() {
            GlyphMode::HalfBlock => self.map_gbuffer_to_buffer_half_block(&gbuf, target),
            _ => self.map_gbuffer_to_buffer(&gbuf, target),
        }
    }

    fn map_gbuffer_to_buffer(&self, gbuf: &GBuffer, target: &mut BufferTarget) {
        let dither_mode = self.config.dither_mode();
        let mut dither = Dither::new(dither_mode, target.width());
        let temporal_cfg = self.config.temporal();
        let mut temporal = self.temporal.borrow_mut();
        if temporal_cfg.enabled {
            temporal.resize(target.width(), target.height());
        }
        for y in 0..target.height() {
            for x in 0..target.width() {
                let Some(p) = gbuf.at(x, y) else {
                    continue;
                };
                if !p.depth.is_finite() {
                    continue;
                }
                let shade_scalar = scalar_for_view(
                    self.config.debug_view(),
                    self.config.shader(),
                    p.depth,
                    p.normal,
                    p.albedo,
                );
                let rgb = rgb_for_view(
                    self.config.debug_view(),
                    self.config.shader(),
                    p.depth,
                    p.normal,
                    p.albedo,
                );
                let rgb_u8 = (rgb.clamp(Vec3::ZERO, Vec3::ONE) * 255.0 + Vec3::splat(0.5)).as_uvec3();
                let mut cell = match self.config.glyph_mode() {
                    GlyphMode::AsciiRamp(ramp) => {
                        let bytes = ramp.bytes();
                        if bytes.is_empty() {
                            Cell::new(' ', Rgb8::BLACK, Rgb8::BLACK, p.depth)
                        } else {
                            let mut t = shade_scalar.clamp(0.0, 1.0);
                            let g = ramp.gamma();
                            if g != 1.0 {
                                t = t.powf(g);
                            }

                            let idx = if temporal_cfg.enabled {
                                let mut q = if temporal_cfg.anchored_dither && dither_mode == DitherMode::None {
                                    TemporalQuantizer::Anchored
                                } else {
                                    TemporalQuantizer::Dither(&mut dither)
                                };
                                temporal.step_index(x, y, t, bytes.len(), &mut q, temporal_cfg)
                            } else {
                                if dither_mode == DitherMode::None {
                                    if bytes.len() <= 1 {
                                        0
                                    } else {
                                        let s = t * (bytes.len() as f32 - 1.0);
                                        (s + 0.5).floor() as usize
                                    }
                                } else {
                                    dither.quantize_index(t, bytes.len(), x, y)
                                }
                            };

                            let idx = idx.min(bytes.len().saturating_sub(1));
                            Cell::new(bytes[idx] as char, Rgb8::BLACK, Rgb8::BLACK, p.depth)
                        }
                    }
                    GlyphMode::HalfBlock => self.config.glyph_mode().cell_from_scalar(shade_scalar, p.depth),
                };
                cell.fg = Rgb8::new(
                    u8::try_from(rgb_u8.x).unwrap_or(255),
                    u8::try_from(rgb_u8.y).unwrap_or(255),
                    u8::try_from(rgb_u8.z).unwrap_or(255),
                );
                cell.bg = Rgb8::BLACK;
                let _ = target.set(x, y, cell);
            }
            dither.finish_row();
        }
    }

    fn map_gbuffer_to_buffer_half_block(&self, gbuf: &GBuffer, target: &mut BufferTarget) {
        fn to_u8_rgb(rgb: Vec3) -> Rgb8 {
            let rgb_u8 = (rgb.clamp(Vec3::ZERO, Vec3::ONE) * 255.0 + Vec3::splat(0.5)).as_uvec3();
            Rgb8::new(rgb_u8.x as u8, rgb_u8.y as u8, rgb_u8.z as u8)
        }

        for y in 0..target.height() {
            let y0 = y.saturating_mul(2);
            let y1 = y0 + 1;
            for x in 0..target.width() {
                let p0 = if y0 < gbuf.height() { gbuf.at(x, y0) } else { None };
                let p1 = if y1 < gbuf.height() { gbuf.at(x, y1) } else { None };

                let p0_ok = p0.map_or(false, |p| p.depth.is_finite());
                let p1_ok = p1.map_or(false, |p| p.depth.is_finite());

                if p0_ok && p1_ok {
                    let p0 = p0.unwrap();
                    let p1 = p1.unwrap();
                    let rgb0 = to_u8_rgb(rgb_for_view(
                        self.config.debug_view(),
                        self.config.shader(),
                        p0.depth,
                        p0.normal,
                        p0.albedo,
                    ));
                    let rgb1 = to_u8_rgb(rgb_for_view(
                        self.config.debug_view(),
                        self.config.shader(),
                        p1.depth,
                        p1.normal,
                        p1.albedo,
                    ));
                    let depth = p0.depth.min(p1.depth);
                    let _ = target.set(x, y, Cell::new('▀', rgb0, rgb1, depth));
                } else if p0_ok {
                    let p0 = p0.unwrap();
                    let rgb0 = to_u8_rgb(rgb_for_view(
                        self.config.debug_view(),
                        self.config.shader(),
                        p0.depth,
                        p0.normal,
                        p0.albedo,
                    ));
                    let _ = target.set(x, y, Cell::new('▀', rgb0, Rgb8::BLACK, p0.depth));
                } else if p1_ok {
                    let p1 = p1.unwrap();
                    let rgb1 = to_u8_rgb(rgb_for_view(
                        self.config.debug_view(),
                        self.config.shader(),
                        p1.depth,
                        p1.normal,
                        p1.albedo,
                    ));
                    let _ = target.set(x, y, Cell::new('▄', rgb1, Rgb8::BLACK, p1.depth));
                } else {
                    let _ = target.set(x, y, Cell::new(' ', Rgb8::BLACK, Rgb8::BLACK, f32::INFINITY));
                }
            }
        }
    }

    pub fn render_image(&self, scene: &Scene, target: &mut ImageTarget) {
        if target.width() != self.config.width() || target.height() != self.config.height() {
            *target = ImageTarget::new(self.config.width(), self.config.height());
        }
        target.clear_rgba(0, 0, 0, 0);
        let mut gbuf = GBuffer::new(self.config.width(), self.config.height());
        raster::render_to_gbuffer(scene, &mut gbuf);
        self.map_gbuffer_to_image(&gbuf, target);
    }

    fn map_gbuffer_to_image(&self, gbuf: &GBuffer, target: &mut ImageTarget) {
        for y in 0..target.height() {
            for x in 0..target.width() {
                let Some(p) = gbuf.at(x, y) else {
                    continue;
                };
                if !p.depth.is_finite() {
                    continue;
                }
                let rgb = rgb_for_view(self.config.debug_view(), self.config.shader(), p.depth, p.normal, p.albedo);
                let rgb_u8 = (rgb.clamp(Vec3::ZERO, Vec3::ONE) * 255.0 + Vec3::splat(0.5)).as_uvec3();
                let _ = target.set_rgba(
                    x,
                    y,
                    u8::try_from(rgb_u8.x).unwrap_or(255),
                    u8::try_from(rgb_u8.y).unwrap_or(255),
                    u8::try_from(rgb_u8.z).unwrap_or(255),
                    255,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DebugView, DitherMode, GlyphMode, Material, Mesh, Renderer, RendererConfig, Scene, ShaderId, TemporalConfig, Transform};

    #[test]
    fn smoke_triangle_renders_deterministically() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::from_rotation(glam::Quat::from_rotation_y(1.0)),
            Material::default(),
        );
        let renderer = Renderer::new(RendererConfig::default().with_size(64, 32));
        let mut target = crate::targets::BufferTarget::new(64, 32);
        let empty_hash = target.hash64();
        renderer.render(&scene, &mut target);
        let h1 = target.hash64();
        assert_ne!(h1, empty_hash);
        renderer.render(&scene, &mut target);
        let h2 = target.hash64();
        assert_eq!(h1, h2);
    }

    #[test]
    fn half_block_mode_changes_buffer_hash_and_uses_blocks() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::from_rotation(glam::Quat::from_rotation_y(1.0)),
            Material::default(),
        );

        let renderer_ascii = Renderer::new(RendererConfig::default().with_size(64, 32));
        let renderer_half = Renderer::new(
            RendererConfig::default()
                .with_size(64, 32)
                .with_glyph_mode(GlyphMode::HalfBlock),
        );

        let mut t_ascii = crate::targets::BufferTarget::new(64, 32);
        let mut t_half = crate::targets::BufferTarget::new(64, 32);

        renderer_ascii.render(&scene, &mut t_ascii);
        renderer_half.render(&scene, &mut t_half);

        assert_ne!(t_ascii.hash64(), t_half.hash64());

        let mut saw_block = false;
        for y in 0..t_half.height() {
            for x in 0..t_half.width() {
                let Some(c) = t_half.cell(x, y) else {
                    continue;
                };
                if c.ch == '▀' || c.ch == '▄' {
                    saw_block = true;
                }
            }
        }
        assert!(saw_block);
    }

    #[test]
    fn dither_modes_change_output_hash_and_are_deterministic() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
            Material::default(),
        );

        let renderer_none =
            Renderer::new(RendererConfig::default().with_size(64, 32).with_dither_mode(DitherMode::None));
        let renderer_ordered =
            Renderer::new(RendererConfig::default().with_size(64, 32).with_dither_mode(DitherMode::Ordered));
        let renderer_noise =
            Renderer::new(RendererConfig::default().with_size(64, 32).with_dither_mode(DitherMode::BlueNoise));
        let renderer_diff =
            Renderer::new(RendererConfig::default().with_size(64, 32).with_dither_mode(DitherMode::ErrorDiffusion));

        let mut t_none = crate::targets::BufferTarget::new(64, 32);
        let mut t_ordered = crate::targets::BufferTarget::new(64, 32);
        let mut t_noise = crate::targets::BufferTarget::new(64, 32);
        let mut t_diff = crate::targets::BufferTarget::new(64, 32);

        renderer_none.render(&scene, &mut t_none);
        renderer_ordered.render(&scene, &mut t_ordered);
        renderer_noise.render(&scene, &mut t_noise);
        renderer_diff.render(&scene, &mut t_diff);

        let h_none = t_none.hash64();
        let h_ordered = t_ordered.hash64();
        let h_noise = t_noise.hash64();
        let h_diff = t_diff.hash64();

        assert_ne!(h_none, h_ordered);
        assert_ne!(h_none, h_noise);
        assert_ne!(h_none, h_diff);

        let mut t_ordered_2 = crate::targets::BufferTarget::new(64, 32);
        renderer_ordered.render(&scene, &mut t_ordered_2);
        assert_eq!(h_ordered, t_ordered_2.hash64());

        let mut t_noise_2 = crate::targets::BufferTarget::new(64, 32);
        renderer_noise.render(&scene, &mut t_noise_2);
        assert_eq!(h_noise, t_noise_2.hash64());

        let mut t_diff_2 = crate::targets::BufferTarget::new(64, 32);
        renderer_diff.render(&scene, &mut t_diff_2);
        assert_eq!(h_diff, t_diff_2.hash64());
    }

    #[test]
    fn smoke_triangle_image_hash_snapshot() {
        let mut scene = Scene::new();
        scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());
        let renderer = Renderer::new(RendererConfig::default().with_size(64, 32));
        let mut img = crate::targets::ImageTarget::new(64, 32);
        let empty_hash = img.hash64();
        renderer.render_image(&scene, &mut img);
        let h1 = img.hash64();
        assert_ne!(h1, empty_hash);
        renderer.render_image(&scene, &mut img);
        let h2 = img.hash64();
        assert_eq!(h1, h2);
    }

    #[test]
    fn debug_normals_changes_output() {
        let mut scene = Scene::new();
        scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());
        let mut img_final = crate::targets::ImageTarget::new(64, 32);
        let mut img_norm = crate::targets::ImageTarget::new(64, 32);
        let r_final = Renderer::new(RendererConfig::default().with_size(64, 32));
        let r_norm = Renderer::new(RendererConfig::default().with_size(64, 32).with_debug_view(DebugView::Normals));
        r_final.render_image(&scene, &mut img_final);
        r_norm.render_image(&scene, &mut img_norm);
        assert_ne!(img_final.hash64(), img_norm.hash64());
    }

    #[test]
    fn shader_changes_output_hash() {
        let mut scene = Scene::new();
        scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());
        let mut img_lambert = crate::targets::ImageTarget::new(64, 32);
        let mut img_unlit = crate::targets::ImageTarget::new(64, 32);
        let r_lambert = Renderer::new(RendererConfig::default().with_size(64, 32).with_shader_id(ShaderId::Lambert));
        let r_unlit = Renderer::new(RendererConfig::default().with_size(64, 32).with_shader_id(ShaderId::Unlit));
        r_lambert.render_image(&scene, &mut img_lambert);
        r_unlit.render_image(&scene, &mut img_unlit);
        assert_ne!(img_lambert.hash64(), img_unlit.hash64());
    }

    #[test]
    fn debug_view_modes_produce_distinct_hashes() {
        let mut scene = Scene::new();
        scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());
        let views = [DebugView::Final, DebugView::Depth, DebugView::Normals, DebugView::Albedo];
        let mut hashes = std::collections::HashSet::new();
        for v in views {
            let renderer = Renderer::new(RendererConfig::default().with_size(64, 32).with_debug_view(v));
            let mut img = crate::targets::ImageTarget::new(64, 32);
            renderer.render_image(&scene, &mut img);
            hashes.insert(img.hash64());
        }
        assert_eq!(hashes.len(), views.len());
    }

    fn glyph_churn(a: &crate::targets::BufferTarget, b: &crate::targets::BufferTarget) -> f32 {
        assert_eq!(a.width(), b.width());
        assert_eq!(a.height(), b.height());
        let mut changed = 0usize;
        let mut total = 0usize;
        for (ca, cb) in a.as_slice().iter().zip(b.as_slice().iter()) {
            let mask = ca.ch != ' ' || cb.ch != ' ';
            if !mask {
                continue;
            }
            total += 1;
            if ca.ch != cb.ch {
                changed += 1;
            }
        }
        if total == 0 {
            return 0.0;
        }
        (changed as f32) / (total as f32)
    }

    fn make_scene(angle: f32) -> Scene {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_cube(),
            Transform::from_rotation(glam::Quat::from_rotation_y(angle)),
            Material::default(),
        );
        scene
    }


    #[test]
    fn temporal_reduces_glyph_churn_two_frames() {
        let scene_a = make_scene(0.55);
        let scene_b = make_scene(0.57);

        let w = 80;
        let h = 40;

        let renderer_no_temporal = Renderer::new(
            RendererConfig::default()
                .with_size(w, h)
                .with_dither_mode(DitherMode::BlueNoise)
                .with_temporal_enabled(false),
        );
        let mut a1 = crate::targets::BufferTarget::new(w, h);
        let mut a2 = crate::targets::BufferTarget::new(w, h);
        renderer_no_temporal.render(&scene_a, &mut a1);
        renderer_no_temporal.render(&scene_b, &mut a2);

        let churn_no = glyph_churn(&a1, &a2);
        assert!(churn_no > 0.01);

        let renderer_temporal = Renderer::new(
            RendererConfig::default()
                .with_size(w, h)
                .with_dither_mode(DitherMode::BlueNoise)
                .with_temporal_config(TemporalConfig {
                    enabled: true,
                    ema_alpha: 0.35,
                    hysteresis: 0.02,
                    anchored_dither: true,
                }),
        );

        let mut b1 = crate::targets::BufferTarget::new(w, h);
        let mut b2 = crate::targets::BufferTarget::new(w, h);
        renderer_temporal.render(&scene_a, &mut b1);
        renderer_temporal.render(&scene_b, &mut b2);

        let churn_temporal = glyph_churn(&b1, &b2);
        assert!(churn_temporal < churn_no);
        assert!(churn_temporal <= churn_no * 0.9);
    }
}
