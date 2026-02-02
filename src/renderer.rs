use crate::{
    debug::{rgb_for_view, scalar_for_view, DebugView},
    glyph::{AsciiRamp, GlyphMode},
    gbuffer::GBuffer,
    raster,
    scene::Scene,
    shader::{BuiltinShader, ShaderId},
    targets::{BufferTarget, Cell, ImageTarget},
};

use glam::Vec3;

#[derive(Clone, Debug)]
pub struct RendererConfig {
    width: usize,
    height: usize,
    shader: BuiltinShader,
    debug_view: DebugView,
    glyph_mode: GlyphMode,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 40,
            shader: BuiltinShader::from_id(ShaderId::Lambert),
            debug_view: DebugView::Final,
            glyph_mode: GlyphMode::default(),
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
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self { config }
    }

    pub fn render(&self, scene: &Scene, target: &mut BufferTarget) {
        if target.width() != self.config.width() || target.height() != self.config.height() {
            *target = BufferTarget::new(self.config.width(), self.config.height());
        }
        target.clear(Cell::new(' ', 0, 0, f32::INFINITY));
        let mut gbuf = GBuffer::new(self.config.width(), self.config.height());
        raster::render_to_gbuffer(scene, &mut gbuf);
        self.map_gbuffer_to_buffer(&gbuf, target);
    }

    fn map_gbuffer_to_buffer(&self, gbuf: &GBuffer, target: &mut BufferTarget) {
        for y in 0..target.height() {
            for x in 0..target.width() {
                let Some(p) = gbuf.at(x, y) else {
                    continue;
                };
                if !p.depth.is_finite() {
                    continue;
                }
                let t = scalar_for_view(self.config.debug_view(), self.config.shader(), p.depth, p.normal, p.albedo);
                let cell = self.config.glyph_mode().cell_from_scalar(t, p.depth);
                let _ = target.set(x, y, cell);
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
                let c = (rgb.clamp(Vec3::ZERO, Vec3::ONE) * 255.0 + Vec3::splat(0.5)).as_uvec3();
                let _ = target.set_rgba(x, y, c.x as u8, c.y as u8, c.z as u8, 255);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DebugView, Material, Mesh, Renderer, RendererConfig, Scene, ShaderId, Transform};

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
}
