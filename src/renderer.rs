use crate::{
    raster,
    shader::{BuiltinShader, Shader, ShaderId},
    targets::{BufferTarget, Cell, ImageTarget},
    GBuffer, Scene,
};
use glam::Vec3;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DebugView {
    Final,
    Depth,
    Normals,
}

#[derive(Clone, Debug)]
pub struct RendererConfig {
    width: usize,
    height: usize,
    ramp: Vec<char>,
    debug_view: DebugView,
    shader: ShaderId,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 40,
            ramp: " .:-=+*#%@".chars().collect(),
            debug_view: DebugView::Final,
            shader: ShaderId::Lambert,
        }
    }
}

impl RendererConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self::default().with_size(width, height)
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_ramp(mut self, ramp: &str) -> Self {
        self.ramp = ramp.chars().collect();
        if self.ramp.is_empty() {
            self.ramp = vec!['#'];
        }
        self
    }

    pub fn with_debug_view(mut self, view: DebugView) -> Self {
        self.debug_view = view;
        self
    }

    pub fn with_shader(mut self, shader: ShaderId) -> Self {
        self.shader = shader;
        self
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn ramp(&self) -> &[char] {
        &self.ramp
    }

    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    pub fn shader(&self) -> ShaderId {
        self.shader
    }
}

#[derive(Clone, Debug)]
pub struct Renderer {
    config: RendererConfig,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    fn pick_ramp_char(ramp: &[char], t: f32) -> char {
        if ramp.is_empty() {
            return '#';
        }
        if ramp.len() == 1 {
            return ramp[0];
        }
        let tt = t.clamp(0.0, 1.0);
        let mut idx = (tt * (ramp.len() as f32 - 1.0)).round() as usize;
        if idx >= ramp.len() {
            idx = ramp.len() - 1;
        }
        if idx == 0 && tt > 0.0001 {
            idx = 1;
        }
        ramp[idx]
    }

    fn map_depth_char(&self, depth: f32) -> char {
        let t = (1.0 - (depth + 1.0) * 0.5).clamp(0.0, 1.0);
        Self::pick_ramp_char(self.config.ramp(), t)
    }

    fn map_normals_char(&self, normal: Vec3) -> char {
        let t = (normal.normalize_or_zero().z * 0.5 + 0.5).clamp(0.0, 1.0);
        Self::pick_ramp_char(self.config.ramp(), t)
    }

    fn map_gbuffer_to_buffer(&self, gbuf: &GBuffer, out: &mut BufferTarget) {
        let w = gbuf.width();
        let h = gbuf.height();
        let depth = gbuf.depth_slice();
        let normal = gbuf.normal_slice();
        let albedo = gbuf.albedo_slice();
        let shader = BuiltinShader::from_id(self.config.shader());
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let z = depth[i];
                if z.is_infinite() {
                    continue;
                }
                let cell = match self.config.debug_view() {
                    DebugView::Final => {
                        let s = shader.shade(z, normal[i], albedo[i]);
                        let ch = Self::pick_ramp_char(self.config.ramp(), s.intensity);
                        Cell::new(ch, 255, 0, z)
                    }
                    DebugView::Depth => {
                        let ch = self.map_depth_char(z);
                        Cell::new(ch, 180, 0, z)
                    }
                    DebugView::Normals => {
                        let ch = self.map_normals_char(normal[i]);
                        let n = normal[i].normalize_or_zero().abs();
                        let c = ((n.x + n.y + n.z) / 3.0 * 255.0).round() as u8;
                        Cell::new(ch, c, 0, z)
                    }
                };
                let _ = out.set(x, y, cell);
            }
        }
    }

    fn map_gbuffer_to_image(&self, gbuf: &GBuffer, out: &mut ImageTarget) {
        let w = gbuf.width();
        let h = gbuf.height();
        let depth = gbuf.depth_slice();
        let normal = gbuf.normal_slice();
        let albedo = gbuf.albedo_slice();
        let shader = BuiltinShader::from_id(self.config.shader());
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let z = depth[i];
                if z.is_infinite() {
                    continue;
                }
                let (rgb, a) = match self.config.debug_view() {
                    DebugView::Final => {
                        let s = shader.shade(z, normal[i], albedo[i]);
                        (s.rgb, 255)
                    }
                    DebugView::Depth => {
                        let t = (1.0 - (z + 1.0) * 0.5).clamp(0.0, 1.0);
                        (Vec3::splat(t), 255)
                    }
                    DebugView::Normals => {
                        let n = normal[i].normalize_or_zero();
                        ((n * 0.5 + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE), 255)
                    }
                };
                let r = (rgb.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                let g = (rgb.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                let b = (rgb.z.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out.set_rgba(x, y, r, g, b, a);
            }
        }
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

    pub fn render_image(&self, scene: &Scene, target: &mut ImageTarget) {
        if target.width() != self.config.width() || target.height() != self.config.height() {
            *target = ImageTarget::new(self.config.width(), self.config.height());
        }
        target.clear_rgba(0, 0, 0, 0);
        let mut gbuf = GBuffer::new(self.config.width(), self.config.height());
        raster::render_to_gbuffer(scene, &mut gbuf);
        self.map_gbuffer_to_image(&gbuf, target);
    }
}

#[cfg(test)]
mod tests {
    use crate::{Material, Mesh, Renderer, RendererConfig, Scene, ShaderId, Transform};

    #[test]
    fn smoke_triangle_renders_deterministically() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
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
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
            Material::default(),
        );
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
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
            Material::default(),
        );
        let mut img_final = crate::targets::ImageTarget::new(64, 32);
        let mut img_norm = crate::targets::ImageTarget::new(64, 32);
        let r_final = Renderer::new(RendererConfig::default().with_size(64, 32));
        let r_norm = Renderer::new(
            RendererConfig::default()
                .with_size(64, 32)
                .with_debug_view(crate::renderer::DebugView::Normals),
        );
        r_final.render_image(&scene, &mut img_final);
        r_norm.render_image(&scene, &mut img_norm);
        assert_ne!(img_final.hash64(), img_norm.hash64());
    }

    #[test]
    fn shader_changes_output_hash() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
            Material::default(),
        );

        let mut img_lambert = crate::targets::ImageTarget::new(64, 32);
        let mut img_unlit = crate::targets::ImageTarget::new(64, 32);

        let r_lambert = Renderer::new(
            RendererConfig::default()
                .with_size(64, 32)
                .with_shader(ShaderId::Lambert),
        );
        let r_unlit = Renderer::new(
            RendererConfig::default()
                .with_size(64, 32)
                .with_shader(ShaderId::Unlit),
        );

        r_lambert.render_image(&scene, &mut img_lambert);
        r_unlit.render_image(&scene, &mut img_unlit);

        assert_ne!(img_lambert.hash64(), img_unlit.hash64());
    }

    #[cfg(feature = "png")]
    #[test]
    fn smoke_triangle_png_writes_deterministically() {
        let mut scene = Scene::new();
        scene.add_object(
            Mesh::unit_triangle(),
            Transform::IDENTITY,
            Material::default(),
        );
        let renderer = Renderer::new(RendererConfig::default().with_size(64, 32));
        let mut img = crate::targets::ImageTarget::new(64, 32);
        renderer.render_image(&scene, &mut img);
        let p1 = img.write_png_to_vec().unwrap();
        let p2 = img.write_png_to_vec().unwrap();
        assert!(p1.len() > 8);
        assert_eq!(&p1[0..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(p1, p2);
    }
}
