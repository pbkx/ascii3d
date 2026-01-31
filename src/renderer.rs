use crate::{
    raster,
    scene::Scene,
    targets::{BufferTarget, ImageTarget},
};

pub struct Renderer {
    config: RendererConfig,
}

#[derive(Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self { width: 80, height: 40 }
    }
}

impl RendererConfig {
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> RendererConfig {
        self.config
    }

    pub fn render(&self, scene: &Scene, target: &mut BufferTarget) {
        raster::render_to_buffer(scene, target);
    }

    pub fn render_image(&self, scene: &Scene, target: &mut ImageTarget) {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let mut buf = BufferTarget::new(w, h);
        self.render(scene, &mut buf);
        target.clear_rgba(0, 0, 0, 255);
        let inf_bits = f32::INFINITY.to_bits();
        let mw = target.width().min(w);
        let mh = target.height().min(h);
        for y in 0..mh {
            for x in 0..mw {
                let cell = buf.get(x, y).unwrap_or_default();
                let occupied = cell.ch != ' ' || cell.fg != 0 || cell.bg != 0 || cell.depth_bits != inf_bits;
            let v = if occupied {
                if cell.fg != 0 { cell.fg } else { 255u8 }
            } else {
                0u8
            };
                let _ = target.set_rgba(x, y, v, v, v, 255);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Renderer, RendererConfig};
    use crate::{
        material::Material,
        mesh::Mesh,
        scene::Scene,
        targets::{BufferTarget, ImageTarget},
        transform::Transform,
    };

    #[test]
    fn smoke_triangle_renders_deterministically() {
        let renderer = Renderer::new(RendererConfig::default().with_size(80, 40));
        let mut scene = Scene::new();
        let _ = scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());

        let mut target = BufferTarget::new(80, 40);
        renderer.render(&scene, &mut target);
        let h1 = target.hash64();
        let empty = BufferTarget::new(80, 40);
        assert_ne!(h1, empty.hash64());

        let mut target2 = BufferTarget::new(80, 40);
        renderer.render(&scene, &mut target2);
        let h2 = target2.hash64();
        assert_eq!(h1, h2);
    }

    #[test]
    fn smoke_triangle_image_hash_snapshot() {
        let renderer = Renderer::new(RendererConfig::default().with_size(80, 40));
        let mut scene = Scene::new();
        let _ = scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, Material::default());

        let mut img = ImageTarget::new(80, 40);
        renderer.render_image(&scene, &mut img);
        let h1 = img.hash64();

        let empty = ImageTarget::new(80, 40);
        assert_ne!(h1, empty.hash64());

        let mut img2 = ImageTarget::new(80, 40);
        renderer.render_image(&scene, &mut img2);
        let h2 = img2.hash64();
        assert_eq!(h1, h2);
    }
}
