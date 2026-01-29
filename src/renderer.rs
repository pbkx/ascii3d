use crate::{
    raster::{rasterize, RasterConfig, RasterSurface},
    scene::Scene,
    targets::buffer::BufferTarget,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
}

impl RendererConfig {
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
        }
    }
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 120,
            height: 40,
        }
    }
}

pub struct Renderer {
    config: RendererConfig,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
        }
    }

    pub fn config(&self) -> RendererConfig {
        self.config
    }

    pub fn render_to_buffer(&self, scene: &Scene, target: &mut BufferTarget) {
        let cfg = RasterConfig {
            width: self.config.width,
            height: self.config.height,
        };
        let mut surf = RasterSurface::new(cfg, target);
        let camera = &scene.camera;
        rasterize(scene, camera, &mut surf);
    }

    pub fn render(&self, scene: &Scene, target: &mut BufferTarget) {
        self.render_to_buffer(scene, target);
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::{
        light::Light,
        material::Material,
        mesh::Mesh,
        renderer::{Renderer, RendererConfig},
        scene::Scene,
        targets::buffer::BufferTarget,
        transform::Transform,
    };

    #[test]
    fn smoke_triangle_renders_deterministically() {
        let mut scene = Scene::new();
        scene.add_light(Light::directional(Vec3::new(0.3, 0.7, 1.0), Vec3::ONE, 1.0));
        let mesh = Mesh::unit_triangle();
        let mat = Material::new(Vec3::ONE);
        scene.add_object(mesh, Transform { translation: Vec3::new(0.0, 0.0, 2.0), rotation: Quat::IDENTITY, scale: Vec3::ONE }, mat);

        let r = Renderer::new(RendererConfig::new(64, 32));
        let mut target = BufferTarget::new(64, 32);

        r.render(&scene, &mut target);

        let empty = BufferTarget::new(64, 32);
        let h1 = target.hash64();
        assert_ne!(h1, empty.hash64());

        let mut target2 = BufferTarget::new(64, 32);
        r.render(&scene, &mut target2);
        let h2 = target2.hash64();
        assert_eq!(h1, h2);
}
}
