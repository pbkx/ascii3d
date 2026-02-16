use crate::texture::{Texture, TextureHandle};
use crate::{Camera, Light, Material, Mesh, Transform};

#[derive(Clone, Debug)]
pub struct Scene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    objects: Vec<SceneObject>,
    textures: Vec<Texture>,
}

#[derive(Clone, Debug)]
struct SceneObject {
    mesh: Mesh,
    material: Material,
    transform: Transform,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            lights: Vec::new(),
            objects: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn with_camera(camera: Camera) -> Self {
        Self {
            camera,
            lights: Vec::new(),
            objects: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub fn add_texture(&mut self, texture: Texture) -> TextureHandle {
        let handle = TextureHandle(self.textures.len());
        self.textures.push(texture);
        handle
    }

    pub fn texture(&self, handle: TextureHandle) -> Option<&Texture> {
        self.textures.get(handle.0)
    }

    pub fn add_object(&mut self, mesh: Mesh, transform: Transform, material: Material) -> usize {
        let idx = self.objects.len();
        self.objects.push(SceneObject {
            mesh,
            material,
            transform,
        });
        idx
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub fn iter_objects(&self) -> impl Iterator<Item = (&Mesh, &Material, &Transform)> {
        self.objects
            .iter()
            .map(|o| (&o.mesh, &o.material, &o.transform))
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
