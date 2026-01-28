use crate::{Camera, Light, Material, Mesh, Transform};

#[derive(Clone, Debug)]
pub struct Scene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    objects: Vec<SceneObject>,
}

#[derive(Clone, Debug)]
struct SceneObject {
    mesh: Mesh,
    material: Material,
    transform: Transform,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            lights: Vec::new(),
            objects: Vec::new(),
        }
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub fn add_object(&mut self, mesh: Mesh, material: Material, transform: Transform) -> usize {
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
