use ascii3d::camera::{Camera, Projection};
use ascii3d::light::Light;
use ascii3d::material::Material;
use ascii3d::mesh::Mesh;
use ascii3d::scene::Scene;
use ascii3d::transform::Transform;

use glam::Vec3;

pub const WIDTH: usize = 320;
pub const HEIGHT: usize = 180;

pub fn make_scene() -> Scene {
    let proj = Projection::Perspective {
        fov_y_radians: 60.0_f32.to_radians(),
        near: 0.05,
        far: 100.0,
    };

    // Default orientation looks down -Z; place the camera a bit away from origin.
    let cam = Camera::new(Transform::from_translation(Vec3::new(0.0, 0.0, 4.5)), proj);

    let mut scene = Scene::with_camera(cam);
    scene.add_light(Light::Directional {
        direction: Vec3::new(-0.4, -0.7, -0.6).normalize(),
        color: Vec3::ONE,
        intensity: 1.0,
    });

    let cube = Mesh::unit_cube();
    let mat = Material::default();

    // A small grid of cubes to make raster+glyph work non-trivial but stable.
    // (Keep object count moderate so benches run quickly on CI.)
    let grid = 4;
    let spacing = 1.1_f32;
    for y in 0..grid {
        for x in 0..grid {
            let tx = (x as f32 - (grid as f32 - 1.0) * 0.5) * spacing;
            let ty = (y as f32 - (grid as f32 - 1.0) * 0.5) * spacing;
            let t = Transform::from_translation(Vec3::new(tx, ty, 0.0));
            scene.add_object(cube.clone(), t, mat.clone());
        }
    }

    scene
}
