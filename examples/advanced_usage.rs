use ascii3d::{
    prelude::{
        DebugView, DitherMode, Light, Material, Mesh, Quat, Renderer, RendererConfig, Scene,
        Transform, Vec3,
    },
    targets::ImageTarget,
};

fn main() {
    let mut scene = Scene::new();
    scene.add_light(Light::directional(Vec3::new(0.3, 0.7, 1.0), Vec3::ONE, 1.0));

    let mut cube_mat = Material::new(Vec3::new(0.7, 0.85, 1.0));
    cube_mat.ks = Vec3::splat(0.4);
    cube_mat.ns = 32.0;
    scene.add_object(
        Mesh::unit_cube(),
        Transform::from_rotation(Quat::from_rotation_y(0.65) * Quat::from_rotation_x(-0.35)),
        cube_mat,
    );

    scene.add_object(
        Mesh::unit_triangle(),
        Transform::from_translation(Vec3::new(0.55, -0.35, 0.15)),
        Material::new(Vec3::new(1.0, 0.45, 0.25)),
    );

    let config = RendererConfig::default()
        .with_size(160, 90)
        .with_debug_view(DebugView::Final)
        .with_dither_mode(DitherMode::BlueNoise)
        .with_tile_binning(true)
        .with_tone_map(true)
        .with_contrast(1.1)
        .with_edge_enhance(0.2);

    let renderer = Renderer::new(config);
    let mut image = ImageTarget::new(160, 90);
    let stats = renderer.render_image_with_stats(&scene, &mut image);

    println!("{}", stats.overlay_text());
    println!("image hash64: {}", image.hash64());
}
