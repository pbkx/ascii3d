use ascii3d::{
    prelude::{Material, Mesh, Renderer, RendererConfig, Scene, Transform},
    targets::{BufferTarget, ImageTarget},
};

fn main() {
    let mut scene = Scene::new();
    scene.add_object(
        Mesh::unit_triangle(),
        Transform::IDENTITY,
        Material::default(),
    );

    let renderer = Renderer::new(RendererConfig::default().with_size(80, 40));

    let mut buffer = BufferTarget::new(80, 40);
    renderer.render(&scene, &mut buffer);

    let mut image = ImageTarget::new(80, 40);
    renderer.render_image(&scene, &mut image);

    println!("buffer hash64: {}", buffer.hash64());
    println!("image  hash64: {}", image.hash64());
}
