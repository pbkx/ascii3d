mod common;

use ascii3d::targets::ImageTarget;
use ascii3d::{Renderer, RendererConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn render_end_to_end(c: &mut Criterion) {
    let scene = common::make_scene();
    let r = Renderer::new(RendererConfig::new(common::WIDTH, common::HEIGHT));
    let mut out = ImageTarget::new(common::WIDTH, common::HEIGHT);

    c.bench_function("render/end_to_end", |b| {
        b.iter(|| {
            r.render_image(black_box(&scene), &mut out);
            black_box(out.hash64());
        })
    });
}

criterion_group!(benches, render_end_to_end);
criterion_main!(benches);
