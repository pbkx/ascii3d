mod common;

use ascii3d::gbuffer::GBuffer;
use ascii3d::raster;
use ascii3d::targets::BufferTarget;
use ascii3d::{Renderer, RendererConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn glyph_resolve(c: &mut Criterion) {
    let scene = common::make_scene();

    // Fill the gbuffer once; the benchmark focuses on resolving to glyphs.
    let mut gbuf = GBuffer::new(common::WIDTH, common::HEIGHT);
    raster::render_to_gbuffer(&scene, &mut gbuf);

    let renderer = Renderer::new(RendererConfig::new(common::WIDTH, common::HEIGHT));
    let mut out = BufferTarget::new(common::WIDTH, common::HEIGHT);

    c.bench_function("glyph/resolve", |b| {
        b.iter(|| {
            renderer.resolve_gbuffer_to_buffer(black_box(&scene), black_box(&gbuf), black_box(&mut out));
            black_box(out.hash64());
        })
    });
}

criterion_group!(benches, glyph_resolve);
criterion_main!(benches);
