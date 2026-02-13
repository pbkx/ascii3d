mod common;

use ascii3d::gbuffer::GBuffer;
use ascii3d::raster;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn raster_tiled(c: &mut Criterion) {
    let scene = common::make_scene();
    let mut gbuf = GBuffer::new(common::WIDTH, common::HEIGHT);

    c.bench_function("raster/tiled", |b| {
        b.iter(|| {
            raster::render_to_gbuffer_tiled(black_box(&scene), black_box(&mut gbuf));
            black_box(gbuf.hash64());
        })
    });
}

criterion_group!(benches, raster_tiled);
criterion_main!(benches);
