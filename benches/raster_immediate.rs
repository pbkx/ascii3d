mod common;

use ascii3d::gbuffer::GBuffer;
use ascii3d::raster;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn raster_immediate(c: &mut Criterion) {
    let scene = common::make_scene();
    let mut gbuf = GBuffer::new(common::WIDTH, common::HEIGHT);

    c.bench_function("raster/immediate", |b| {
        b.iter(|| {
            gbuf.clear();
            raster::render_to_gbuffer(black_box(&scene), &mut gbuf);
            black_box(gbuf.hash64())
        })
    });
}

criterion_group!(benches, raster_immediate);
criterion_main!(benches);
