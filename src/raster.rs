use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{Camera, GBuffer, Material, Scene, Texture, TextureHandle};

use std::cell::RefCell;

#[cfg(feature = "rayon")]
use crate::gbuffer::GBufferSlicesMut;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

type ClipVert = (Vertex, Vec4);

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    nrm: Vec3,
    uv: Vec2,
    inv_w: f32,
}

#[derive(Copy, Clone, Debug)]
struct Tri {
    a: Vertex,
    b: Vertex,
    c: Vertex,
}

const TILE_SIZE: i32 = 16;
const W_CLIP_EPS: f32 = 1e-6;
const FRUSTUM_CLIP_PLANE_COUNT: usize = 7;

#[derive(Copy, Clone, Debug)]
struct TiledTri {
    a: Vertex,
    b: Vertex,
    c: Vertex,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    map_kd: Option<TextureHandle>,
}

#[derive(Copy, Clone, Debug)]
struct RasterConfig {
    width: u32,
    height: u32,
}

struct RasterSurface<'a> {
    cfg: RasterConfig,
    gbuf: &'a mut GBuffer,
}

impl<'a> RasterSurface<'a> {
    fn new(cfg: RasterConfig, gbuf: &'a mut GBuffer) -> Self {
        Self { cfg, gbuf }
    }
}

pub(crate) struct Scratch {
    clip_a: Vec<ClipVert>,
    clip_b: Vec<ClipVert>,
    tiled_tris: Vec<TiledTri>,
    tile_counts: Vec<u32>,
    tile_starts: Vec<u32>,
    tile_cursor: Vec<u32>,
    tile_items: Vec<u32>,
    grow_events: usize,
    last_grow_events: usize,
}

impl Scratch {
    fn new() -> Self {
        Self {
            clip_a: Vec::with_capacity(16),
            clip_b: Vec::with_capacity(16),
            tiled_tris: Vec::new(),
            tile_counts: Vec::new(),
            tile_starts: Vec::new(),
            tile_cursor: Vec::new(),
            tile_items: Vec::new(),
            grow_events: 0,
            last_grow_events: 0,
        }
    }

    fn begin_frame(&mut self) {
        self.grow_events = 0;
    }

    fn finish_frame(&mut self) {
        self.last_grow_events = self.grow_events;
    }
}

thread_local! {
    static TL_SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

fn project_to_screen(pos: Vec3, width: u32, height: u32) -> Vec2 {
    Vec2::new(
        (pos.x * 0.5 + 0.5) * (width.saturating_sub(1) as f32),
        (1.0 - (pos.y * 0.5 + 0.5)) * (height.saturating_sub(1) as f32),
    )
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn perspective_denom(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> f32 {
    w0 * v0.inv_w + w1 * v1.inv_w + w2 * v2.inv_w
}

fn interpolate_normal(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> Vec3 {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        ((v0.nrm * (w0 * v0.inv_w) + v1.nrm * (w1 * v1.inv_w) + v2.nrm * (w2 * v2.inv_w))
            / denom)
            .normalize_or_zero()
    } else {
        (w0 * v0.nrm + w1 * v1.nrm + w2 * v2.nrm).normalize_or_zero()
    }
}

fn interpolate_uv(
    w0: f32,
    w1: f32,
    w2: f32,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
) -> Option<Vec2> {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        Some(
            (v0.uv * (w0 * v0.inv_w) + v1.uv * (w1 * v1.inv_w) + v2.uv * (w2 * v2.inv_w))
                / denom,
        )
    } else {
        None
    }
}

fn push_checked(out: &mut Vec<ClipVert>, grow_events: &mut usize, v: ClipVert) {
    if out.len() == out.capacity() {
        *grow_events += 1;
        out.reserve(1);
    }
    out.push(v);
}

fn clip_edge(
    vert_a: Vertex,
    vert_b: Vertex,
    pos_a: Vec4,
    pos_b: Vec4,
    dist_a: f32,
    dist_b: f32,
) -> ClipVert {
    let denom = dist_a - dist_b;
    let t_param = if denom.abs() > 1e-8 {
        dist_a / denom
    } else {
        0.0
    };
    let pos = vert_a.pos + t_param * (vert_b.pos - vert_a.pos);
    let nrm = (vert_a.nrm + t_param * (vert_b.nrm - vert_a.nrm)).normalize_or_zero();
    let uv = vert_a.uv + t_param * (vert_b.uv - vert_a.uv);
    let pos_clip = pos_a + t_param * (pos_b - pos_a);
    (
        Vertex {
            pos,
            nrm,
            uv,
            inv_w: 1.0,
        },
        pos_clip,
    )
}

fn clip_plane_distance(plane: usize, p: Vec4) -> f32 {
    match plane {
        0 => p.z + p.w,
        1 => -p.z + p.w,
        2 => p.x + p.w,
        3 => -p.x + p.w,
        4 => p.y + p.w,
        5 => -p.y + p.w,
        _ => p.w - W_CLIP_EPS,
    }
}

fn draw_tri(
    surf: &mut RasterSurface,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    material: &Material,
    tex: Option<&Texture>,
) {
    let s0 = project_to_screen(v0.pos, surf.cfg.width, surf.cfg.height);
    let s1 = project_to_screen(v1.pos, surf.cfg.width, surf.cfg.height);
    let s2 = project_to_screen(v2.pos, surf.cfg.width, surf.cfg.height);

    let min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as i32;
    let max_x =
        s0.x.max(s1.x)
            .max(s2.x)
            .ceil()
            .min((surf.cfg.width as i32 - 1) as f32) as i32;
    let min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as i32;
    let max_y =
        s0.y.max(s1.y)
            .max(s2.y)
            .ceil()
            .min((surf.cfg.height as i32 - 1) as f32) as i32;

    let area = edge(s0, s1, s2);
    if area.abs() < 1e-8 {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(s1, s2, sample) / area;
            let w1 = edge(s2, s0, sample) / area;
            let w2 = edge(s0, s1, sample) / area;

            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            let depth = w0 * v0.pos.z + w1 * v1.pos.z + w2 * v2.pos.z;
            let normal = interpolate_normal(w0, w1, w2, v0, v1, v2);
            let mut kd = material.kd;
            if let Some(t) = tex {
                if let Some(uv) = interpolate_uv(w0, w1, w2, v0, v1, v2) {
                    kd *= t.sample_nearest(uv);
                }
            }

            let _ = surf.gbuf.try_write(
                x as usize,
                y as usize,
                depth,
                normal,
                kd,
                material.ks,
                material.ns,
                material.ke,
            );
        }
    }
}

#[cfg(not(feature = "rayon"))]
fn draw_tri_params_clamped(
    surf: &mut RasterSurface,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    kd_base: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    tex: Option<&Texture>,
    clamp_min_x: i32,
    clamp_max_x: i32,
    clamp_min_y: i32,
    clamp_max_y: i32,
) {
    let s0 = project_to_screen(v0.pos, surf.cfg.width, surf.cfg.height);
    let s1 = project_to_screen(v1.pos, surf.cfg.width, surf.cfg.height);
    let s2 = project_to_screen(v2.pos, surf.cfg.width, surf.cfg.height);

    let mut min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as i32;
    let mut max_x =
        s0.x.max(s1.x)
            .max(s2.x)
            .ceil()
            .min((surf.cfg.width as i32 - 1) as f32) as i32;
    let mut min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as i32;
    let mut max_y =
        s0.y.max(s1.y)
            .max(s2.y)
            .ceil()
            .min((surf.cfg.height as i32 - 1) as f32) as i32;

    min_x = min_x.max(clamp_min_x);
    max_x = max_x.min(clamp_max_x);
    min_y = min_y.max(clamp_min_y);
    max_y = max_y.min(clamp_max_y);

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge(s0, s1, s2);
    if area.abs() < 1e-8 {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(s1, s2, sample) / area;
            let w1 = edge(s2, s0, sample) / area;
            let w2 = edge(s0, s1, sample) / area;

            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            let depth = w0 * v0.pos.z + w1 * v1.pos.z + w2 * v2.pos.z;
            let normal = interpolate_normal(w0, w1, w2, v0, v1, v2);
            let mut kd = kd_base;
            if let Some(t) = tex {
                if let Some(uv) = interpolate_uv(w0, w1, w2, v0, v1, v2) {
                    kd *= t.sample_nearest(uv);
                }
            }

            let _ = surf
                .gbuf
                .try_write(x as usize, y as usize, depth, normal, kd, ks, ns, ke);
        }
    }
}

#[cfg(feature = "rayon")]
struct GBufferChunkMut<'a> {
    width: usize,
    y0: usize,
    depth: &'a mut [f32],
    nx: &'a mut [f32],
    ny: &'a mut [f32],
    nz: &'a mut [f32],
    kd: &'a mut [Vec3],
    ks: &'a mut [Vec3],
    ns: &'a mut [f32],
    ke: &'a mut [Vec3],
}

#[cfg(feature = "rayon")]
impl<'a> GBufferChunkMut<'a> {
    fn rows(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.depth.len() / self.width
        }
    }

    fn try_write(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> bool {
        if x >= self.width {
            return false;
        }
        let rows = self.rows();
        if y < self.y0 || y >= self.y0 + rows {
            return false;
        }
        let i = (y - self.y0) * self.width + x;
        if depth < self.depth[i] {
            self.depth[i] = depth;
            self.nx[i] = normal.x;
            self.ny[i] = normal.y;
            self.nz[i] = normal.z;
            self.kd[i] = kd;
            self.ks[i] = ks;
            self.ns[i] = ns;
            self.ke[i] = ke;
            true
        } else {
            false
        }
    }
}

#[cfg(feature = "rayon")]
fn draw_tri_params_clamped_chunk(
    cfg: RasterConfig,
    gbuf: &mut GBufferChunkMut<'_>,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    kd_base: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    tex: Option<&Texture>,
    clamp_min_x: i32,
    clamp_max_x: i32,
    clamp_min_y: i32,
    clamp_max_y: i32,
) {
    let s0 = project_to_screen(v0.pos, cfg.width, cfg.height);
    let s1 = project_to_screen(v1.pos, cfg.width, cfg.height);
    let s2 = project_to_screen(v2.pos, cfg.width, cfg.height);

    let mut min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as i32;
    let mut max_x =
        s0.x.max(s1.x)
            .max(s2.x)
            .ceil()
            .min((cfg.width as i32 - 1) as f32) as i32;
    let mut min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as i32;
    let mut max_y =
        s0.y.max(s1.y)
            .max(s2.y)
            .ceil()
            .min((cfg.height as i32 - 1) as f32) as i32;

    min_x = min_x.max(clamp_min_x);
    max_x = max_x.min(clamp_max_x);
    min_y = min_y.max(clamp_min_y);
    max_y = max_y.min(clamp_max_y);

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge(s0, s1, s2);
    if area.abs() < 1e-8 {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(s1, s2, sample) / area;
            let w1 = edge(s2, s0, sample) / area;
            let w2 = edge(s0, s1, sample) / area;

            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            let depth = w0 * v0.pos.z + w1 * v1.pos.z + w2 * v2.pos.z;
            let normal = interpolate_normal(w0, w1, w2, v0, v1, v2);
            let mut kd = kd_base;
            if let Some(t) = tex {
                if let Some(uv) = interpolate_uv(w0, w1, w2, v0, v1, v2) {
                    kd *= t.sample_nearest(uv);
                }
            }

            let _ = gbuf.try_write(x as usize, y as usize, depth, normal, kd, ks, ns, ke);
        }
    }
}

#[cfg(feature = "rayon")]
fn render_tiles_parallel(
    scene: &Scene,
    cfg: RasterConfig,
    gbuf: GBufferSlicesMut<'_>,
    tiled_tris: &[TiledTri],
    tile_starts: &[u32],
    tile_items: &[u32],
    tiles_x: i32,
    tiles_y: i32,
) {
    let width = cfg.width as usize;
    let height = cfg.height as usize;
    let chunk_rows = TILE_SIZE as usize;
    let row_stride = width * chunk_rows;

    let iter = gbuf
        .depth
        .par_chunks_mut(row_stride)
        .zip(gbuf.nx.par_chunks_mut(row_stride))
        .zip(gbuf.ny.par_chunks_mut(row_stride))
        .zip(gbuf.nz.par_chunks_mut(row_stride))
        .zip(gbuf.kd.par_chunks_mut(row_stride))
        .zip(gbuf.ks.par_chunks_mut(row_stride))
        .zip(gbuf.ns.par_chunks_mut(row_stride))
        .zip(gbuf.ke.par_chunks_mut(row_stride));

    iter.enumerate().for_each(
        |(ty, (((((((depth_c, nx_c), ny_c), nz_c), kd_c), ks_c), ns_c), ke_c))| {
            let y0 = ty * chunk_rows;
            if y0 >= height {
                return;
            }
            let y1 = (y0 + chunk_rows).min(height);
            let mut chunk = GBufferChunkMut {
                width,
                y0,
                depth: depth_c,
                nx: nx_c,
                ny: ny_c,
                nz: nz_c,
                kd: kd_c,
                ks: ks_c,
                ns: ns_c,
                ke: ke_c,
            };

            let ty_i32 = ty as i32;
            if ty_i32 >= tiles_y {
                return;
            }

            for tx in 0..tiles_x {
                let tile_id = ty_i32 * tiles_x + tx;
                if tile_id < 0 {
                    continue;
                }
                let tile_id = tile_id as usize;
                if tile_id + 1 >= tile_starts.len() {
                    continue;
                }

                let tile_min_x = tx * TILE_SIZE;
                let tile_max_x = ((tx + 1) * TILE_SIZE - 1).min(cfg.width as i32 - 1);
                let tile_min_y = (ty_i32 * TILE_SIZE).max(y0 as i32);
                let tile_max_y =
                    (((ty_i32 + 1) * TILE_SIZE - 1).min(cfg.height as i32 - 1)).min(y1 as i32 - 1);
                if tile_min_y > tile_max_y {
                    continue;
                }

                let start = tile_starts[tile_id] as usize;
                let end = tile_starts[tile_id + 1] as usize;
                for i in start..end {
                    let tri = tiled_tris[tile_items[i] as usize];
                    let tex = tri.map_kd.and_then(|h| scene.texture(h));
                    draw_tri_params_clamped_chunk(
                        cfg, &mut chunk, tri.a, tri.b, tri.c, tri.kd, tri.ks, tri.ns, tri.ke, tex,
                        tile_min_x, tile_max_x, tile_min_y, tile_max_y,
                    );
                }
            }
        },
    );
}

fn rasterize_tri_collect(
    _cam: &Camera,
    clip_a: &mut Vec<ClipVert>,
    clip_b: &mut Vec<ClipVert>,
    grow_events: &mut usize,
    tri: &Tri,
    mv: Mat4,
    mvp: Mat4,
    nm: Mat4,
    material: &Material,
    out: &mut Vec<TiledTri>,
) {
    let mut v0 = Vertex {
        pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.a.nrm.extend(0.0)).truncate(),
        uv: tri.a.uv,
        inv_w: 1.0,
    };
    let mut v1 = Vertex {
        pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.b.nrm.extend(0.0)).truncate(),
        uv: tri.b.uv,
        inv_w: 1.0,
    };
    let mut v2 = Vertex {
        pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.c.nrm.extend(0.0)).truncate(),
        uv: tri.c.uv,
        inv_w: 1.0,
    };

    v0.nrm = v0.nrm.normalize_or_zero();
    v1.nrm = v1.nrm.normalize_or_zero();
    v2.nrm = v2.nrm.normalize_or_zero();

    let v0c = mvp * tri.a.pos.extend(1.0);
    let v1c = mvp * tri.b.pos.extend(1.0);
    let v2c = mvp * tri.c.pos.extend(1.0);

    clip_a.clear();
    clip_b.clear();
    clip_a.push((v0, v0c));
    clip_a.push((v1, v1c));
    clip_a.push((v2, v2c));

    let mut in_is_a = true;
    for plane in 0..FRUSTUM_CLIP_PLANE_COUNT {
        let (in_buf, out_buf): (&Vec<ClipVert>, &mut Vec<ClipVert>) = if in_is_a {
            (&*clip_a, clip_b)
        } else {
            (&*clip_b, clip_a)
        };
        out_buf.clear();

        if in_buf.is_empty() {
            in_is_a = !in_is_a;
            break;
        }

        let n = in_buf.len();
        for i in 0..n {
            let (va, pa) = in_buf[i];
            let (vb, pb) = in_buf[(i + 1) % n];

            let da = clip_plane_distance(plane, pa);
            let db = clip_plane_distance(plane, pb);

            let ina = da >= 0.0;
            let inb = db >= 0.0;

            if ina && inb {
                push_checked(out_buf, grow_events, (vb, pb));
            } else if ina && !inb {
                push_checked(out_buf, grow_events, clip_edge(va, vb, pa, pb, da, db));
            } else if !ina && inb {
                push_checked(out_buf, grow_events, clip_edge(va, vb, pa, pb, da, db));
                push_checked(out_buf, grow_events, (vb, pb));
            }
        }

        in_is_a = !in_is_a;
    }

    let verts: &Vec<ClipVert> = if in_is_a { &*clip_a } else { &*clip_b };
    if verts.len() < 3 {
        return;
    }

    for i in 1..(verts.len() - 1) {
        let (a, pa) = verts[0];
        let (b, pb) = verts[i];
        let (c, pc) = verts[i + 1];

        let mut aa = a;
        let mut bb = b;
        let mut cc = c;

        if !(pa.w.is_finite() && pb.w.is_finite() && pc.w.is_finite()) {
            continue;
        }
        if pa.w <= W_CLIP_EPS || pb.w <= W_CLIP_EPS || pc.w <= W_CLIP_EPS {
            continue;
        }

        aa.inv_w = 1.0 / pa.w;
        bb.inv_w = 1.0 / pb.w;
        cc.inv_w = 1.0 / pc.w;

        aa.pos = (pa / pa.w).truncate();
        bb.pos = (pb / pb.w).truncate();
        cc.pos = (pc / pc.w).truncate();

        out.push(TiledTri {
            a: aa,
            b: bb,
            c: cc,
            kd: material.kd,
            ks: material.ks,
            ns: material.ns,
            ke: material.ke,
            map_kd: material.map_kd,
        });
    }
}

fn tri_bbox_pixels(
    cfg: RasterConfig,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(i32, i32, i32, i32)> {
    let s0 = project_to_screen(v0, cfg.width, cfg.height);
    let s1 = project_to_screen(v1, cfg.width, cfg.height);
    let s2 = project_to_screen(v2, cfg.width, cfg.height);

    let min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as i32;
    let max_x =
        s0.x.max(s1.x)
            .max(s2.x)
            .ceil()
            .min((cfg.width as i32 - 1) as f32) as i32;
    let min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as i32;
    let max_y =
        s0.y.max(s1.y)
            .max(s2.y)
            .ceil()
            .min((cfg.height as i32 - 1) as f32) as i32;

    if min_x > max_x || min_y > max_y {
        return None;
    }

    let area = edge(s0, s1, s2);
    if area.abs() < 1e-8 {
        return None;
    }

    Some((min_x, max_x, min_y, max_y))
}

fn rasterize_tiled(scene: &Scene, cam: &Camera, surf: &mut RasterSurface, scratch: &mut Scratch) {
    if surf.cfg.width == 0 || surf.cfg.height == 0 {
        return;
    }
    scratch.tiled_tris.clear();
    {
        let (clip_a, clip_b, tiled_tris, grow_events) = (
            &mut scratch.clip_a,
            &mut scratch.clip_b,
            &mut scratch.tiled_tris,
            &mut scratch.grow_events,
        );

        for (mesh, material, transform) in scene.iter_objects() {
            let mv = cam.view_matrix() * transform.to_mat4();
            let aspect = surf.cfg.width as f32 / (surf.cfg.height.max(1) as f32);
            let mvp = cam.projection_matrix(aspect) * mv;
            let nm = mv.inverse().transpose();

            for idx in mesh.indices.iter() {
                let i0 = idx[0] as usize;
                let i1 = idx[1] as usize;
                let i2 = idx[2] as usize;

                if i0 >= mesh.positions.len()
                    || i1 >= mesh.positions.len()
                    || i2 >= mesh.positions.len()
                {
                    continue;
                }

                let n0 = mesh.normals.get(i0).copied().unwrap_or(Vec3::Z);
                let n1 = mesh.normals.get(i1).copied().unwrap_or(Vec3::Z);
                let n2 = mesh.normals.get(i2).copied().unwrap_or(Vec3::Z);

                let uv0 = mesh.uvs.get(i0).copied().unwrap_or(Vec2::ZERO);
                let uv1 = mesh.uvs.get(i1).copied().unwrap_or(Vec2::ZERO);
                let uv2 = mesh.uvs.get(i2).copied().unwrap_or(Vec2::ZERO);

                let tri = Tri {
                    a: Vertex {
                        pos: mesh.positions[i0],
                        nrm: n0,
                        uv: uv0,
                        inv_w: 1.0,
                    },
                    b: Vertex {
                        pos: mesh.positions[i1],
                        nrm: n1,
                        uv: uv1,
                        inv_w: 1.0,
                    },
                    c: Vertex {
                        pos: mesh.positions[i2],
                        nrm: n2,
                        uv: uv2,
                        inv_w: 1.0,
                    },
                };

                rasterize_tri_collect(
                    cam,
                    clip_a,
                    clip_b,
                    grow_events,
                    &tri,
                    mv,
                    mvp,
                    nm,
                    material,
                    tiled_tris,
                );
            }
        }
    }

    let width = surf.cfg.width as i32;
    let height = surf.cfg.height as i32;

    if width <= 0 || height <= 0 {
        return;
    }

    let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    let total_tiles = (tiles_x * tiles_y) as usize;

    scratch.tile_counts.resize(total_tiles, 0);
    scratch.tile_starts.resize(total_tiles + 1, 0);
    scratch.tile_cursor.resize(total_tiles, 0);

    for c in scratch.tile_counts.iter_mut() {
        *c = 0;
    }

    if scratch.tiled_tris.len() > (u32::MAX as usize) {
        rasterize(scene, cam, surf, scratch);
        return;
    }

    let mut overflowed = false;

    for tri in scratch.tiled_tris.iter() {
        let Some((min_x, max_x, min_y, max_y)) =
            tri_bbox_pixels(surf.cfg, tri.a.pos, tri.b.pos, tri.c.pos)
        else {
            continue;
        };
        let tx0 = (min_x / TILE_SIZE).clamp(0, tiles_x - 1);
        let tx1 = (max_x / TILE_SIZE).clamp(0, tiles_x - 1);
        let ty0 = (min_y / TILE_SIZE).clamp(0, tiles_y - 1);
        let ty1 = (max_y / TILE_SIZE).clamp(0, tiles_y - 1);

        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let id = (ty * tiles_x + tx) as usize;
                let v = scratch.tile_counts[id];
                if v == u32::MAX {
                    overflowed = true;
                } else {
                    scratch.tile_counts[id] = v + 1;
                }
            }
        }
    }

    if overflowed {
        rasterize(scene, cam, surf, scratch);
        return;
    }

    let mut sum: u32 = 0;
    for i in 0..total_tiles {
        scratch.tile_starts[i] = sum;
        let Some(next) = sum.checked_add(scratch.tile_counts[i]) else {
            overflowed = true;
            break;
        };
        sum = next;
        scratch.tile_cursor[i] = scratch.tile_starts[i];
    }
    if overflowed {
        rasterize(scene, cam, surf, scratch);
        return;
    }
    scratch.tile_starts[total_tiles] = sum;

    scratch.tile_items.resize(sum as usize, 0);

    'bin_fill: for (tri_idx, tri) in scratch.tiled_tris.iter().enumerate() {
        let Some((min_x, max_x, min_y, max_y)) =
            tri_bbox_pixels(surf.cfg, tri.a.pos, tri.b.pos, tri.c.pos)
        else {
            continue;
        };
        let tx0 = (min_x / TILE_SIZE).clamp(0, tiles_x - 1);
        let tx1 = (max_x / TILE_SIZE).clamp(0, tiles_x - 1);
        let ty0 = (min_y / TILE_SIZE).clamp(0, tiles_y - 1);
        let ty1 = (max_y / TILE_SIZE).clamp(0, tiles_y - 1);

        let tri_u32 = u32::try_from(tri_idx).unwrap();

        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let id = (ty * tiles_x + tx) as usize;
                let pos = scratch.tile_cursor[id] as usize;
                if pos >= scratch.tile_items.len() {
                    overflowed = true;
                    break 'bin_fill;
                }
                scratch.tile_items[pos] = tri_u32;
                let Some(next) = scratch.tile_cursor[id].checked_add(1) else {
                    overflowed = true;
                    break 'bin_fill;
                };
                scratch.tile_cursor[id] = next;
            }
        }
    }

    if overflowed {
        rasterize(scene, cam, surf, scratch);
        return;
    }

    #[cfg(feature = "rayon")]
    {
        let gbuf = surf.gbuf.slices_mut();
        render_tiles_parallel(
            scene,
            surf.cfg,
            gbuf,
            &scratch.tiled_tris,
            &scratch.tile_starts,
            &scratch.tile_items,
            tiles_x,
            tiles_y,
        );
    }

    #[cfg(not(feature = "rayon"))]
    {
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let id = (ty * tiles_x + tx) as usize;
                let start = scratch.tile_starts[id] as usize;
                let end = scratch.tile_starts[id + 1] as usize;

                if start >= end || end > scratch.tile_items.len() {
                    continue;
                }

                let clamp_min_x = tx * TILE_SIZE;
                let clamp_min_y = ty * TILE_SIZE;
                let clamp_max_x = ((tx + 1) * TILE_SIZE - 1).min(width - 1);
                let clamp_max_y = ((ty + 1) * TILE_SIZE - 1).min(height - 1);

                for k in start..end {
                    let tri_id = scratch.tile_items[k] as usize;
                    if tri_id >= scratch.tiled_tris.len() {
                        continue;
                    }
                    let t = scratch.tiled_tris[tri_id];
                    let tex = t.map_kd.and_then(|h| scene.texture(h));

                    draw_tri_params_clamped(
                        surf,
                        t.a,
                        t.b,
                        t.c,
                        t.kd,
                        t.ks,
                        t.ns,
                        t.ke,
                        tex,
                        clamp_min_x,
                        clamp_max_x,
                        clamp_min_y,
                        clamp_max_y,
                    );
                }
            }
        }
    }
}

fn rasterize_tri(
    surf: &mut RasterSurface,
    _cam: &Camera,
    scratch: &mut Scratch,
    tri: &Tri,
    mv: Mat4,
    mvp: Mat4,
    nm: Mat4,
    material: &Material,
    tex: Option<&Texture>,
) {
    let mut v0 = Vertex {
        pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.a.nrm.extend(0.0)).truncate(),
        uv: tri.a.uv,
        inv_w: 1.0,
    };
    let mut v1 = Vertex {
        pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.b.nrm.extend(0.0)).truncate(),
        uv: tri.b.uv,
        inv_w: 1.0,
    };
    let mut v2 = Vertex {
        pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.c.nrm.extend(0.0)).truncate(),
        uv: tri.c.uv,
        inv_w: 1.0,
    };

    v0.nrm = v0.nrm.normalize_or_zero();
    v1.nrm = v1.nrm.normalize_or_zero();
    v2.nrm = v2.nrm.normalize_or_zero();

    let v0c = mvp * tri.a.pos.extend(1.0);
    let v1c = mvp * tri.b.pos.extend(1.0);
    let v2c = mvp * tri.c.pos.extend(1.0);

    scratch.clip_a.clear();
    scratch.clip_b.clear();
    scratch.clip_a.push((v0, v0c));
    scratch.clip_a.push((v1, v1c));
    scratch.clip_a.push((v2, v2c));

    let mut in_is_a = true;
    for plane in 0..FRUSTUM_CLIP_PLANE_COUNT {
        let (in_buf, out_buf) = if in_is_a {
            (&scratch.clip_a, &mut scratch.clip_b)
        } else {
            (&scratch.clip_b, &mut scratch.clip_a)
        };
        out_buf.clear();

        if in_buf.is_empty() {
            in_is_a = !in_is_a;
            break;
        }

        let n = in_buf.len();
        for i in 0..n {
            let (va, pa) = in_buf[i];
            let (vb, pb) = in_buf[(i + 1) % n];

            let da = clip_plane_distance(plane, pa);
            let db = clip_plane_distance(plane, pb);

            let ina = da >= 0.0;
            let inb = db >= 0.0;

            if ina && inb {
                push_checked(out_buf, &mut scratch.grow_events, (vb, pb));
            } else if ina && !inb {
                push_checked(
                    out_buf,
                    &mut scratch.grow_events,
                    clip_edge(va, vb, pa, pb, da, db),
                );
            } else if !ina && inb {
                push_checked(
                    out_buf,
                    &mut scratch.grow_events,
                    clip_edge(va, vb, pa, pb, da, db),
                );
                push_checked(out_buf, &mut scratch.grow_events, (vb, pb));
            }
        }

        in_is_a = !in_is_a;
    }

    let verts = if in_is_a {
        &scratch.clip_a
    } else {
        &scratch.clip_b
    };
    if verts.len() < 3 {
        return;
    }

    for i in 1..(verts.len() - 1) {
        let (a, pa) = verts[0];
        let (b, pb) = verts[i];
        let (c, pc) = verts[i + 1];

        let mut aa = a;
        let mut bb = b;
        let mut cc = c;

        if !(pa.w.is_finite() && pb.w.is_finite() && pc.w.is_finite()) {
            continue;
        }
        if pa.w <= W_CLIP_EPS || pb.w <= W_CLIP_EPS || pc.w <= W_CLIP_EPS {
            continue;
        }

        aa.inv_w = 1.0 / pa.w;
        bb.inv_w = 1.0 / pb.w;
        cc.inv_w = 1.0 / pc.w;

        aa.pos = (pa / pa.w).truncate();
        bb.pos = (pb / pb.w).truncate();
        cc.pos = (pc / pc.w).truncate();

        draw_tri(surf, aa, bb, cc, material, tex);
    }
}

fn rasterize(scene: &Scene, cam: &Camera, surf: &mut RasterSurface, scratch: &mut Scratch) {
    if surf.cfg.width == 0 || surf.cfg.height == 0 {
        return;
    }
    for (mesh, material, transform) in scene.iter_objects() {
        let mv = cam.view_matrix() * transform.to_mat4();
        let aspect = surf.cfg.width as f32 / (surf.cfg.height.max(1) as f32);
        let mvp = cam.projection_matrix(aspect) * mv;
        let nm = mv.inverse().transpose();

        let tex = material.map_kd.and_then(|h| scene.texture(h));

        for idx in mesh.indices.iter() {
            let i0 = idx[0] as usize;
            let i1 = idx[1] as usize;
            let i2 = idx[2] as usize;

            if i0 >= mesh.positions.len()
                || i1 >= mesh.positions.len()
                || i2 >= mesh.positions.len()
            {
                continue;
            }

            let n0 = mesh.normals.get(i0).copied().unwrap_or(Vec3::Z);
            let n1 = mesh.normals.get(i1).copied().unwrap_or(Vec3::Z);
            let n2 = mesh.normals.get(i2).copied().unwrap_or(Vec3::Z);

            let uv0 = mesh.uvs.get(i0).copied().unwrap_or(Vec2::ZERO);
            let uv1 = mesh.uvs.get(i1).copied().unwrap_or(Vec2::ZERO);
            let uv2 = mesh.uvs.get(i2).copied().unwrap_or(Vec2::ZERO);

            let tri = Tri {
                a: Vertex {
                    pos: mesh.positions[i0],
                    nrm: n0,
                    uv: uv0,
                    inv_w: 1.0,
                },
                b: Vertex {
                    pos: mesh.positions[i1],
                    nrm: n1,
                    uv: uv1,
                    inv_w: 1.0,
                },
                c: Vertex {
                    pos: mesh.positions[i2],
                    nrm: n2,
                    uv: uv2,
                    inv_w: 1.0,
                },
            };

            rasterize_tri(surf, cam, scratch, &tri, mv, mvp, nm, material, tex);
        }
    }
}

fn render_to_gbuffer_with_scratch(scene: &Scene, gbuf: &mut GBuffer, scratch: &mut Scratch) {
    let cfg = RasterConfig {
        width: gbuf.width() as u32,
        height: gbuf.height() as u32,
    };
    gbuf.clear();
    scratch.begin_frame();
    let mut surf = RasterSurface::new(cfg, gbuf);
    rasterize(scene, &scene.camera, &mut surf, scratch);
    scratch.finish_frame();
}

fn render_to_gbuffer_tiled_with_scratch(scene: &Scene, gbuf: &mut GBuffer, scratch: &mut Scratch) {
    let cfg = RasterConfig {
        width: gbuf.width() as u32,
        height: gbuf.height() as u32,
    };
    gbuf.clear();
    scratch.begin_frame();
    let mut surf = RasterSurface::new(cfg, gbuf);
    rasterize_tiled(scene, &scene.camera, &mut surf, scratch);
    scratch.finish_frame();
}

pub fn render_to_gbuffer_tiled(scene: &Scene, gbuf: &mut GBuffer) {
    TL_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        render_to_gbuffer_tiled_with_scratch(scene, gbuf, &mut scratch);
    });
}

pub fn render_to_gbuffer(scene: &Scene, gbuf: &mut GBuffer) {
    TL_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        render_to_gbuffer_with_scratch(scene, gbuf, &mut scratch);
    });
}

#[cfg(test)]
pub(crate) fn scratch_grow_events_last_frame() -> usize {
    TL_SCRATCH.with(|s| s.borrow().last_grow_events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GBuffer, Material, Mesh, Scene, Transform};

    #[test]
    fn pr21_raster_clip_scratch_does_not_grow() {
        let mut scene = Scene::new();
        let mesh = Mesh::unit_cube();
        scene.add_object(mesh, Transform::default(), Material::default());

        let mut gbuf = GBuffer::new(64, 64);
        render_to_gbuffer(&scene, &mut gbuf);

        assert_eq!(scratch_grow_events_last_frame(), 0);
    }
}
