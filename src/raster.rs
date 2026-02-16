use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::texture::TextureSampleFootprint;
use crate::{AlphaMode, Camera, GBuffer, Material, Scene, Texture, TextureHandle};

use std::{cell::RefCell, cmp::Ordering};

#[cfg(feature = "rayon")]
use crate::gbuffer::GBufferSlicesMut;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

type ClipVert = (Vertex, Vec4);

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    view_pos: Vec3,
    nrm: Vec3,
    color: Vec4,
    uv0: Vec2,
    uv1: Vec2,
    tangent: Vec4,
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
    alpha: f32,
    alpha_mode: AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    metallic: f32,
    roughness: f32,
    map_kd_texcoord_set: usize,
    map_kd_uv_transform: Mat3,
    map_kd: Option<TextureHandle>,
    map_normal_texcoord_set: usize,
    map_normal_uv_transform: Mat3,
    map_normal_scale: f32,
    map_normal: Option<TextureHandle>,
    map_occlusion_texcoord_set: usize,
    map_occlusion_uv_transform: Mat3,
    map_occlusion_strength: f32,
    map_occlusion: Option<TextureHandle>,
    map_emissive_texcoord_set: usize,
    map_emissive_uv_transform: Mat3,
    map_emissive: Option<TextureHandle>,
    map_metallic_roughness_texcoord_set: usize,
    map_metallic_roughness_uv_transform: Mat3,
    map_metallic_roughness: Option<TextureHandle>,
    pbr_metallic_roughness: bool,
}

#[derive(Copy, Clone, Debug)]
struct RasterConfig {
    width: u32,
    height: u32,
    camera_aspect: f32,
}

struct RasterSurface<'a> {
    cfg: RasterConfig,
    gbuf: &'a mut GBuffer,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendSample {
    pub x: usize,
    pub y: usize,
    pub depth: f32,
    pub view_pos: Vec3,
    pub normal: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub ns: f32,
    pub ke: Vec3,
    pub alpha: f32,
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
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    Vec2::new((pos.x * 0.5 + 0.5) * w, (1.0 - (pos.y * 0.5 + 0.5)) * h)
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn perspective_denom(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> f32 {
    w0 * v0.inv_w + w1 * v1.inv_w + w2 * v2.inv_w
}

fn uv_for_set(v: Vertex, texcoord_set: usize) -> Vec2 {
    match texcoord_set {
        1 => v.uv1,
        _ => v.uv0,
    }
}

fn interpolate_normal(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> Vec3 {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        ((v0.nrm * (w0 * v0.inv_w) + v1.nrm * (w1 * v1.inv_w) + v2.nrm * (w2 * v2.inv_w)) / denom)
            .normalize_or_zero()
    } else {
        (w0 * v0.nrm + w1 * v1.nrm + w2 * v2.nrm).normalize_or_zero()
    }
}

fn interpolate_uv_set(
    w0: f32,
    w1: f32,
    w2: f32,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    texcoord_set: usize,
) -> Option<Vec2> {
    let uv0 = uv_for_set(v0, texcoord_set);
    let uv1 = uv_for_set(v1, texcoord_set);
    let uv2 = uv_for_set(v2, texcoord_set);
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        Some((uv0 * (w0 * v0.inv_w) + uv1 * (w1 * v1.inv_w) + uv2 * (w2 * v2.inv_w)) / denom)
    } else {
        None
    }
}

fn interpolate_color(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> Vec4 {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        (v0.color * (w0 * v0.inv_w) + v1.color * (w1 * v1.inv_w) + v2.color * (w2 * v2.inv_w))
            / denom
    } else {
        v0.color * w0 + v1.color * w1 + v2.color * w2
    }
}

fn interpolate_tangent(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> Vec4 {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        (v0.tangent * (w0 * v0.inv_w) + v1.tangent * (w1 * v1.inv_w) + v2.tangent * (w2 * v2.inv_w))
            / denom
    } else {
        v0.tangent * w0 + v1.tangent * w1 + v2.tangent * w2
    }
}

fn transform_uv(uv: Vec2, uv_transform: Mat3) -> Vec2 {
    let t = uv_transform * uv.extend(1.0);
    Vec2::new(t.x, t.y)
}

fn interpolate_view_pos(w0: f32, w1: f32, w2: f32, v0: Vertex, v1: Vertex, v2: Vertex) -> Vec3 {
    let denom = perspective_denom(w0, w1, w2, v0, v1, v2);
    if denom.abs() > 1e-8 {
        (v0.view_pos * (w0 * v0.inv_w)
            + v1.view_pos * (w1 * v1.inv_w)
            + v2.view_pos * (w2 * v2.inv_w))
            / denom
    } else {
        w0 * v0.view_pos + w1 * v1.view_pos + w2 * v2.view_pos
    }
}

fn uv_footprint(
    sample: Vec2,
    area: f32,
    s0: Vec2,
    s1: Vec2,
    s2: Vec2,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    uv: Vec2,
    texcoord_set: usize,
    uv_transform: Mat3,
) -> Option<TextureSampleFootprint> {
    let sample_dx = sample + Vec2::new(1.0, 0.0);
    let wx0 = edge(s1, s2, sample_dx) / area;
    let wx1 = edge(s2, s0, sample_dx) / area;
    let wx2 = edge(s0, s1, sample_dx) / area;
    let uv_dx = transform_uv(
        interpolate_uv_set(wx0, wx1, wx2, v0, v1, v2, texcoord_set)?,
        uv_transform,
    );

    let sample_dy = sample + Vec2::new(0.0, 1.0);
    let wy0 = edge(s1, s2, sample_dy) / area;
    let wy1 = edge(s2, s0, sample_dy) / area;
    let wy2 = edge(s0, s1, sample_dy) / area;
    let uv_dy = transform_uv(
        interpolate_uv_set(wy0, wy1, wy2, v0, v1, v2, texcoord_set)?,
        uv_transform,
    );
    let uv = transform_uv(uv, uv_transform);

    Some(TextureSampleFootprint {
        ddx_uv: uv_dx - uv,
        ddy_uv: uv_dy - uv,
    })
}

fn is_top_left_edge(a: Vec2, b: Vec2) -> bool {
    let dy = b.y - a.y;
    let dx = b.x - a.x;
    dy > 0.0 || (dy.abs() <= 1e-8 && dx < 0.0)
}

fn edge_is_inside(e: f32, a: Vec2, b: Vec2, area_sign: f32) -> bool {
    let ee = e * area_sign;
    if ee > 1e-8 {
        true
    } else if ee >= -1e-8 {
        if area_sign >= 0.0 {
            is_top_left_edge(a, b)
        } else {
            is_top_left_edge(b, a)
        }
    } else {
        false
    }
}

#[derive(Clone, Copy)]
struct MaterialRasterParams<'a> {
    kd: Vec3,
    alpha: f32,
    alpha_mode: AlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    metallic: f32,
    roughness: f32,
    map_kd_texcoord_set: usize,
    map_kd_uv_transform: Mat3,
    map_kd: Option<&'a Texture>,
    map_normal_texcoord_set: usize,
    map_normal_uv_transform: Mat3,
    map_normal_scale: f32,
    map_normal: Option<&'a Texture>,
    map_occlusion_texcoord_set: usize,
    map_occlusion_uv_transform: Mat3,
    map_occlusion_strength: f32,
    map_occlusion: Option<&'a Texture>,
    map_emissive_texcoord_set: usize,
    map_emissive_uv_transform: Mat3,
    map_emissive: Option<&'a Texture>,
    map_metallic_roughness_texcoord_set: usize,
    map_metallic_roughness_uv_transform: Mat3,
    map_metallic_roughness: Option<&'a Texture>,
    pbr_metallic_roughness: bool,
}

#[derive(Clone, Copy)]
struct FragmentMaterial {
    depth: f32,
    view_pos: Vec3,
    normal: Vec3,
    kd: Vec3,
    ks: Vec3,
    ns: f32,
    ke: Vec3,
    alpha: f32,
}

fn triangle_tangent_basis(
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    texcoord_set: usize,
    uv_transform: Mat3,
    n: Vec3,
) -> Option<(Vec3, Vec3)> {
    let p0 = v0.view_pos;
    let p1 = v1.view_pos;
    let p2 = v2.view_pos;
    let uv0 = transform_uv(uv_for_set(v0, texcoord_set), uv_transform);
    let uv1 = transform_uv(uv_for_set(v1, texcoord_set), uv_transform);
    let uv2 = transform_uv(uv_for_set(v2, texcoord_set), uv_transform);

    let dp1 = p1 - p0;
    let dp2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let det = duv1.x * duv2.y - duv1.y * duv2.x;
    if det.abs() <= 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;
    let t_raw = (dp1 * duv2.y - dp2 * duv1.y) * inv_det;
    let b_raw = (dp2 * duv1.x - dp1 * duv2.x) * inv_det;
    let t = (t_raw - n * n.dot(t_raw)).normalize_or_zero();
    if t.length_squared() <= 1e-10 {
        return None;
    }
    let mut b = n.cross(t).normalize_or_zero();
    if b.length_squared() <= 1e-10 {
        return None;
    }
    if b.dot(b_raw) < 0.0 {
        b = -b;
    }
    Some((t, b))
}

fn texture_sample_rgba(
    tex: &Texture,
    uv: Vec2,
    footprint: Option<TextureSampleFootprint>,
) -> Vec4 {
    if let Some(fp) = footprint {
        tex.sample_rgba_with_footprint(uv, fp)
    } else {
        tex.sample_rgba(uv)
    }
}

fn transform_tangent(nm: Mat4, tangent: Vec4) -> Vec4 {
    let t = (nm * tangent.truncate().extend(0.0)).truncate();
    Vec4::new(t.x, t.y, t.z, tangent.w)
}

fn eval_fragment_material(
    sample: Vec2,
    s0: Vec2,
    s1: Vec2,
    s2: Vec2,
    area: f32,
    w0: f32,
    w1: f32,
    w2: f32,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    params: MaterialRasterParams<'_>,
) -> Option<FragmentMaterial> {
    let depth = w0 * v0.pos.z + w1 * v1.pos.z + w2 * v2.pos.z;
    let mut normal = interpolate_normal(w0, w1, w2, v0, v1, v2);
    let view_pos = interpolate_view_pos(w0, w1, w2, v0, v1, v2);
    let vertex_color = interpolate_color(w0, w1, w2, v0, v1, v2).clamp(Vec4::ZERO, Vec4::ONE);
    let mut kd = params.kd * vertex_color.truncate();
    let mut alpha = (params.alpha * vertex_color.w).clamp(0.0, 1.0);
    let mut ke = params.ke;
    let mut metallic = params.metallic.clamp(0.0, 1.0);
    let mut roughness = params.roughness.clamp(0.045, 1.0);
    let mut ks = params.ks;
    let mut ns = params.ns;

    if let Some(t) = params.map_kd {
        if let Some(raw_uv) = interpolate_uv_set(
            w0,
            w1,
            w2,
            v0,
            v1,
            v2,
            params.map_kd_texcoord_set,
        ) {
            let uv = transform_uv(raw_uv, params.map_kd_uv_transform);
            let fp = uv_footprint(
                sample,
                area,
                s0,
                s1,
                s2,
                v0,
                v1,
                v2,
                raw_uv,
                params.map_kd_texcoord_set,
                params.map_kd_uv_transform,
            );
            let rgba = texture_sample_rgba(t, uv, fp);
            kd *= rgba.truncate();
            alpha *= rgba.w.clamp(0.0, 1.0);
        }
    }

    if params.pbr_metallic_roughness {
        if let Some(t) = params.map_metallic_roughness {
            if let Some(raw_uv) = interpolate_uv_set(
                w0,
                w1,
                w2,
                v0,
                v1,
                v2,
                params.map_metallic_roughness_texcoord_set,
            ) {
                let uv = transform_uv(raw_uv, params.map_metallic_roughness_uv_transform);
                let fp = uv_footprint(
                    sample,
                    area,
                    s0,
                    s1,
                    s2,
                    v0,
                    v1,
                    v2,
                    raw_uv,
                    params.map_metallic_roughness_texcoord_set,
                    params.map_metallic_roughness_uv_transform,
                );
                let rgba = texture_sample_rgba(t, uv, fp);
                metallic *= rgba.z.clamp(0.0, 1.0);
                roughness *= rgba.y.clamp(0.0, 1.0);
            }
        }
    }

    if let Some(t) = params.map_emissive {
        if let Some(raw_uv) = interpolate_uv_set(
            w0,
            w1,
            w2,
            v0,
            v1,
            v2,
            params.map_emissive_texcoord_set,
        ) {
            let uv = transform_uv(raw_uv, params.map_emissive_uv_transform);
            let fp = uv_footprint(
                sample,
                area,
                s0,
                s1,
                s2,
                v0,
                v1,
                v2,
                raw_uv,
                params.map_emissive_texcoord_set,
                params.map_emissive_uv_transform,
            );
            let rgba = texture_sample_rgba(t, uv, fp);
            ke *= rgba.truncate();
        }
    }

    if let Some(t) = params.map_occlusion {
        if let Some(raw_uv) = interpolate_uv_set(
            w0,
            w1,
            w2,
            v0,
            v1,
            v2,
            params.map_occlusion_texcoord_set,
        ) {
            let uv = transform_uv(raw_uv, params.map_occlusion_uv_transform);
            let fp = uv_footprint(
                sample,
                area,
                s0,
                s1,
                s2,
                v0,
                v1,
                v2,
                raw_uv,
                params.map_occlusion_texcoord_set,
                params.map_occlusion_uv_transform,
            );
            let rgba = texture_sample_rgba(t, uv, fp);
            let ao_sample = rgba.x.clamp(0.0, 1.0);
            let ao = (1.0 + params.map_occlusion_strength * (ao_sample - 1.0)).clamp(0.0, 1.0);
            kd *= ao;
        }
    }

    if let Some(t) = params.map_normal {
        if let Some(raw_uv) = interpolate_uv_set(
            w0,
            w1,
            w2,
            v0,
            v1,
            v2,
            params.map_normal_texcoord_set,
        ) {
            let uv = transform_uv(raw_uv, params.map_normal_uv_transform);
            let fp = uv_footprint(
                sample,
                area,
                s0,
                s1,
                s2,
                v0,
                v1,
                v2,
                raw_uv,
                params.map_normal_texcoord_set,
                params.map_normal_uv_transform,
            );
            let rgba = texture_sample_rgba(t, uv, fp);
            let mut n_tangent = Vec3::new(rgba.x * 2.0 - 1.0, rgba.y * 2.0 - 1.0, rgba.z * 2.0 - 1.0);
            n_tangent.x *= params.map_normal_scale;
            n_tangent.y *= params.map_normal_scale;
            n_tangent = n_tangent.normalize_or_zero();

            let tangent_interp = interpolate_tangent(w0, w1, w2, v0, v1, v2);
            let tangent3 = tangent_interp.truncate();
            let tangent_handedness = if tangent_interp.w < 0.0 { -1.0 } else { 1.0 };
            let tbn = if tangent3.length_squared() > 1e-10 {
                let t = (tangent3 - normal * normal.dot(tangent3)).normalize_or_zero();
                if t.length_squared() > 1e-10 {
                    let b = normal.cross(t) * tangent_handedness;
                    if b.length_squared() > 1e-10 {
                        Some((t, b.normalize_or_zero()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                triangle_tangent_basis(
                    v0,
                    v1,
                    v2,
                    params.map_normal_texcoord_set,
                    params.map_normal_uv_transform,
                    normal,
                )
            };

            if let Some((t, b)) = tbn {
                normal = (t * n_tangent.x + b * n_tangent.y + normal * n_tangent.z).normalize_or_zero();
            }
        }
    }

    if params.pbr_metallic_roughness {
        let roughness = roughness.clamp(0.045, 1.0);
        ks = Vec3::splat(0.04).lerp(kd, metallic.clamp(0.0, 1.0));
        ns = ((2.0 / (roughness * roughness)) - 2.0).clamp(1.0, 1024.0);
    }

    Some(FragmentMaterial {
        depth,
        view_pos,
        normal,
        kd,
        ks,
        ns,
        ke,
        alpha: alpha.clamp(0.0, 1.0),
    })
}

fn params_from_material<'a>(material: &'a Material, scene: &'a Scene) -> MaterialRasterParams<'a> {
    MaterialRasterParams {
        kd: material.kd,
        alpha: material.alpha,
        alpha_mode: material.alpha_mode,
        alpha_cutoff: material.alpha_cutoff,
        double_sided: material.double_sided,
        ks: material.ks,
        ns: material.ns,
        ke: material.ke,
        metallic: material.metallic,
        roughness: material.roughness,
        map_kd_texcoord_set: material.map_kd_texcoord_set,
        map_kd_uv_transform: material.map_kd_uv_transform,
        map_kd: material.map_kd.and_then(|h| scene.texture(h)),
        map_normal_texcoord_set: material.map_normal_texcoord_set,
        map_normal_uv_transform: material.map_normal_uv_transform,
        map_normal_scale: material.map_normal_scale,
        map_normal: material.map_normal.and_then(|h| scene.texture(h)),
        map_occlusion_texcoord_set: material.map_occlusion_texcoord_set,
        map_occlusion_uv_transform: material.map_occlusion_uv_transform,
        map_occlusion_strength: material.map_occlusion_strength,
        map_occlusion: material.map_occlusion.and_then(|h| scene.texture(h)),
        map_emissive_texcoord_set: material.map_emissive_texcoord_set,
        map_emissive_uv_transform: material.map_emissive_uv_transform,
        map_emissive: material.map_emissive.and_then(|h| scene.texture(h)),
        map_metallic_roughness_texcoord_set: material.map_metallic_roughness_texcoord_set,
        map_metallic_roughness_uv_transform: material.map_metallic_roughness_uv_transform,
        map_metallic_roughness: material.map_metallic_roughness.and_then(|h| scene.texture(h)),
        pbr_metallic_roughness: material.pbr_metallic_roughness,
    }
}

fn params_from_tiled_tri<'a>(tri: &TiledTri, scene: &'a Scene) -> MaterialRasterParams<'a> {
    MaterialRasterParams {
        kd: tri.kd,
        alpha: tri.alpha,
        alpha_mode: tri.alpha_mode,
        alpha_cutoff: tri.alpha_cutoff,
        double_sided: tri.double_sided,
        ks: tri.ks,
        ns: tri.ns,
        ke: tri.ke,
        metallic: tri.metallic,
        roughness: tri.roughness,
        map_kd_texcoord_set: tri.map_kd_texcoord_set,
        map_kd_uv_transform: tri.map_kd_uv_transform,
        map_kd: tri.map_kd.and_then(|h| scene.texture(h)),
        map_normal_texcoord_set: tri.map_normal_texcoord_set,
        map_normal_uv_transform: tri.map_normal_uv_transform,
        map_normal_scale: tri.map_normal_scale,
        map_normal: tri.map_normal.and_then(|h| scene.texture(h)),
        map_occlusion_texcoord_set: tri.map_occlusion_texcoord_set,
        map_occlusion_uv_transform: tri.map_occlusion_uv_transform,
        map_occlusion_strength: tri.map_occlusion_strength,
        map_occlusion: tri.map_occlusion.and_then(|h| scene.texture(h)),
        map_emissive_texcoord_set: tri.map_emissive_texcoord_set,
        map_emissive_uv_transform: tri.map_emissive_uv_transform,
        map_emissive: tri.map_emissive.and_then(|h| scene.texture(h)),
        map_metallic_roughness_texcoord_set: tri.map_metallic_roughness_texcoord_set,
        map_metallic_roughness_uv_transform: tri.map_metallic_roughness_uv_transform,
        map_metallic_roughness: tri.map_metallic_roughness.and_then(|h| scene.texture(h)),
        pbr_metallic_roughness: tri.pbr_metallic_roughness,
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
    let view_pos = vert_a.view_pos + t_param * (vert_b.view_pos - vert_a.view_pos);
    let nrm = (vert_a.nrm + t_param * (vert_b.nrm - vert_a.nrm)).normalize_or_zero();
    let color = vert_a.color + t_param * (vert_b.color - vert_a.color);
    let uv0 = vert_a.uv0 + t_param * (vert_b.uv0 - vert_a.uv0);
    let uv1 = vert_a.uv1 + t_param * (vert_b.uv1 - vert_a.uv1);
    let tangent = vert_a.tangent + t_param * (vert_b.tangent - vert_a.tangent);
    let pos_clip = pos_a + t_param * (pos_b - pos_a);
    (
        Vertex {
            pos,
            view_pos,
            nrm,
            color,
            uv0,
            uv1,
            tangent,
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
    params: MaterialRasterParams<'_>,
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
    if !params.double_sided && area <= 0.0 {
        return;
    }
    let area_sign = if area >= 0.0 { 1.0 } else { -1.0 };

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let e0 = edge(s1, s2, sample);
            let e1 = edge(s2, s0, sample);
            let e2 = edge(s0, s1, sample);
            if !edge_is_inside(e0, s1, s2, area_sign)
                || !edge_is_inside(e1, s2, s0, area_sign)
                || !edge_is_inside(e2, s0, s1, area_sign)
            {
                continue;
            }
            let w0 = e0 / area;
            let w1 = e1 / area;
            let w2 = e2 / area;

            let Some(frag) = eval_fragment_material(
                sample, s0, s1, s2, area, w0, w1, w2, v0, v1, v2, params,
            ) else {
                continue;
            };
            match params.alpha_mode {
                AlphaMode::Opaque => {}
                AlphaMode::Mask => {
                    if frag.alpha < params.alpha_cutoff {
                        continue;
                    }
                }
                AlphaMode::Blend => {
                    continue;
                }
            }

            let _ = surf.gbuf.try_write(
                x as usize,
                y as usize,
                frag.depth,
                frag.view_pos,
                frag.normal,
                frag.kd,
                frag.ks,
                frag.ns,
                frag.ke,
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
    params: MaterialRasterParams<'_>,
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
    if !params.double_sided && area <= 0.0 {
        return;
    }
    let area_sign = if area >= 0.0 { 1.0 } else { -1.0 };

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let e0 = edge(s1, s2, sample);
            let e1 = edge(s2, s0, sample);
            let e2 = edge(s0, s1, sample);
            if !edge_is_inside(e0, s1, s2, area_sign)
                || !edge_is_inside(e1, s2, s0, area_sign)
                || !edge_is_inside(e2, s0, s1, area_sign)
            {
                continue;
            }
            let w0 = e0 / area;
            let w1 = e1 / area;
            let w2 = e2 / area;

            let Some(frag) = eval_fragment_material(
                sample, s0, s1, s2, area, w0, w1, w2, v0, v1, v2, params,
            ) else {
                continue;
            };
            match params.alpha_mode {
                AlphaMode::Opaque => {}
                AlphaMode::Mask => {
                    if frag.alpha < params.alpha_cutoff {
                        continue;
                    }
                }
                AlphaMode::Blend => {
                    continue;
                }
            }

            let _ = surf.gbuf.try_write(x as usize, y as usize, frag.depth, frag.view_pos, frag.normal, frag.kd, frag.ks, frag.ns, frag.ke);
        }
    }
}

#[cfg(feature = "rayon")]
struct GBufferChunkMut<'a> {
    width: usize,
    y0: usize,
    depth: &'a mut [f32],
    vx: &'a mut [f32],
    vy: &'a mut [f32],
    vz: &'a mut [f32],
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
        view_pos: Vec3,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> bool {
        const DEPTH_TEST_EPS: f32 = 1e-6;
        if x >= self.width {
            return false;
        }
        let rows = self.rows();
        if y < self.y0 || y >= self.y0 + rows {
            return false;
        }
        let i = (y - self.y0) * self.width + x;
        if depth <= self.depth[i] + DEPTH_TEST_EPS {
            self.depth[i] = depth;
            self.vx[i] = view_pos.x;
            self.vy[i] = view_pos.y;
            self.vz[i] = view_pos.z;
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
    params: MaterialRasterParams<'_>,
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
    if !params.double_sided && area <= 0.0 {
        return;
    }
    let area_sign = if area >= 0.0 { 1.0 } else { -1.0 };

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let e0 = edge(s1, s2, sample);
            let e1 = edge(s2, s0, sample);
            let e2 = edge(s0, s1, sample);
            if !edge_is_inside(e0, s1, s2, area_sign)
                || !edge_is_inside(e1, s2, s0, area_sign)
                || !edge_is_inside(e2, s0, s1, area_sign)
            {
                continue;
            }
            let w0 = e0 / area;
            let w1 = e1 / area;
            let w2 = e2 / area;

            let Some(frag) = eval_fragment_material(
                sample, s0, s1, s2, area, w0, w1, w2, v0, v1, v2, params,
            ) else {
                continue;
            };
            match params.alpha_mode {
                AlphaMode::Opaque => {}
                AlphaMode::Mask => {
                    if frag.alpha < params.alpha_cutoff {
                        continue;
                    }
                }
                AlphaMode::Blend => {
                    continue;
                }
            }

            let _ = gbuf.try_write(x as usize, y as usize, frag.depth, frag.view_pos, frag.normal, frag.kd, frag.ks, frag.ns, frag.ke);
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
        .zip(gbuf.vx.par_chunks_mut(row_stride))
        .zip(gbuf.vy.par_chunks_mut(row_stride))
        .zip(gbuf.vz.par_chunks_mut(row_stride))
        .zip(gbuf.nx.par_chunks_mut(row_stride))
        .zip(gbuf.ny.par_chunks_mut(row_stride))
        .zip(gbuf.nz.par_chunks_mut(row_stride))
        .zip(gbuf.kd.par_chunks_mut(row_stride))
        .zip(gbuf.ks.par_chunks_mut(row_stride))
        .zip(gbuf.ns.par_chunks_mut(row_stride))
        .zip(gbuf.ke.par_chunks_mut(row_stride));

    iter.enumerate().for_each(
        |(
            ty,
            ((((((((((depth_c, vx_c), vy_c), vz_c), nx_c), ny_c), nz_c), kd_c), ks_c), ns_c), ke_c),
        )| {
            let y0 = ty * chunk_rows;
            if y0 >= height {
                return;
            }
            let y1 = (y0 + chunk_rows).min(height);
            let mut chunk = GBufferChunkMut {
                width,
                y0,
                depth: depth_c,
                vx: vx_c,
                vy: vy_c,
                vz: vz_c,
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
                    let params = params_from_tiled_tri(&tri, scene);
                    draw_tri_params_clamped_chunk(
                        cfg,
                        &mut chunk,
                        tri.a,
                        tri.b,
                        tri.c,
                        params,
                        tile_min_x,
                        tile_max_x,
                        tile_min_y,
                        tile_max_y,
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
        view_pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.a.nrm.extend(0.0)).truncate(),
        color: tri.a.color,
        uv0: tri.a.uv0,
        uv1: tri.a.uv1,
        tangent: transform_tangent(nm, tri.a.tangent),
        inv_w: 1.0,
    };
    let mut v1 = Vertex {
        pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        view_pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.b.nrm.extend(0.0)).truncate(),
        color: tri.b.color,
        uv0: tri.b.uv0,
        uv1: tri.b.uv1,
        tangent: transform_tangent(nm, tri.b.tangent),
        inv_w: 1.0,
    };
    let mut v2 = Vertex {
        pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        view_pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.c.nrm.extend(0.0)).truncate(),
        color: tri.c.color,
        uv0: tri.c.uv0,
        uv1: tri.c.uv1,
        tangent: transform_tangent(nm, tri.c.tangent),
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
            alpha: material.alpha,
            alpha_mode: material.alpha_mode,
            alpha_cutoff: material.alpha_cutoff,
            double_sided: material.double_sided,
            ks: material.ks,
            ns: material.ns,
            ke: material.ke,
            metallic: material.metallic,
            roughness: material.roughness,
            map_kd_texcoord_set: material.map_kd_texcoord_set,
            map_kd_uv_transform: material.map_kd_uv_transform,
            map_kd: material.map_kd,
            map_normal_texcoord_set: material.map_normal_texcoord_set,
            map_normal_uv_transform: material.map_normal_uv_transform,
            map_normal_scale: material.map_normal_scale,
            map_normal: material.map_normal,
            map_occlusion_texcoord_set: material.map_occlusion_texcoord_set,
            map_occlusion_uv_transform: material.map_occlusion_uv_transform,
            map_occlusion_strength: material.map_occlusion_strength,
            map_occlusion: material.map_occlusion,
            map_emissive_texcoord_set: material.map_emissive_texcoord_set,
            map_emissive_uv_transform: material.map_emissive_uv_transform,
            map_emissive: material.map_emissive,
            map_metallic_roughness_texcoord_set: material.map_metallic_roughness_texcoord_set,
            map_metallic_roughness_uv_transform: material.map_metallic_roughness_uv_transform,
            map_metallic_roughness: material.map_metallic_roughness,
            pbr_metallic_roughness: material.pbr_metallic_roughness,
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
            let aspect = surf.cfg.camera_aspect;
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
                let uv10 = mesh.uvs1.get(i0).copied().unwrap_or(Vec2::ZERO);
                let uv11 = mesh.uvs1.get(i1).copied().unwrap_or(Vec2::ZERO);
                let uv12 = mesh.uvs1.get(i2).copied().unwrap_or(Vec2::ZERO);
                let t0 = mesh.tangents.get(i0).copied().unwrap_or(Vec4::ZERO);
                let t1 = mesh.tangents.get(i1).copied().unwrap_or(Vec4::ZERO);
                let t2 = mesh.tangents.get(i2).copied().unwrap_or(Vec4::ZERO);

                let tri = Tri {
                    a: Vertex {
                        pos: mesh.positions[i0],
                        view_pos: mesh.positions[i0],
                        nrm: n0,
                        color: mesh.colors.get(i0).copied().unwrap_or(Vec4::ONE),
                        uv0,
                        uv1: uv10,
                        tangent: t0,
                        inv_w: 1.0,
                    },
                    b: Vertex {
                        pos: mesh.positions[i1],
                        view_pos: mesh.positions[i1],
                        nrm: n1,
                        color: mesh.colors.get(i1).copied().unwrap_or(Vec4::ONE),
                        uv0: uv1,
                        uv1: uv11,
                        tangent: t1,
                        inv_w: 1.0,
                    },
                    c: Vertex {
                        pos: mesh.positions[i2],
                        view_pos: mesh.positions[i2],
                        nrm: n2,
                        color: mesh.colors.get(i2).copied().unwrap_or(Vec4::ONE),
                        uv0: uv2,
                        uv1: uv12,
                        tangent: t2,
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
                    let params = params_from_tiled_tri(&t, scene);

                    draw_tri_params_clamped(
                        surf,
                        t.a,
                        t.b,
                        t.c,
                        params,
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
    scene: &Scene,
    _cam: &Camera,
    scratch: &mut Scratch,
    tri: &Tri,
    mv: Mat4,
    mvp: Mat4,
    nm: Mat4,
    material: &Material,
) {
    let mut v0 = Vertex {
        pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        view_pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.a.nrm.extend(0.0)).truncate(),
        color: tri.a.color,
        uv0: tri.a.uv0,
        uv1: tri.a.uv1,
        tangent: transform_tangent(nm, tri.a.tangent),
        inv_w: 1.0,
    };
    let mut v1 = Vertex {
        pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        view_pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.b.nrm.extend(0.0)).truncate(),
        color: tri.b.color,
        uv0: tri.b.uv0,
        uv1: tri.b.uv1,
        tangent: transform_tangent(nm, tri.b.tangent),
        inv_w: 1.0,
    };
    let mut v2 = Vertex {
        pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        view_pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.c.nrm.extend(0.0)).truncate(),
        color: tri.c.color,
        uv0: tri.c.uv0,
        uv1: tri.c.uv1,
        tangent: transform_tangent(nm, tri.c.tangent),
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

        let params = params_from_material(material, scene);
        draw_tri(surf, aa, bb, cc, params);
    }
}

fn rasterize(scene: &Scene, cam: &Camera, surf: &mut RasterSurface, scratch: &mut Scratch) {
    if surf.cfg.width == 0 || surf.cfg.height == 0 {
        return;
    }
    for (mesh, material, transform) in scene.iter_objects() {
        let mv = cam.view_matrix() * transform.to_mat4();
        let aspect = surf.cfg.camera_aspect;
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
            let uv10 = mesh.uvs1.get(i0).copied().unwrap_or(Vec2::ZERO);
            let uv11 = mesh.uvs1.get(i1).copied().unwrap_or(Vec2::ZERO);
            let uv12 = mesh.uvs1.get(i2).copied().unwrap_or(Vec2::ZERO);
            let t0 = mesh.tangents.get(i0).copied().unwrap_or(Vec4::ZERO);
            let t1 = mesh.tangents.get(i1).copied().unwrap_or(Vec4::ZERO);
            let t2 = mesh.tangents.get(i2).copied().unwrap_or(Vec4::ZERO);

            let tri = Tri {
                a: Vertex {
                    pos: mesh.positions[i0],
                    view_pos: mesh.positions[i0],
                    nrm: n0,
                    color: mesh.colors.get(i0).copied().unwrap_or(Vec4::ONE),
                    uv0,
                    uv1: uv10,
                    tangent: t0,
                    inv_w: 1.0,
                },
                b: Vertex {
                    pos: mesh.positions[i1],
                    view_pos: mesh.positions[i1],
                    nrm: n1,
                    color: mesh.colors.get(i1).copied().unwrap_or(Vec4::ONE),
                    uv0: uv1,
                    uv1: uv11,
                    tangent: t1,
                    inv_w: 1.0,
                },
                c: Vertex {
                    pos: mesh.positions[i2],
                    view_pos: mesh.positions[i2],
                    nrm: n2,
                    color: mesh.colors.get(i2).copied().unwrap_or(Vec4::ONE),
                    uv0: uv2,
                    uv1: uv12,
                    tangent: t2,
                    inv_w: 1.0,
                },
            };

            rasterize_tri(surf, scene, cam, scratch, &tri, mv, mvp, nm, material);
        }
    }
}

fn sanitize_camera_aspect(width: u32, height: u32, camera_aspect: f32) -> f32 {
    if camera_aspect.is_finite() && camera_aspect > 0.0 {
        camera_aspect
    } else {
        width as f32 / (height.max(1) as f32)
    }
}

fn render_to_gbuffer_with_scratch(
    scene: &Scene,
    gbuf: &mut GBuffer,
    scratch: &mut Scratch,
    camera_aspect: f32,
) {
    let cfg = RasterConfig {
        width: gbuf.width() as u32,
        height: gbuf.height() as u32,
        camera_aspect: sanitize_camera_aspect(
            gbuf.width() as u32,
            gbuf.height() as u32,
            camera_aspect,
        ),
    };
    gbuf.clear();
    scratch.begin_frame();
    let mut surf = RasterSurface::new(cfg, gbuf);
    rasterize(scene, &scene.camera, &mut surf, scratch);
    scratch.finish_frame();
}

fn render_to_gbuffer_tiled_with_scratch(
    scene: &Scene,
    gbuf: &mut GBuffer,
    scratch: &mut Scratch,
    camera_aspect: f32,
) {
    let cfg = RasterConfig {
        width: gbuf.width() as u32,
        height: gbuf.height() as u32,
        camera_aspect: sanitize_camera_aspect(
            gbuf.width() as u32,
            gbuf.height() as u32,
            camera_aspect,
        ),
    };
    gbuf.clear();
    scratch.begin_frame();
    let mut surf = RasterSurface::new(cfg, gbuf);
    rasterize_tiled(scene, &scene.camera, &mut surf, scratch);
    scratch.finish_frame();
}

pub fn render_to_gbuffer_tiled(scene: &Scene, gbuf: &mut GBuffer) {
    let camera_aspect = gbuf.width() as f32 / (gbuf.height().max(1) as f32);
    render_to_gbuffer_tiled_with_camera_aspect(scene, gbuf, camera_aspect);
}

pub fn render_to_gbuffer_tiled_with_camera_aspect(
    scene: &Scene,
    gbuf: &mut GBuffer,
    camera_aspect: f32,
) {
    TL_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        render_to_gbuffer_tiled_with_scratch(scene, gbuf, &mut scratch, camera_aspect);
    });
}

pub fn render_to_gbuffer(scene: &Scene, gbuf: &mut GBuffer) {
    let camera_aspect = gbuf.width() as f32 / (gbuf.height().max(1) as f32);
    render_to_gbuffer_with_camera_aspect(scene, gbuf, camera_aspect);
}

pub fn render_to_gbuffer_with_camera_aspect(scene: &Scene, gbuf: &mut GBuffer, camera_aspect: f32) {
    TL_SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        render_to_gbuffer_with_scratch(scene, gbuf, &mut scratch, camera_aspect);
    });
}

fn draw_tri_collect_blend_samples(
    cfg: RasterConfig,
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
    params: MaterialRasterParams<'_>,
    opaque_depth: &[f32],
    out: &mut Vec<BlendSample>,
) {
    if params.alpha_mode != AlphaMode::Blend {
        return;
    }
    let s0 = project_to_screen(v0.pos, cfg.width, cfg.height);
    let s1 = project_to_screen(v1.pos, cfg.width, cfg.height);
    let s2 = project_to_screen(v2.pos, cfg.width, cfg.height);

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

    let area = edge(s0, s1, s2);
    if area.abs() < 1e-8 {
        return;
    }
    if !params.double_sided && area <= 0.0 {
        return;
    }
    let area_sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let width = cfg.width as usize;
    const OPAQUE_DEPTH_EPS: f32 = 1e-6;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let e0 = edge(s1, s2, sample);
            let e1 = edge(s2, s0, sample);
            let e2 = edge(s0, s1, sample);
            if !edge_is_inside(e0, s1, s2, area_sign)
                || !edge_is_inside(e1, s2, s0, area_sign)
                || !edge_is_inside(e2, s0, s1, area_sign)
            {
                continue;
            }
            let w0 = e0 / area;
            let w1 = e1 / area;
            let w2 = e2 / area;

            let Some(frag) = eval_fragment_material(
                sample, s0, s1, s2, area, w0, w1, w2, v0, v1, v2, params,
            ) else {
                continue;
            };
            if frag.alpha <= 0.0 {
                continue;
            }

            let ux = x as usize;
            let uy = y as usize;
            let i = uy.saturating_mul(width).saturating_add(ux);
            if i >= opaque_depth.len() {
                continue;
            }
            let od = opaque_depth[i];
            if od.is_finite() && frag.depth > od + OPAQUE_DEPTH_EPS {
                continue;
            }

            out.push(BlendSample {
                x: ux,
                y: uy,
                depth: frag.depth,
                view_pos: frag.view_pos,
                normal: frag.normal,
                kd: frag.kd,
                ks: frag.ks,
                ns: frag.ns,
                ke: frag.ke,
                alpha: frag.alpha,
            });
        }
    }
}

pub fn rasterize_blend_samples_with_camera_aspect(
    scene: &Scene,
    width: usize,
    height: usize,
    camera_aspect: f32,
    opaque_depth: &[f32],
) -> Vec<BlendSample> {
    let pixel_count = width.saturating_mul(height);
    if width == 0 || height == 0 || opaque_depth.len() != pixel_count {
        return Vec::new();
    }

    let cfg = RasterConfig {
        width: width as u32,
        height: height as u32,
        camera_aspect: sanitize_camera_aspect(width as u32, height as u32, camera_aspect),
    };

    let mut clip_a: Vec<ClipVert> = Vec::with_capacity(16);
    let mut clip_b: Vec<ClipVert> = Vec::with_capacity(16);
    let mut tiled_tris: Vec<TiledTri> = Vec::new();
    let mut grow_events: usize = 0;

    for (mesh, material, transform) in scene.iter_objects() {
        if material.alpha_mode != AlphaMode::Blend {
            continue;
        }
        let mv = scene.camera.view_matrix() * transform.to_mat4();
        let mvp = scene.camera.projection_matrix(cfg.camera_aspect) * mv;
        let nm = mv.inverse().transpose();

        for idx in mesh.indices.iter() {
            let i0 = idx[0] as usize;
            let i1 = idx[1] as usize;
            let i2 = idx[2] as usize;

            if i0 >= mesh.positions.len() || i1 >= mesh.positions.len() || i2 >= mesh.positions.len() {
                continue;
            }

            let n0 = mesh.normals.get(i0).copied().unwrap_or(Vec3::Z);
            let n1 = mesh.normals.get(i1).copied().unwrap_or(Vec3::Z);
            let n2 = mesh.normals.get(i2).copied().unwrap_or(Vec3::Z);

            let uv0 = mesh.uvs.get(i0).copied().unwrap_or(Vec2::ZERO);
            let uv1 = mesh.uvs.get(i1).copied().unwrap_or(Vec2::ZERO);
            let uv2 = mesh.uvs.get(i2).copied().unwrap_or(Vec2::ZERO);
            let uv10 = mesh.uvs1.get(i0).copied().unwrap_or(Vec2::ZERO);
            let uv11 = mesh.uvs1.get(i1).copied().unwrap_or(Vec2::ZERO);
            let uv12 = mesh.uvs1.get(i2).copied().unwrap_or(Vec2::ZERO);
            let t0 = mesh.tangents.get(i0).copied().unwrap_or(Vec4::ZERO);
            let t1 = mesh.tangents.get(i1).copied().unwrap_or(Vec4::ZERO);
            let t2 = mesh.tangents.get(i2).copied().unwrap_or(Vec4::ZERO);

            let tri = Tri {
                a: Vertex {
                    pos: mesh.positions[i0],
                    view_pos: mesh.positions[i0],
                    nrm: n0,
                    color: mesh.colors.get(i0).copied().unwrap_or(Vec4::ONE),
                    uv0,
                    uv1: uv10,
                    tangent: t0,
                    inv_w: 1.0,
                },
                b: Vertex {
                    pos: mesh.positions[i1],
                    view_pos: mesh.positions[i1],
                    nrm: n1,
                    color: mesh.colors.get(i1).copied().unwrap_or(Vec4::ONE),
                    uv0: uv1,
                    uv1: uv11,
                    tangent: t1,
                    inv_w: 1.0,
                },
                c: Vertex {
                    pos: mesh.positions[i2],
                    view_pos: mesh.positions[i2],
                    nrm: n2,
                    color: mesh.colors.get(i2).copied().unwrap_or(Vec4::ONE),
                    uv0: uv2,
                    uv1: uv12,
                    tangent: t2,
                    inv_w: 1.0,
                },
            };

            rasterize_tri_collect(
                &scene.camera,
                &mut clip_a,
                &mut clip_b,
                &mut grow_events,
                &tri,
                mv,
                mvp,
                nm,
                material,
                &mut tiled_tris,
            );
        }
    }

    let mut blend_tris: Vec<TiledTri> = tiled_tris
        .iter()
        .copied()
        .filter(|t| t.alpha_mode == AlphaMode::Blend)
        .collect();
    blend_tris.sort_by(|a, b| {
        let za = (a.a.pos.z + a.b.pos.z + a.c.pos.z) * (1.0 / 3.0);
        let zb = (b.a.pos.z + b.b.pos.z + b.c.pos.z) * (1.0 / 3.0);
        zb.partial_cmp(&za).unwrap_or(Ordering::Equal)
    });

    let mut out = Vec::new();
    for tri in blend_tris {
        let params = params_from_tiled_tri(&tri, scene);
        draw_tri_collect_blend_samples(cfg, tri.a, tri.b, tri.c, params, opaque_depth, &mut out);
    }
    out
}

#[cfg(test)]
pub(crate) fn scratch_grow_events_last_frame() -> usize {
    TL_SCRATCH.with(|s| s.borrow().last_grow_events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera, GBuffer, Material, Mesh, Projection, Scene, Transform};

    fn finite_x_span(gbuf: &GBuffer) -> Option<(usize, usize)> {
        let mut min_x = usize::MAX;
        let mut max_x = 0usize;
        let mut any = false;
        for y in 0..gbuf.height() {
            for x in 0..gbuf.width() {
                let Some(p) = gbuf.at(x, y) else {
                    continue;
                };
                if !p.depth.is_finite() {
                    continue;
                }
                any = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
            }
        }
        if any {
            Some((min_x, max_x))
        } else {
            None
        }
    }

    #[test]
    fn pr21_raster_clip_scratch_does_not_grow() {
        let mut scene = Scene::new();
        let mesh = Mesh::unit_cube();
        scene.add_object(mesh, Transform::default(), Material::default());

        let mut gbuf = GBuffer::new(64, 64);
        render_to_gbuffer(&scene, &mut gbuf);

        assert_eq!(scratch_grow_events_last_frame(), 0);
    }

    #[test]
    fn camera_aspect_override_matches_display_framing() {
        let mut scene = Scene::new();
        scene.camera = Camera::new(
            Transform::IDENTITY,
            Projection::Orthographic {
                half_height: 1.0,
                near: -10.0,
                far: 10.0,
            },
        );
        scene.add_object(Mesh::unit_cube(), Transform::IDENTITY, Material::default());

        let mut gbuf_display = GBuffer::new(64, 32);
        render_to_gbuffer(&scene, &mut gbuf_display);
        let span_display = finite_x_span(&gbuf_display).unwrap();

        let mut gbuf_halfblock_raster = GBuffer::new(64, 64);
        render_to_gbuffer_with_camera_aspect(&scene, &mut gbuf_halfblock_raster, 64.0 / 32.0);
        let span_override = finite_x_span(&gbuf_halfblock_raster).unwrap();

        assert!((span_display.0 as i32 - span_override.0 as i32).abs() <= 1);
        assert!((span_display.1 as i32 - span_override.1 as i32).abs() <= 1);
    }
}
