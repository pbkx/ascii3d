use glam::{Mat3, Vec2, Vec3};

use crate::{
    camera::{Camera, Projection},
    material::Material,
    scene::Scene,
    targets::buffer::{BufferTarget, Cell},
    transform::Transform,
};

const RAMP_SMOOTH: &str = r#"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^'. `"#;

#[derive(Clone, Copy, Debug)]
pub struct RasterConfig {
    pub width: u32,
    pub height: u32,
}

pub struct RasterSurface<'a> {
    pub config: RasterConfig,
    pub buf: &'a mut BufferTarget,
}

impl<'a> RasterSurface<'a> {
    pub fn new(config: RasterConfig, buf: &'a mut BufferTarget) -> Self {
        Self {
            config,
            buf,
        }
    }

    pub fn clear(&mut self) {
        self.buf.clear(Cell::new(' ', 255, 0, f32::INFINITY));
    }
}

#[derive(Clone, Copy)]
struct Vtx3 {
    p: Vec3,
    n: Vec3,
}

#[derive(Clone, Copy)]
struct Vtx2 {
    x: f32,
    y: f32,
    z: f32,
    n: Vec3,
}

fn projection_params(camera: &Camera, aspect: f32) -> (bool, f32, f32, f32, f32) {
    match camera.projection {
        Projection::Perspective {
            fov_y_radians,
            near,
            far,
        } => {
            let inv_tan = 1.0 / (0.5 * fov_y_radians).tan().max(1e-6);
            (true, inv_tan, near.max(1e-4), far.max(near + 1e-3), aspect.max(1e-6))
        }
        Projection::Orthographic {
            half_height,
            near,
            far,
        } => (false, half_height.max(1e-4), near.max(1e-4), far.max(near + 1e-3), aspect.max(1e-6)),
    }
}

fn world_to_view(transform: Transform, camera: &Camera) -> glam::Mat4 {
    camera.view_matrix() * transform.to_mat4()
}

fn normal_matrix(mv: glam::Mat4) -> Mat3 {
    Mat3::from_mat4(mv).inverse().transpose()
}

fn clip_against_near(tri: [Vtx3; 3], near: f32) -> Vec<[Vtx3; 3]> {
    let mut poly = Vec::with_capacity(4);
    poly.push(tri[0]);
    poly.push(tri[1]);
    poly.push(tri[2]);

    let mut out = Vec::with_capacity(4);

    let inside = |v: &Vtx3| v.p.z >= near;

    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let a_in = inside(&a);
        let b_in = inside(&b);

        if a_in && b_in {
            out.push(b);
        } else if a_in && !b_in {
            out.push(intersect_near(a, b, near));
        } else if !a_in && b_in {
            out.push(intersect_near(a, b, near));
            out.push(b);
        }
    }

    if out.len() < 3 {
        return Vec::new();
    }

    let mut tris = Vec::new();
    for i in 1..(out.len() - 1) {
        tris.push([out[0], out[i], out[i + 1]]);
    }
    tris
}

fn intersect_near(a: Vtx3, b: Vtx3, near: f32) -> Vtx3 {
    let denom = b.p.z - a.p.z;
    let t = if denom.abs() < 1e-12 {
        0.0
    } else {
        (near - a.p.z) / denom
    };

    let p = a.p + (b.p - a.p) * t;
    let n = (a.n + (b.n - a.n) * t).normalize_or_zero();
    Vtx3 {
        p,
        n,
    }
}

fn project(v: Vtx3, persp: bool, a: f32, inv_tan_or_half_h: f32) -> Option<Vtx2> {
    if persp {
        let z = v.p.z.max(1e-6);
        let x = v.p.x * inv_tan_or_half_h / (z * a);
        let y = v.p.y * inv_tan_or_half_h / z;
        Some(Vtx2 { x, y, z, n: v.n })
    } else {
        let hh = inv_tan_or_half_h.max(1e-6);
        let hw = hh * a;
        let x = v.p.x / hw;
        let y = v.p.y / hh;
        Some(Vtx2 { x, y, z: v.p.z, n: v.n })
    }
}

fn ndc_to_pixel(x: f32, y: f32, w: u32, h: u32) -> (f32, f32) {
    let sx = (x * 0.5 + 0.5) * (w as f32 - 1.0);
    let sy = (1.0 - (y * 0.5 + 0.5)) * (h as f32 - 1.0);
    (sx, sy)
}

fn edge(a: Vec2, b: Vec2, p: Vec2) -> f32 {
    (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)
}

fn shade_char(luma: f32, x: i32, y: i32) -> char {
    let ramp = RAMP_SMOOTH.as_bytes();
    let n = ramp.len().max(1) as i32;

    let l = luma.clamp(0.0, 1.0);
    let t = l * (n as f32 - 1.0);
    let base = t.floor() as i32;
    let frac = t - base as f32;

    let bayer = [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ];

    let tx = (x & 3) as usize;
    let ty = (y & 3) as usize;
    let threshold = (bayer[ty][tx] as f32 + 0.5) / 16.0;

    let idx = if frac > threshold {
        (base + 1).clamp(0, n - 1)
    } else {
        base.clamp(0, n - 1)
    };

    ramp[idx as usize] as char
}

fn draw_tri(
    surf: &mut RasterSurface<'_>,
    v0: Vtx2,
    v1: Vtx2,
    v2: Vtx2,
    material: Material,
) {
    let (w, h) = (surf.config.width as i32, surf.config.height as i32);

    let (x0, y0) = ndc_to_pixel(v0.x, v0.y, surf.config.width, surf.config.height);
    let (x1, y1) = ndc_to_pixel(v1.x, v1.y, surf.config.width, surf.config.height);
    let (x2, y2) = ndc_to_pixel(v2.x, v2.y, surf.config.width, surf.config.height);

    let p0 = Vec2::new(x0, y0);
    let p1 = Vec2::new(x1, y1);
    let p2 = Vec2::new(x2, y2);

    let area = edge(p0, p1, p2);
    if area.abs() < 1e-9 {
        return;
    }

    let min_x = (p0.x.min(p1.x).min(p2.x).floor() as i32).clamp(0, w - 1);
    let max_x = (p0.x.max(p1.x).max(p2.x).ceil() as i32).clamp(0, w - 1);
    let min_y = (p0.y.min(p1.y).min(p2.y).floor() as i32).clamp(0, h - 1);
    let max_y = (p0.y.max(p1.y).max(p2.y).ceil() as i32).clamp(0, h - 1);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let pc = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);

            let w0 = edge(p1, p2, pc);
            let w1 = edge(p2, p0, pc);
            let w2 = edge(p0, p1, pc);

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 && area > 0.0)
                || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0 && area < 0.0)
            {
                let inv = 1.0 / area;
                let l0 = w0 * inv;
                let l1 = w1 * inv;
                let l2 = w2 * inv;

                let z = v0.z * l0 + v1.z * l1 + v2.z * l2;

                let ux = x as u32;
                let uy = y as u32;

                let pass = match surf.buf.get(ux as usize, uy as usize) {
                    None => true,
                    Some(prev) => z < prev.depth(),
                };

                if pass {
                    let n = (v0.n * l0 + v1.n * l1 + v2.n * l2).normalize_or_zero();
                    let luma = lambert_dummy(n, material);
                    let ch = shade_char(luma, x, y);
                    let _ = surf.buf.set(ux as usize, uy as usize, Cell::new(ch, 255, 0, z));
                }
            }
        }
    }
}

fn lambert_dummy(n: Vec3, material: Material) -> f32 {
    let base = n.dot(Vec3::new(0.3, 0.7, 1.0).normalize_or_zero()).max(0.0);
    let l = (0.15 + 0.85 * base).clamp(0.0, 1.0);
    let m = (material.albedo.x + material.albedo.y + material.albedo.z) / 3.0;
    (l * (0.5 + 0.5 * m.clamp(0.0, 1.0))).clamp(0.0, 1.0)
}

pub fn rasterize(scene: &Scene, camera: &Camera, surf: &mut RasterSurface<'_>) {
    surf.clear();

    let aspect = surf.config.width as f32 / surf.config.height.max(1) as f32;
    let (persp, inv_tan_or_half_h, near, _far, aspect) = projection_params(camera, aspect);

    for (mesh, material, transform) in scene.iter_objects() {
        let mv = world_to_view(*transform, camera);
        let nm = normal_matrix(mv);

        let material = *material;

        let positions = &mesh.positions;
        let normals = &mesh.normals;
        let indices = &mesh.indices;

        for [a, b, c] in indices.iter().copied() {
            let ia = a as usize;
            let ib = b as usize;
            let ic = c as usize;

            if ia >= positions.len() || ib >= positions.len() || ic >= positions.len() {
                continue;
            }

            let pa = mv.transform_point3(positions[ia]);
            let pb = mv.transform_point3(positions[ib]);
            let pc = mv.transform_point3(positions[ic]);

            let face_n = (pb - pa).cross(pc - pa).normalize_or_zero();

            let na = match normals.get(ia).copied() {
                None => face_n,
                Some(n) => (nm * n).normalize_or_zero(),
            };
            let nb = match normals.get(ib).copied() {
                None => face_n,
                Some(n) => (nm * n).normalize_or_zero(),
            };
            let nc = match normals.get(ic).copied() {
                None => face_n,
                Some(n) => (nm * n).normalize_or_zero(),
            };

            let clipped = clip_against_near(
                [
                    Vtx3 { p: pa, n: na },
                    Vtx3 { p: pb, n: nb },
                    Vtx3 { p: pc, n: nc },
                ],
                near,
            );

            for t in clipped {
                let v0 = match project(t[0], persp, aspect, inv_tan_or_half_h) {
                    None => continue,
                    Some(v) => v,
                };
                let v1 = match project(t[1], persp, aspect, inv_tan_or_half_h) {
                    None => continue,
                    Some(v) => v,
                };
                let v2 = match project(t[2], persp, aspect, inv_tan_or_half_h) {
                    None => continue,
                    Some(v) => v,
                };

                draw_tri(surf, v0, v1, v2, material);
            }
        }
    }
}

pub fn render_to_buffer(scene: &Scene, buf: &mut BufferTarget) {
    let cfg = RasterConfig {
        width: buf.width() as u32,
        height: buf.height() as u32,
    };
    let mut surf = RasterSurface::new(cfg, buf);
    rasterize(scene, &scene.camera, &mut surf);
}

