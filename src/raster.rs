use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{Camera, GBuffer, Material, Scene};

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    nrm: Vec3,
}

#[derive(Copy, Clone, Debug)]
struct Tri {
    a: Vertex,
    b: Vertex,
    c: Vertex,
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

fn project_to_screen(pos: Vec3, width: u32, height: u32) -> Vec2 {
    Vec2::new(
        (pos.x * 0.5 + 0.5) * (width.saturating_sub(1) as f32),
        (1.0 - (pos.y * 0.5 + 0.5)) * (height.saturating_sub(1) as f32),
    )
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn draw_tri(surf: &mut RasterSurface, v0: Vertex, v1: Vertex, v2: Vertex, material: &Material) {
    let s0 = project_to_screen(v0.pos, surf.cfg.width, surf.cfg.height);
    let s1 = project_to_screen(v1.pos, surf.cfg.width, surf.cfg.height);
    let s2 = project_to_screen(v2.pos, surf.cfg.width, surf.cfg.height);

    let min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as i32;
    let max_x = s0
        .x
        .max(s1.x)
        .max(s2.x)
        .ceil()
        .min((surf.cfg.width as i32 - 1) as f32) as i32;
    let min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as i32;
    let max_y = s0
        .y
        .max(s1.y)
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
            let normal = (w0 * v0.nrm + w1 * v1.nrm + w2 * v2.nrm).normalize_or_zero();
            let _ = surf.gbuf.try_write(
                x as usize,
                y as usize,
                depth,
                normal,
                material.kd,
                material.ks,
                material.ns,
                material.ke,
            );
        }
    }
}

fn rasterize_tri(
    surf: &mut RasterSurface,
    cam: &Camera,
    tri: &Tri,
    mv: Mat4,
    mvp: Mat4,
    nm: Mat4,
    material: &Material,
) {
    let mut v0 = Vertex {
        pos: (mv * tri.a.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.a.nrm.extend(0.0)).truncate(),
    };
    let mut v1 = Vertex {
        pos: (mv * tri.b.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.b.nrm.extend(0.0)).truncate(),
    };
    let mut v2 = Vertex {
        pos: (mv * tri.c.pos.extend(1.0)).truncate(),
        nrm: (nm * tri.c.nrm.extend(0.0)).truncate(),
    };

    v0.nrm = v0.nrm.normalize_or_zero();
    v1.nrm = v1.nrm.normalize_or_zero();
    v2.nrm = v2.nrm.normalize_or_zero();

    let v0c = mvp * tri.a.pos.extend(1.0);
    let v1c = mvp * tri.b.pos.extend(1.0);
    let v2c = mvp * tri.c.pos.extend(1.0);

    let p0 = v0c / v0c.w;
    let p1 = v1c / v1c.w;
    let p2 = v2c / v2c.w;

    let near = cam.near();
    let far = cam.far();
    fn clip_edge(vert_a: Vertex, vert_b: Vertex, pos_a: Vec4, pos_b: Vec4, z_plane: f32) -> (Vertex, Vec4) {
        let t_param = (z_plane - pos_a.z) / (pos_b.z - pos_a.z);
        let pos = vert_a.pos + t_param * (vert_b.pos - vert_a.pos);
        let nrm = (vert_a.nrm + t_param * (vert_b.nrm - vert_a.nrm)).normalize_or_zero();
        let pos_clip = pos_a + t_param * (pos_b - pos_a);
        (
            Vertex {
                pos,
                nrm,
            },
            pos_clip,
        )
    }

    let mut verts = vec![(v0, p0), (v1, p1), (v2, p2)];
    for &(plane_z, keep_greater) in &[(near, true), (far, false)] {
        let mut out: Vec<(Vertex, Vec4)> = Vec::new();
        if verts.is_empty() {
            break;
        }
        for i in 0..verts.len() {
            let (va, pa) = verts[i];
            let (vb, pb) = verts[(i + 1) % verts.len()];

            let ina = if keep_greater { pa.z >= plane_z } else { pa.z <= plane_z };
            let inb = if keep_greater { pb.z >= plane_z } else { pb.z <= plane_z };

            if ina && inb {
                out.push((vb, pb));
            } else if ina && !inb {
                out.push(clip_edge(va, vb, pa, pb, plane_z));
            } else if !ina && inb {
                out.push(clip_edge(va, vb, pa, pb, plane_z));
                out.push((vb, pb));
            }
        }
        verts = out;
    }

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

        aa.pos = pa.truncate();
        bb.pos = pb.truncate();
        cc.pos = pc.truncate();

        draw_tri(surf, aa, bb, cc, material);
    }
}

fn rasterize(scene: &Scene, cam: &Camera, surf: &mut RasterSurface) {
    for (mesh, material, transform) in scene.iter_objects() {
        let mv = cam.view_matrix() * transform.to_mat4();
        let aspect = surf.cfg.width as f32 / surf.cfg.height as f32;
        let mvp = cam.projection_matrix(aspect) * mv;
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

            let tri = Tri {
                a: Vertex {
                    pos: mesh.positions[i0],
                    nrm: n0,
                },
                b: Vertex {
                    pos: mesh.positions[i1],
                    nrm: n1,
                },
                c: Vertex {
                    pos: mesh.positions[i2],
                    nrm: n2,
                },
            };

            rasterize_tri(surf, cam, &tri, mv, mvp, nm, material);
        }
    }
}

pub fn render_to_gbuffer(scene: &Scene, gbuf: &mut GBuffer) {
    let cfg = RasterConfig {
        width: gbuf.width() as u32,
        height: gbuf.height() as u32,
    };
    gbuf.clear();
    let mut surf = RasterSurface::new(cfg, gbuf);
    rasterize(scene, &scene.camera, &mut surf);
}
