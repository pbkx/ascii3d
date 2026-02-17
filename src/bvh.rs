use glam::{Mat4, Vec2, Vec3};

use crate::{Mesh, Scene, Transform};

const LEAF_TRI_LIMIT: usize = 4;
const DIR_EPS: f32 = 1e-12;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        let d = if dir.length_squared().is_finite() && dir.length_squared() > 0.0 {
            dir.normalize()
        } else {
            Vec3::ZERO
        };
        Self { origin, dir: d }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(-f32::INFINITY),
        }
    }

    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let mut bb = Self::empty();
        bb.grow(a);
        bb.grow(b);
        bb.grow(c);
        bb
    }

    pub fn grow(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn hit(&self, ray: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        let o = ray.origin;
        let d = ray.dir;

        for axis in 0..3 {
            let origin = o[axis];
            let dir = d[axis];
            let minp = self.min[axis];
            let maxp = self.max[axis];

            if dir.abs() <= DIR_EPS {
                if origin < minp || origin > maxp {
                    return false;
                }
                continue;
            }

            let inv = 1.0 / dir;
            let mut t0 = (minp - origin) * inv;
            let mut t1 = (maxp - origin) * inv;

            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);
            t_max = t_max.min(t1);

            if t_max < t_min {
                return false;
            }
        }

        true
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshHit {
    pub tri_index: usize,
    pub t: f32,
    pub barycentric: Vec3,
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Option<Vec2>,
}

#[derive(Clone, Copy, Debug)]
pub struct SceneHit {
    pub object_index: usize,
    pub tri_index: usize,
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Option<Vec2>,
}

#[derive(Clone, Copy, Debug)]
struct TriRayHit {
    t: f32,
    u: f32,
    v: f32,
}

fn ray_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3, t_min: f32, t_max: f32) -> Option<TriRayHit> {
    let e1 = b - a;
    let e2 = c - a;

    let pvec = ray.dir.cross(e2);
    let det = e1.dot(pvec);

    let eps = 1e-8f32;
    if det.abs() < eps {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = ray.origin - a;

    let u = tvec.dot(pvec) * inv_det;
    if u < -eps || u > 1.0 + eps {
        return None;
    }

    let qvec = tvec.cross(e1);
    let v = ray.dir.dot(qvec) * inv_det;
    if v < -eps || (u + v) > 1.0 + eps {
        return None;
    }

    let t = e2.dot(qvec) * inv_det;
    if !(t.is_finite() && t > t_min && t < t_max) {
        return None;
    }

    Some(TriRayHit { t, u, v })
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bounds: Aabb,
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

impl BvhNode {
    fn leaf(bounds: Aabb, start: u32, count: u32) -> Self {
        Self {
            bounds,
            left: u32::MAX,
            right: u32::MAX,
            start,
            count,
        }
    }

    fn inner(bounds: Aabb, left: u32, right: u32) -> Self {
        Self {
            bounds,
            left,
            right,
            start: 0,
            count: 0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.count != 0
    }
}

#[derive(Clone, Debug)]
pub struct MeshBvh {
    nodes: Vec<BvhNode>,
    tri_indices: Vec<u32>,
    root: u32,
}

impl MeshBvh {
    pub fn build(mesh: &Mesh) -> Self {
        let tri_count = mesh.indices.len();
        let mut tri_indices: Vec<u32> = (0..tri_count as u32).collect();
        let mut nodes: Vec<BvhNode> =
            Vec::with_capacity(tri_count.saturating_mul(2).saturating_add(1));

        let root = build_node(mesh, &mut tri_indices, 0, tri_count, &mut nodes) as u32;

        Self {
            nodes,
            tri_indices,
            root,
        }
    }

    pub fn intersect(&self, mesh: &Mesh, ray: &Ray) -> Option<MeshHit> {
        self.intersect_range(mesh, ray, 0.0, f32::INFINITY)
    }

    pub fn intersect_range(
        &self,
        mesh: &Mesh,
        ray: &Ray,
        t_min: f32,
        t_max: f32,
    ) -> Option<MeshHit> {
        if self.nodes.is_empty() || mesh.indices.is_empty() {
            return None;
        }

        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(self.root);

        let mut best_t = t_max;
        let mut best_tri = usize::MAX;
        let mut best_bary = Vec3::ZERO;

        while let Some(nidx) = stack.pop() {
            let node = self.nodes[nidx as usize];
            if !node.bounds.hit(ray, t_min, best_t) {
                continue;
            }

            if node.is_leaf() {
                let start = node.start as usize;
                let end = start + node.count as usize;
                for i in start..end {
                    let tri_idx = self.tri_indices[i] as usize;
                    let tri = mesh.indices[tri_idx];
                    let a = mesh.positions[tri[0] as usize];
                    let b = mesh.positions[tri[1] as usize];
                    let c = mesh.positions[tri[2] as usize];

                    if let Some(h) = ray_triangle(ray, a, b, c, t_min, best_t) {
                        if h.t < best_t || (h.t == best_t && tri_idx < best_tri) {
                            best_t = h.t;
                            best_tri = tri_idx;
                            best_bary = Vec3::new(1.0 - h.u - h.v, h.u, h.v);
                        }
                    }
                }
                continue;
            }

            let left = node.left;
            let right = node.right;

            let mut left_t = f32::INFINITY;
            let mut right_t = f32::INFINITY;

            let lnode = self.nodes[left as usize];
            let rnode = self.nodes[right as usize];

            if lnode.bounds.hit(ray, t_min, best_t) {
                left_t = bounds_near_t(&lnode.bounds, ray);
            }
            if rnode.bounds.hit(ray, t_min, best_t) {
                right_t = bounds_near_t(&rnode.bounds, ray);
            }

            if left_t.is_finite() || right_t.is_finite() {
                if left_t.total_cmp(&right_t).is_gt() {
                    if left_t.is_finite() {
                        stack.push(left);
                    }
                    if right_t.is_finite() {
                        stack.push(right);
                    }
                } else {
                    if right_t.is_finite() {
                        stack.push(right);
                    }
                    if left_t.is_finite() {
                        stack.push(left);
                    }
                }
            }
        }

        if best_tri == usize::MAX {
            return None;
        }

        let tri = mesh.indices[best_tri];
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];

        let position = ray.at(best_t);

        let mut normal = (b - a).cross(c - a);
        if normal.length_squared() > 0.0 {
            normal = normal.normalize();
        } else {
            normal = Vec3::ZERO;
        }

        if !mesh.normals.is_empty() && mesh.normals.len() == mesh.positions.len() {
            let na = mesh.normals[tri[0] as usize];
            let nb = mesh.normals[tri[1] as usize];
            let nc = mesh.normals[tri[2] as usize];
            let n = na * best_bary.x + nb * best_bary.y + nc * best_bary.z;
            if n.length_squared() > 0.0 {
                normal = n.normalize();
            }
        }

        let uv = if !mesh.uvs.is_empty() && mesh.uvs.len() == mesh.positions.len() {
            let uva = mesh.uvs[tri[0] as usize];
            let uvb = mesh.uvs[tri[1] as usize];
            let uvc = mesh.uvs[tri[2] as usize];
            Some(uva * best_bary.x + uvb * best_bary.y + uvc * best_bary.z)
        } else {
            None
        };

        Some(MeshHit {
            tri_index: best_tri,
            t: best_t,
            barycentric: best_bary,
            position,
            normal,
            uv,
        })
    }
}

fn bounds_near_t(bounds: &Aabb, ray: &Ray) -> f32 {
    let mut t_min = 0.0f32;
    let mut t_max = f32::INFINITY;

    let o = ray.origin;
    let d = ray.dir;

    for axis in 0..3 {
        let origin = o[axis];
        let dir = d[axis];
        let minp = bounds.min[axis];
        let maxp = bounds.max[axis];

        if dir.abs() <= DIR_EPS {
            if origin < minp || origin > maxp {
                return f32::INFINITY;
            }
            continue;
        }

        let inv = 1.0 / dir;
        let mut t0 = (minp - origin) * inv;
        let mut t1 = (maxp - origin) * inv;

        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }

        t_min = t_min.max(t0);
        t_max = t_max.min(t1);

        if t_max < t_min {
            return f32::INFINITY;
        }
    }

    t_min
}

fn tri_bounds(mesh: &Mesh, tri_idx: u32) -> Aabb {
    let tri = mesh.indices[tri_idx as usize];
    let a = mesh.positions[tri[0] as usize];
    let b = mesh.positions[tri[1] as usize];
    let c = mesh.positions[tri[2] as usize];
    Aabb::from_points(a, b, c)
}

fn tri_centroid(mesh: &Mesh, tri_idx: u32) -> Vec3 {
    tri_bounds(mesh, tri_idx).centroid()
}

fn build_node(
    mesh: &Mesh,
    tri_indices: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let mut bounds = Aabb::empty();
    for i in start..end {
        let b = tri_bounds(mesh, tri_indices[i]);
        bounds = bounds.union(&b);
    }

    let count = end - start;
    if count <= LEAF_TRI_LIMIT {
        let idx = nodes.len();
        nodes.push(BvhNode::leaf(bounds, start as u32, count as u32));
        return idx;
    }

    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(-f32::INFINITY);

    for i in start..end {
        let c = tri_centroid(mesh, tri_indices[i]);
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }

    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    if extent[axis] <= 0.0 {
        let idx = nodes.len();
        nodes.push(BvhNode::leaf(bounds, start as u32, count as u32));
        return idx;
    }

    tri_indices[start..end].sort_by(|&a, &b| {
        let ca = tri_centroid(mesh, a)[axis];
        let cb = tri_centroid(mesh, b)[axis];
        ca.total_cmp(&cb).then_with(|| a.cmp(&b))
    });

    let mid = start + count / 2;

    let idx = nodes.len();
    nodes.push(BvhNode::leaf(bounds, 0, 0));

    let left = build_node(mesh, tri_indices, start, mid, nodes) as u32;
    let right = build_node(mesh, tri_indices, mid, end, nodes) as u32;

    nodes[idx] = BvhNode::inner(bounds, left, right);
    idx
}

#[derive(Clone, Debug)]
pub struct SceneBvh {
    meshes: Vec<MeshBvh>,
}

impl SceneBvh {
    pub fn build(scene: &Scene) -> Self {
        let mut meshes = Vec::with_capacity(scene.object_count());
        for (mesh, _mat, _xf) in scene.iter_objects() {
            meshes.push(MeshBvh::build(mesh));
        }
        Self { meshes }
    }

    pub fn pick(&self, scene: &Scene, ray_world: &Ray) -> Option<SceneHit> {
        let mut best_t = f32::INFINITY;
        let mut best: Option<SceneHit> = None;

        for (object_index, (mesh, _mat, xf)) in scene.iter_objects().enumerate() {
            let bvh = match self.meshes.get(object_index) {
                Some(b) => b,
                None => continue,
            };

            let (ray_local, world_from_local, normal_mat) = transform_ray_to_local(ray_world, xf);

            if let Some(hit_local) = bvh.intersect_range(mesh, &ray_local, 0.0, f32::INFINITY) {
                let pos_world = world_from_local.transform_point3(hit_local.position);
                let t_world = (pos_world - ray_world.origin).dot(ray_world.dir);

                if t_world.is_finite()
                    && t_world >= 0.0
                    && (t_world < best_t
                        || (t_world == best_t
                            && object_index
                                < best.as_ref().map(|h| h.object_index).unwrap_or(usize::MAX)))
                {
                    let mut n_world = normal_mat.transform_vector3(hit_local.normal);
                    if n_world.length_squared() > 0.0 {
                        n_world = n_world.normalize();
                    } else {
                        n_world = Vec3::ZERO;
                    }

                    best_t = t_world;
                    best = Some(SceneHit {
                        object_index,
                        tri_index: hit_local.tri_index,
                        t: t_world,
                        position: pos_world,
                        normal: n_world,
                        uv: hit_local.uv,
                    });
                }
            }
        }

        best
    }
}

fn transform_ray_to_local(ray: &Ray, xf: &Transform) -> (Ray, Mat4, Mat4) {
    let world_from_local = xf.to_mat4();
    let local_from_world = world_from_local.inverse();

    let origin_local = local_from_world.transform_point3(ray.origin);
    let mut dir_local = local_from_world.transform_vector3(ray.dir);
    if dir_local.length_squared() > 0.0 {
        dir_local = dir_local.normalize();
    } else {
        dir_local = Vec3::ZERO;
    }

    let normal_mat = local_from_world.transpose();

    (
        Ray::new(origin_local, dir_local),
        world_from_local,
        normal_mat,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn bvh_picks_same_as_bruteforce_unit_cube() {
        let mesh = Mesh::unit_cube();
        let bvh = MeshBvh::build(&mesh);

        let ray = Ray::new(Vec3::new(0.25, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0));

        let hit_bvh = bvh.intersect(&mesh, &ray).expect("bvh hit");
        assert!(hit_bvh.t.is_finite());
        assert!((hit_bvh.position.z - 0.5).abs() < 1e-5);

        let mut best_t = f32::INFINITY;
        let mut best_tri = usize::MAX;
        for (ti, tri) in mesh.indices.iter().enumerate() {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            if let Some(h) = ray_triangle(&ray, a, b, c, 0.0, best_t) {
                if h.t < best_t || (h.t == best_t && ti < best_tri) {
                    best_t = h.t;
                    best_tri = ti;
                }
            }
        }

        assert_eq!(hit_bvh.tri_index, best_tri);
        assert!((hit_bvh.t - best_t).abs() < 1e-6);
    }

    #[test]
    fn scene_bvh_pick_returns_deterministic_hit() {
        let mut scene = Scene::new();
        let mesh = Mesh::unit_triangle();
        let mat = crate::Material::default();
        let xf = Transform::IDENTITY;
        scene.add_object(mesh, xf, mat);

        let sbvh = SceneBvh::build(&scene);

        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = sbvh.pick(&scene, &ray).expect("scene hit");

        assert_eq!(hit.object_index, 0);
        assert!(hit.t.is_finite());
        assert!(hit.position.z.abs() < 1e-6);
    }

    #[test]
    fn scene_bvh_pick_ignores_objects_added_after_build() {
        let mut scene = Scene::new();
        let mat = crate::Material::default();
        scene.add_object(Mesh::unit_triangle(), Transform::IDENTITY, mat.clone());
        let sbvh = SceneBvh::build(&scene);

        scene.add_object(
            Mesh::unit_triangle(),
            Transform::from_translation(Vec3::new(10.0, 10.0, 0.0)),
            mat,
        );

        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = sbvh.pick(&scene, &ray).expect("scene hit");

        assert_eq!(hit.object_index, 0);
        assert!(hit.t.is_finite());
    }
}
