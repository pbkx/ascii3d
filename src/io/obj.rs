use crate::{io::mtl::parse_mtl, Material, Mesh};
use glam::{Vec2, Vec3};
use std::{
    collections::HashMap,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct LoadedObj {
    pub mesh: Mesh,
    pub material_names: Vec<String>,
    pub materials: Vec<Material>,
    pub tri_materials: Vec<u32>,
}

#[derive(Clone, Debug)]
pub enum ObjError {
    Io,
    ParseFloat,
    ParseIndex,
    MissingVertex,
    MissingFaceVertex,
    MtlParse,
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => write!(f, "io error"),
            Self::ParseFloat => write!(f, "failed to parse float"),
            Self::ParseIndex => write!(f, "failed to parse index"),
            Self::MissingVertex => write!(f, "vertex index out of range"),
            Self::MissingFaceVertex => write!(f, "face missing vertex"),
            Self::MtlParse => write!(f, "failed to parse mtl"),
        }
    }
}

impl Error for ObjError {}

fn parse_f32(s: &str) -> Result<f32, ObjError> {
    s.parse::<f32>().map_err(|_| ObjError::ParseFloat)
}

fn parse_i32(s: &str) -> Result<i32, ObjError> {
    s.parse::<i32>().map_err(|_| ObjError::ParseIndex)
}

fn resolve_index(idx: i32, len: usize) -> Result<usize, ObjError> {
    if idx == 0 {
        return Err(ObjError::ParseIndex);
    }
    if idx > 0 {
        let i = (idx - 1) as isize;
        if i < 0 || i as usize >= len {
            return Err(ObjError::MissingVertex);
        }
        return Ok(i as usize);
    }
    let i = len as isize + idx as isize;
    if i < 0 || i as usize >= len {
        return Err(ObjError::MissingVertex);
    }
    Ok(i as usize)
}

#[derive(Clone, Copy, Debug)]
struct FaceIndex {
    v: i32,
    vt: i32,
    vn: i32,
}

fn parse_face_index(tok: &str) -> Result<FaceIndex, ObjError> {
    let mut parts = tok.split('/');
    let v = parts.next().ok_or(ObjError::MissingFaceVertex)?;
    let v = parse_i32(v)?;

    let mut vt: i32 = 0;
    let mut vn: i32 = 0;

    if let Some(vt_s) = parts.next() {
        if !vt_s.is_empty() {
            vt = parse_i32(vt_s)?;
        }
    }

    if let Some(vn_s) = parts.next() {
        if !vn_s.is_empty() {
            vn = parse_i32(vn_s)?;
        }
    }

    Ok(FaceIndex { v, vt, vn })
}

#[derive(Clone, Debug)]
struct ParsedFace {
    corners: Vec<FaceIndex>,
    mat: u32,
}

#[derive(Clone, Debug)]
struct ParsedObj {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    texcoords: Vec<Vec2>,
    faces: Vec<ParsedFace>,
    mtllib: Option<String>,
}

fn parse_obj(src: &str, mat_map: &HashMap<String, u32>) -> Result<ParsedObj, ObjError> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut texcoords: Vec<Vec2> = Vec::new();
    let mut faces: Vec<ParsedFace> = Vec::new();
    let mut mtllib: Option<String> = None;

    let mut current_mat: u32 = 0;

    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else {
            continue;
        };
        match key {
            "v" => {
                let x = it.next().ok_or(ObjError::ParseFloat)?;
                let y = it.next().ok_or(ObjError::ParseFloat)?;
                let z = it.next().ok_or(ObjError::ParseFloat)?;
                positions.push(Vec3::new(parse_f32(x)?, parse_f32(y)?, parse_f32(z)?));
            }
            "vn" => {
                let x = it.next().ok_or(ObjError::ParseFloat)?;
                let y = it.next().ok_or(ObjError::ParseFloat)?;
                let z = it.next().ok_or(ObjError::ParseFloat)?;
                normals.push(Vec3::new(parse_f32(x)?, parse_f32(y)?, parse_f32(z)?));
            }
            "vt" => {
                let u = it.next().ok_or(ObjError::ParseFloat)?;
                let v = it.next().ok_or(ObjError::ParseFloat)?;
                texcoords.push(Vec2::new(parse_f32(u)?, parse_f32(v)?));
            }
            "f" => {
                let corners: Result<Vec<FaceIndex>, ObjError> = it.map(parse_face_index).collect();
                let corners = corners?;
                if corners.len() < 3 {
                    continue;
                }
                faces.push(ParsedFace {
                    corners,
                    mat: current_mat,
                });
            }
            "usemtl" => {
                if let Some(name) = it.next() {
                    current_mat = *mat_map.get(name).unwrap_or(&0);
                }
            }
            "mtllib" => {
                if mtllib.is_none() {
                    mtllib = it.next().map(|s| s.to_string());
                }
            }
            _ => {}
        }
    }

    Ok(ParsedObj {
        positions,
        normals,
        texcoords,
        faces,
        mtllib,
    })
}

fn build_mesh(
    parsed: ParsedObj,
    mut materials: Vec<Material>,
    mut names: Vec<String>,
) -> Result<LoadedObj, ObjError> {
    let mut all_corner_normals = !parsed.normals.is_empty();
    if all_corner_normals {
        for face in &parsed.faces {
            for c in &face.corners {
                if c.vn == 0 {
                    all_corner_normals = false;
                    break;
                }
            }
            if !all_corner_normals {
                break;
            }
        }
    }

    let use_obj_normals = all_corner_normals && !parsed.normals.is_empty();
    let has_texcoords = !parsed.texcoords.is_empty();

    let mut mesh = Mesh::new();
    let mut tri_materials: Vec<u32> = Vec::new();

    if use_obj_normals {
        let mut map: HashMap<(i32, i32, i32), u32> = HashMap::new();
        for face in parsed.faces {
            let mut idxs: Vec<u32> = Vec::with_capacity(face.corners.len());
            for c in face.corners {
                let key = (c.v, c.vt, c.vn);
                let out = if let Some(&o) = map.get(&key) {
                    o
                } else {
                    let pi = resolve_index(c.v, parsed.positions.len())?;
                    let ni = resolve_index(c.vn, parsed.normals.len())?;
                    let uv = if has_texcoords && c.vt != 0 {
                        let ti = resolve_index(c.vt, parsed.texcoords.len())?;
                        parsed.texcoords[ti]
                    } else {
                        Vec2::ZERO
                    };

                    let out = mesh.positions.len() as u32;
                    mesh.positions.push(parsed.positions[pi]);
                    mesh.normals.push(parsed.normals[ni]);
                    if has_texcoords {
                        mesh.uvs.push(uv);
                    }
                    map.insert(key, out);
                    out
                };
                idxs.push(out);
            }
            triangulate(&idxs, face.mat, &mut mesh, &mut tri_materials);
        }
    } else {
        let mut map: HashMap<(i32, i32), u32> = HashMap::new();
        for face in parsed.faces {
            let mut idxs: Vec<u32> = Vec::with_capacity(face.corners.len());
            for c in face.corners {
                let key = (c.v, c.vt);
                let out = if let Some(&o) = map.get(&key) {
                    o
                } else {
                    let pi = resolve_index(c.v, parsed.positions.len())?;
                    let uv = if has_texcoords && c.vt != 0 {
                        let ti = resolve_index(c.vt, parsed.texcoords.len())?;
                        parsed.texcoords[ti]
                    } else {
                        Vec2::ZERO
                    };

                    let out = mesh.positions.len() as u32;
                    mesh.positions.push(parsed.positions[pi]);
                    if has_texcoords {
                        mesh.uvs.push(uv);
                    }
                    map.insert(key, out);
                    out
                };
                idxs.push(out);
            }
            triangulate(&idxs, face.mat, &mut mesh, &mut tri_materials);
        }
    }

    if mesh.normals.is_empty() {
        mesh.ensure_normals();
    }

    if materials.is_empty() {
        names.insert(0, "default".to_string());
        materials.insert(0, Material::default());
    }

    Ok(LoadedObj {
        mesh,
        material_names: names,
        materials,
        tri_materials,
    })
}

fn triangulate(idxs: &[u32], mat: u32, mesh: &mut Mesh, tri_materials: &mut Vec<u32>) {
    if idxs.len() < 3 {
        return;
    }
    if idxs.len() == 3 {
        mesh.indices.push([idxs[0], idxs[1], idxs[2]]);
        tri_materials.push(mat);
        return;
    }

    if triangulate_ear_clip(idxs, mat, mesh, tri_materials) {
        return;
    }

    let base = idxs[0];
    for i in 1..(idxs.len() - 1) {
        mesh.indices.push([base, idxs[i], idxs[i + 1]]);
        tri_materials.push(mat);
    }
}

fn triangulate_ear_clip(idxs: &[u32], mat: u32, mesh: &mut Mesh, tri_materials: &mut Vec<u32>) -> bool {
    const EPS: f32 = 1e-8;
    let mut poly3: Vec<Vec3> = Vec::with_capacity(idxs.len());
    for &idx in idxs {
        let Some(p) = mesh.positions.get(idx as usize).copied() else {
            return false;
        };
        poly3.push(p);
    }

    let normal = polygon_normal_newell(&poly3);
    if normal.length_squared() <= EPS {
        return false;
    }
    let an = normal.abs();
    let drop_axis = if an.x >= an.y && an.x >= an.z {
        0
    } else if an.y >= an.z {
        1
    } else {
        2
    };

    let poly2: Vec<Vec2> = poly3
        .iter()
        .copied()
        .map(|p| project_to_2d(p, drop_axis))
        .collect();
    let signed_area = polygon_signed_area_2d(&poly2);
    if signed_area.abs() <= EPS {
        return false;
    }
    let winding = if signed_area >= 0.0 { 1.0 } else { -1.0 };

    let mut rem: Vec<usize> = (0..idxs.len()).collect();
    let mut guard = 0usize;
    let guard_max = idxs.len().saturating_mul(idxs.len()).saturating_add(1);
    while rem.len() > 3 {
        guard = guard.saturating_add(1);
        if guard > guard_max {
            return false;
        }

        let len = rem.len();
        let mut ear_found = false;
        for i in 0..len {
            let ip = rem[(i + len - 1) % len];
            let ic = rem[i];
            let inx = rem[(i + 1) % len];

            let a = poly2[ip];
            let b = poly2[ic];
            let c = poly2[inx];
            let turn = cross2(b - a, c - a);
            if turn * winding <= EPS {
                continue;
            }

            let mut contains = false;
            for &j in rem.iter() {
                if j == ip || j == ic || j == inx {
                    continue;
                }
                if point_in_triangle_2d(poly2[j], a, b, c) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }

            mesh.indices.push([idxs[ip], idxs[ic], idxs[inx]]);
            tri_materials.push(mat);
            rem.remove(i);
            ear_found = true;
            break;
        }

        if !ear_found {
            return false;
        }
    }

    if rem.len() == 3 {
        mesh.indices.push([idxs[rem[0]], idxs[rem[1]], idxs[rem[2]]]);
        tri_materials.push(mat);
        true
    } else {
        false
    }
}

fn polygon_normal_newell(poly: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        n.x += (p.y - q.y) * (p.z + q.z);
        n.y += (p.z - q.z) * (p.x + q.x);
        n.z += (p.x - q.x) * (p.y + q.y);
    }
    n
}

fn project_to_2d(p: Vec3, drop_axis: usize) -> Vec2 {
    match drop_axis {
        0 => Vec2::new(p.y, p.z),
        1 => Vec2::new(p.x, p.z),
        _ => Vec2::new(p.x, p.y),
    }
}

fn polygon_signed_area_2d(poly: &[Vec2]) -> f32 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        s += p.x * q.y - q.x * p.y;
    }
    0.5 * s
}

fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn point_in_triangle_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    const EPS: f32 = 1e-8;
    let c0 = cross2(b - a, p - a);
    let c1 = cross2(c - b, p - b);
    let c2 = cross2(a - c, p - c);
    let has_neg = c0 < -EPS || c1 < -EPS || c2 < -EPS;
    let has_pos = c0 > EPS || c1 > EPS || c2 > EPS;
    !(has_neg && has_pos)
}

fn read_to_string(path: &Path) -> Result<String, ObjError> {
    fs::read_to_string(path).map_err(|_| ObjError::Io)
}

pub fn load_obj_with_mtl(path: impl AsRef<Path>) -> Result<LoadedObj, ObjError> {
    let path = path.as_ref();
    let obj_src = read_to_string(path)?;

    let mut material_names = vec!["default".to_string()];
    let mut materials = vec![Material::default()];
    let mut mat_map: HashMap<String, u32> = HashMap::new();
    mat_map.insert("default".to_string(), 0);

    let parsed = parse_obj(&obj_src, &mat_map)?;

    if let Some(mtl_name) = &parsed.mtllib {
        let mtl_path = resolve_mtl_path(path, mtl_name);
        if let Ok(mtl_src) = read_to_string(&mtl_path) {
            let lib = parse_mtl(&mtl_src).map_err(|_| ObjError::MtlParse)?;
            for (n, m) in lib.names.into_iter().zip(lib.materials.into_iter()) {
                let idx = materials.len() as u32;
                mat_map.insert(n.clone(), idx);
                material_names.push(n);
                materials.push(m);
            }
        }
    }

    let parsed = parse_obj(&obj_src, &mat_map)?;
    build_mesh(parsed, materials, material_names)
}

fn resolve_mtl_path(obj_path: &Path, mtl_name: &str) -> PathBuf {
    if Path::new(mtl_name).is_absolute() {
        return PathBuf::from(mtl_name);
    }
    let base = obj_path.parent().unwrap_or_else(|| Path::new("."));
    base.join(mtl_name)
}

pub fn load_obj_with_mtl_mesh_materials(
    path: impl AsRef<Path>,
) -> Result<(Mesh, Vec<Material>), ObjError> {
    let loaded = load_obj_with_mtl(path)?;
    Ok((loaded.mesh, loaded.materials))
}

pub fn load_obj_with_mtl_str_mesh_materials(
    obj_src: &str,
    mtl_src: Option<&str>,
) -> Result<(Mesh, Vec<Material>), ObjError> {
    let loaded = load_obj_with_mtl_str(obj_src, mtl_src)?;
    Ok((loaded.mesh, loaded.materials))
}

pub fn load_obj_with_mtl_str(obj_src: &str, mtl_src: Option<&str>) -> Result<LoadedObj, ObjError> {
    let mut material_names = vec!["default".to_string()];
    let mut materials = vec![Material::default()];
    let mut mat_map: HashMap<String, u32> = HashMap::new();
    mat_map.insert("default".to_string(), 0);

    if let Some(mtl_src) = mtl_src {
        let lib = parse_mtl(mtl_src).map_err(|_| ObjError::MtlParse)?;
        for (n, m) in lib.names.into_iter().zip(lib.materials.into_iter()) {
            let idx = materials.len() as u32;
            mat_map.insert(n.clone(), idx);
            material_names.push(n);
            materials.push(m);
        }
    }

    let parsed = parse_obj(obj_src, &mat_map)?;
    build_mesh(parsed, materials, material_names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_triangle_parses_and_assigns_material() {
        let obj = r"
mtllib test.mtl
v 0 0 0
v 1 0 0
v 0 1 0
usemtl Red
f 1 2 3
";
        let mtl = r"
newmtl Red
Kd 1 0 0
";
        let loaded = load_obj_with_mtl_str(obj, Some(mtl)).unwrap();
        assert_eq!(loaded.mesh.indices.len(), 1);
        assert_eq!(loaded.mesh.positions.len(), 3);
        assert_eq!(loaded.materials.len(), 2);
        assert_eq!(loaded.material_names[1], "Red");
        assert_eq!(loaded.tri_materials.len(), 1);
        assert_eq!(loaded.tri_materials[0], 1);
    }

    #[test]
    fn obj_quad_is_triangulated() {
        let obj = r"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
";
        let loaded = load_obj_with_mtl_str(obj, None).unwrap();
        assert_eq!(loaded.mesh.indices.len(), 2);
        loaded.mesh.assert_invariants();
    }

    #[test]
    fn obj_negative_indices_work() {
        let obj = r"
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
";
        let loaded = load_obj_with_mtl_str(obj, None).unwrap();
        assert_eq!(loaded.mesh.indices.len(), 1);
        loaded.mesh.assert_invariants();
    }

    #[test]
    fn obj_uvs_parse_and_align() {
        let obj = r"
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
";
        let loaded = load_obj_with_mtl_str(obj, None).unwrap();
        assert_eq!(loaded.mesh.positions.len(), 3);
        assert_eq!(loaded.mesh.uvs.len(), 3);
        loaded.mesh.assert_invariants();
    }

    #[test]
    fn obj_concave_face_triangulates_without_overlap() {
        let obj = r"
v 0 0 0
v 2 0 0
v 2 2 0
v 1 1 0
v 0 2 0
f 1 2 3 4 5
";
        let loaded = load_obj_with_mtl_str(obj, None).unwrap();
        assert_eq!(loaded.mesh.indices.len(), 3);
        loaded.mesh.assert_invariants();

        let tri_area_sum: f32 = loaded
            .mesh
            .indices
            .iter()
            .map(|tri| {
                let a = loaded.mesh.positions[tri[0] as usize];
                let b = loaded.mesh.positions[tri[1] as usize];
                let c = loaded.mesh.positions[tri[2] as usize];
                0.5 * (b - a).cross(c - a).length()
            })
            .sum();

        assert!((tri_area_sum - 3.0).abs() < 1e-5);
    }
}
