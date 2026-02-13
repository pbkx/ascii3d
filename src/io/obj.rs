use crate::{io::mtl::parse_mtl, Material, Mesh};
use glam::{Vec2, Vec3};
use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs,
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
    let base = idxs[0];
    for i in 1..(idxs.len() - 1) {
        mesh.indices.push([base, idxs[i], idxs[i + 1]]);
        tri_materials.push(mat);
    }
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

pub fn load_obj_with_mtl_mesh_materials(path: impl AsRef<Path>) -> Result<(Mesh, Vec<Material>), ObjError> {
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
}
