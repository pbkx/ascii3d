use crate::Material;
use glam::Vec3;
use std::{
    collections::HashMap,
    error::Error,
    fmt,
};

#[derive(Clone, Debug)]
pub struct MtlLibrary {
    pub names: Vec<String>,
    pub materials: Vec<Material>,
    index: HashMap<String, usize>,
}

impl MtlLibrary {
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }
}

#[derive(Clone, Debug)]
pub enum MtlError {
    MissingMaterialName,
    ParseFloat,
}

impl fmt::Display for MtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMaterialName => write!(f, "missing material name"),
            Self::ParseFloat => write!(f, "failed to parse float"),
        }
    }
}

impl Error for MtlError {}

fn parse_f32(s: &str) -> Result<f32, MtlError> {
    s.parse::<f32>().map_err(|_| MtlError::ParseFloat)
}

fn parse_vec3(parts: &[&str]) -> Result<Vec3, MtlError> {
    if parts.len() < 3 {
        return Err(MtlError::ParseFloat);
    }
    Ok(Vec3::new(
        parse_f32(parts[0])?,
        parse_f32(parts[1])?,
        parse_f32(parts[2])?,
    ))
}

pub fn parse_mtl(src: &str) -> Result<MtlLibrary, MtlError> {
    let mut names: Vec<String> = Vec::new();
    let mut materials: Vec<Material> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    let mut cur_name: Option<String> = None;
    let mut cur_material = Material::default();

    let mut flush = |name: Option<String>, mat: &Material| -> Result<(), MtlError> {
        if let Some(n) = name {
            if n.is_empty() {
                return Err(MtlError::MissingMaterialName);
            }
            let idx = materials.len();
            names.push(n.clone());
            materials.push(*mat);
            index.insert(n, idx);
        }
        Ok(())
    };

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
            "newmtl" => {
                flush(cur_name.take(), &cur_material)?;
                cur_material = Material::default();
                cur_name = it.next().map(|s| s.to_string());
            }
            "Kd" => {
                let rest: Vec<&str> = it.collect();
                cur_material.albedo = parse_vec3(&rest)?;
            }
            "Ke" => {
                let rest: Vec<&str> = it.collect();
                cur_material.emissive = parse_vec3(&rest)?;
            }
            "d" => {
                if let Some(v) = it.next() {
                    cur_material.alpha = parse_f32(v)?;
                }
            }
            "Tr" => {
                if let Some(v) = it.next() {
                    cur_material.alpha = 1.0 - parse_f32(v)?;
                }
            }
            _ => {}
        }
    }

    flush(cur_name.take(), &cur_material)?;

    Ok(MtlLibrary {
        names,
        materials,
        index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mtl_extracts_basic_fields() {
        let src = r"
newmtl Red
Kd 1 0 0
Ke 0.1 0.2 0.3
d 0.5

newmtl Blue
Kd 0 0 1
Tr 0.25
";
        let lib = parse_mtl(src).unwrap();
        assert_eq!(lib.names.len(), 2);
        assert_eq!(lib.materials.len(), 2);

        let red = lib.materials[lib.index_of("Red").unwrap()];
        assert_eq!(red.albedo, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(red.emissive, Vec3::new(0.1, 0.2, 0.3));
        assert!((red.alpha - 0.5).abs() < 1e-6);

        let blue = lib.materials[lib.index_of("Blue").unwrap()];
        assert_eq!(blue.albedo, Vec3::new(0.0, 0.0, 1.0));
        assert!((blue.alpha - 0.75).abs() < 1e-6);
    }
}