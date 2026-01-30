use crate::Mesh;
use glam::Vec3;
use std::{
    error::Error,
    fmt,
    fs,
    path::Path,
};

#[derive(Clone, Debug)]
pub enum StlError {
    Io,
    InvalidFormat,
    UnexpectedEof,
}

impl fmt::Display for StlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => write!(f, "io error"),
            Self::InvalidFormat => write!(f, "invalid stl format"),
            Self::UnexpectedEof => write!(f, "unexpected end of file"),
        }
    }
}

impl Error for StlError {}

fn read_to_bytes(path: &Path) -> Result<Vec<u8>, StlError> {
    fs::read(path).map_err(|_| StlError::Io)
}

fn read_f32_le(buf: &[u8], off: &mut usize) -> Result<f32, StlError> {
    if *off + 4 > buf.len() {
        return Err(StlError::UnexpectedEof);
    }
    let b = [buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]];
    *off += 4;
    Ok(f32::from_le_bytes(b))
}

fn parse_binary(buf: &[u8]) -> Result<Mesh, StlError> {
    if buf.len() < 84 {
        return Err(StlError::InvalidFormat);
    }
    let mut off = 80;
    let tri_count = u32::from_le_bytes(
        buf.get(off..off + 4)
            .ok_or(StlError::InvalidFormat)?
            .try_into()
            .map_err(|_| StlError::InvalidFormat)?,
    ) as usize;
    off += 4;

    let mut mesh = Mesh::new();
    mesh.positions.reserve(tri_count * 3);
    mesh.normals.reserve(tri_count * 3);
    mesh.indices.reserve(tri_count);

    for t in 0..tri_count {
        let nx = read_f32_le(buf, &mut off)?;
        let ny = read_f32_le(buf, &mut off)?;
        let nz = read_f32_le(buf, &mut off)?;
        let n = Vec3::new(nx, ny, nz);

        let v0 = Vec3::new(read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?);
        let v1 = Vec3::new(read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?);
        let v2 = Vec3::new(read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?, read_f32_le(buf, &mut off)?);

        if off + 2 > buf.len() {
            return Err(StlError::UnexpectedEof);
        }
        off += 2;

        let base = (t * 3) as u32;
        mesh.positions.push(v0);
        mesh.positions.push(v1);
        mesh.positions.push(v2);

        if n.length_squared() > 0.0 {
            let n = n.normalize_or_zero();
            mesh.normals.push(n);
            mesh.normals.push(n);
            mesh.normals.push(n);
        }

        mesh.indices.push([base, base + 1, base + 2]);
    }

    if mesh.normals.is_empty() {
        mesh.ensure_normals();
    }

    Ok(mesh)
}

fn parse_ascii(src: &str) -> Result<Mesh, StlError> {
    let mut verts: Vec<Vec3> = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if !line.starts_with("vertex") {
            continue;
        }
        let mut it = line.split_whitespace();
        let _ = it.next();
        let Some(x) = it.next() else { return Err(StlError::InvalidFormat); };
        let Some(y) = it.next() else { return Err(StlError::InvalidFormat); };
        let Some(z) = it.next() else { return Err(StlError::InvalidFormat); };
        let x = x.parse::<f32>().map_err(|_| StlError::InvalidFormat)?;
        let y = y.parse::<f32>().map_err(|_| StlError::InvalidFormat)?;
        let z = z.parse::<f32>().map_err(|_| StlError::InvalidFormat)?;
        verts.push(Vec3::new(x, y, z));
    }

    if !verts.len().is_multiple_of(3) {
        return Err(StlError::InvalidFormat);
    }

    let tri_count = verts.len() / 3;
    let mut mesh = Mesh::new();
    mesh.positions = verts;
    mesh.indices = (0..tri_count)
        .map(|i| {
            let b = (i * 3) as u32;
            [b, b + 1, b + 2]
        })
        .collect();
    mesh.ensure_normals();
    Ok(mesh)
}

pub fn load_stl(path: impl AsRef<Path>) -> Result<Mesh, StlError> {
    let path = path.as_ref();
    let bytes = read_to_bytes(path)?;
    if bytes.starts_with(b"solid") {
        if let Ok(s) = std::str::from_utf8(&bytes) {
            if s.contains("facet") && s.contains("vertex") {
                return parse_ascii(s);
            }
        }
    }
    parse_binary(&bytes)
}

pub fn load_stl_str(src: &str) -> Result<Mesh, StlError> {
    parse_ascii(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stl_ascii_triangle_parses() {
        let src = r#"
solid s
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid
"#;
        let mesh = load_stl_str(src).unwrap();
        assert_eq!(mesh.indices.len(), 1);
        assert_eq!(mesh.positions.len(), 3);
        mesh.assert_invariants();
    }
}