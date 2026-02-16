use glam::{Vec2, Vec3, Vec4};
use std::fmt;

pub type Tri = [u32; 3];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshValidationError {
    PositionNotFinite {
        index: usize,
    },
    NormalNotFinite {
        index: usize,
    },
    UvNotFinite {
        index: usize,
    },
    ColorNotFinite {
        index: usize,
    },
    NormalsLenMismatch {
        normals: usize,
        positions: usize,
    },
    UvsLenMismatch {
        uvs: usize,
        positions: usize,
    },
    ColorsLenMismatch {
        colors: usize,
        positions: usize,
    },
    TriangleIndexOutOfBounds {
        tri: usize,
        index: u32,
        vertex_count: usize,
    },
}

impl fmt::Display for MeshValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            MeshValidationError::PositionNotFinite { index } => {
                write!(f, "position_not_finite:{index}")
            }
            MeshValidationError::NormalNotFinite { index } => {
                write!(f, "normal_not_finite:{index}")
            }
            MeshValidationError::UvNotFinite { index } => {
                write!(f, "uv_not_finite:{index}")
            }
            MeshValidationError::ColorNotFinite { index } => {
                write!(f, "color_not_finite:{index}")
            }
            MeshValidationError::NormalsLenMismatch { normals, positions } => {
                write!(f, "normals_len_mismatch:{normals}:{positions}")
            }
            MeshValidationError::UvsLenMismatch { uvs, positions } => {
                write!(f, "uvs_len_mismatch:{uvs}:{positions}")
            }
            MeshValidationError::ColorsLenMismatch { colors, positions } => {
                write!(f, "colors_len_mismatch:{colors}:{positions}")
            }
            MeshValidationError::TriangleIndexOutOfBounds {
                tri,
                index,
                vertex_count,
            } => {
                write!(f, "tri_oob:{tri}:{index}:{vertex_count}")
            }
        }
    }
}

impl std::error::Error for MeshValidationError {}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<Tri>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<Vec4>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_positions(mut self, positions: Vec<Vec3>) -> Self {
        self.positions = positions;
        self
    }

    pub fn with_indices(mut self, indices: Vec<Tri>) -> Self {
        self.indices = indices;
        self
    }

    pub fn with_normals(mut self, normals: Vec<Vec3>) -> Self {
        self.normals = normals;
        self
    }

    pub fn with_uvs(mut self, uvs: Vec<Vec2>) -> Self {
        self.uvs = uvs;
        self
    }

    pub fn with_colors(mut self, colors: Vec<Vec4>) -> Self {
        self.colors = colors;
        self
    }

    pub fn push_triangle(&mut self, i0: u32, i1: u32, i2: u32) {
        self.indices.push([i0, i1, i2]);
    }

    pub fn unit_triangle() -> Self {
        let positions = vec![
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
        ];
        let indices = vec![[0, 1, 2]];
        Self {
            positions,
            indices,
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
        }
    }

    pub fn unit_cube() -> Self {
        let positions = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];

        let indices = vec![
            [4, 5, 6],
            [4, 6, 7],
            [1, 0, 3],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [5, 1, 2],
            [5, 2, 6],
            [3, 7, 6],
            [3, 6, 2],
            [0, 1, 5],
            [0, 5, 4],
        ];

        Self {
            positions,
            indices,
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
        }
    }

    pub fn build_bvh(&self) -> crate::bvh::MeshBvh {
        crate::bvh::MeshBvh::build(self)
    }

    pub fn validate_basic(&self) -> Result<(), MeshValidationError> {
        for (i, p) in self.positions.iter().enumerate() {
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                return Err(MeshValidationError::PositionNotFinite { index: i });
            }
        }

        if !self.normals.is_empty() && self.normals.len() != self.positions.len() {
            return Err(MeshValidationError::NormalsLenMismatch {
                normals: self.normals.len(),
                positions: self.positions.len(),
            });
        }

        for (i, n) in self.normals.iter().enumerate() {
            if !(n.x.is_finite() && n.y.is_finite() && n.z.is_finite()) {
                return Err(MeshValidationError::NormalNotFinite { index: i });
            }
        }

        if !self.uvs.is_empty() && self.uvs.len() != self.positions.len() {
            return Err(MeshValidationError::UvsLenMismatch {
                uvs: self.uvs.len(),
                positions: self.positions.len(),
            });
        }

        for (i, uv) in self.uvs.iter().enumerate() {
            if !(uv.x.is_finite() && uv.y.is_finite()) {
                return Err(MeshValidationError::UvNotFinite { index: i });
            }
        }

        if !self.colors.is_empty() && self.colors.len() != self.positions.len() {
            return Err(MeshValidationError::ColorsLenMismatch {
                colors: self.colors.len(),
                positions: self.positions.len(),
            });
        }
        for (i, c) in self.colors.iter().enumerate() {
            if !(c.x.is_finite() && c.y.is_finite() && c.z.is_finite() && c.w.is_finite()) {
                return Err(MeshValidationError::ColorNotFinite { index: i });
            }
        }

        let vc = self.positions.len();
        for (ti, t) in self.indices.iter().enumerate() {
            for &ix in t.iter() {
                if ix as usize >= vc {
                    return Err(MeshValidationError::TriangleIndexOutOfBounds {
                        tri: ti,
                        index: ix,
                        vertex_count: vc,
                    });
                }
            }
        }

        Ok(())
    }

    pub fn assert_invariants(&self) {
        if let Err(e) = self.validate_basic() {
            panic!("{e:?}");
        }
    }

    pub fn ensure_normals(&mut self) {
        if self.positions.is_empty() {
            self.normals.clear();
            return;
        }

        if self.normals.len() != self.positions.len() {
            self.normals
                .resize(self.positions.len(), Vec3::new(0.0, 0.0, 0.0));
        }

        fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
            let l = v.length();
            if l > 1e-8 {
                v / l
            } else {
                fallback
            }
        }

        fn angle(u: Vec3, v: Vec3) -> f32 {
            let uu = normalize_or(u, Vec3::new(0.0, 0.0, 0.0));
            let vv = normalize_or(v, Vec3::new(0.0, 0.0, 0.0));
            let dot = uu.dot(vv).clamp(-1.0, 1.0);
            let cr = uu.cross(vv).length();
            cr.atan2(dot)
        }

        let mut acc = vec![Vec3::new(0.0, 0.0, 0.0); self.positions.len()];

        for t in &self.indices {
            let [i0, i1, i2] = *t;
            let (i0u, i1u, i2u) = (i0 as usize, i1 as usize, i2 as usize);
            if i0u >= self.positions.len()
                || i1u >= self.positions.len()
                || i2u >= self.positions.len()
            {
                continue;
            }

            let a = self.positions[i0u];
            let b = self.positions[i1u];
            let c = self.positions[i2u];

            let fn_raw = (b - a).cross(c - a);
            let fn_u = normalize_or(fn_raw, Vec3::new(0.0, 0.0, 0.0));
            if fn_u.length() < 1e-8 {
                continue;
            }

            let wa = angle(b - a, c - a);
            let wb = angle(c - b, a - b);
            let wc = angle(a - c, b - c);

            acc[i0u] += fn_u * wa;
            acc[i1u] += fn_u * wb;
            acc[i2u] += fn_u * wc;
        }

        for (i, v) in acc.iter().enumerate().take(self.normals.len()) {
            self.normals[i] = v.normalize_or_zero();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mesh, MeshValidationError};
    use glam::Vec3;

    #[test]
    fn unit_triangle_invariants() {
        let m = Mesh::unit_triangle();
        assert_eq!(m.validate_basic(), Ok(()));
    }

    #[test]
    fn ensure_normals_unit_triangle() {
        let mut m = Mesh::unit_triangle();
        m.ensure_normals();
        assert_eq!(m.normals.len(), m.positions.len());
        for n in &m.normals {
            assert!((n.length() - 1.0).abs() < 1e-4);
            assert!(n.z > 0.9);
            assert!(n.x.abs() < 1e-4);
            assert!(n.y.abs() < 1e-4);
        }
    }

    #[test]
    fn ensure_normals_preserves_existing() {
        let mut m = Mesh::unit_triangle();
        m.normals = vec![
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, 4.0),
        ];
        m.ensure_normals();
        for n in &m.normals {
            assert!((n.length() - 1.0).abs() < 1e-4);
            assert!(n.z > 0.99);
        }
    }

    #[test]
    fn validate_rejects_oob_indices() {
        let mut m = Mesh::unit_triangle();
        m.indices = vec![[0, 1, 3]];
        assert!(matches!(
            m.validate_basic(),
            Err(MeshValidationError::TriangleIndexOutOfBounds { .. })
        ));
    }
}
