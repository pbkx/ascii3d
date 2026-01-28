use glam::Vec3;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    pub normals: Vec<Vec3>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_positions(mut self, positions: Vec<Vec3>) -> Self {
        self.positions = positions;
        self
    }

    pub fn with_indices(mut self, indices: Vec<[u32; 3]>) -> Self {
        self.indices = indices;
        self
    }

    pub fn with_normals(mut self, normals: Vec<Vec3>) -> Self {
        self.normals = normals;
        self
    }

    pub fn push_triangle(&mut self, i0: u32, i1: u32, i2: u32) {
        self.indices.push([i0, i1, i2]);
    }
}
