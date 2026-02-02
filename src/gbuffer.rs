use glam::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GBufferPixel {
    pub depth: f32,
    pub normal: Vec3,
    pub albedo: Vec3,
}

#[derive(Clone, Debug)]
pub struct GBuffer {
    width: usize,
    height: usize,
    depth: Vec<f32>,
    normal: Vec<Vec3>,
    albedo: Vec<Vec3>,
}

impl GBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width.saturating_mul(height);
        Self {
            width,
            height,
            depth: vec![f32::INFINITY; n],
            normal: vec![Vec3::ZERO; n],
            albedo: vec![Vec3::ZERO; n],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self) {
        self.depth.fill(f32::INFINITY);
        self.normal.fill(Vec3::ZERO);
        self.albedo.fill(Vec3::ZERO);
    }

    pub fn depth_slice(&self) -> &[f32] {
        &self.depth
    }

    pub fn normal_slice(&self) -> &[Vec3] {
        &self.normal
    }

    pub fn albedo_slice(&self) -> &[Vec3] {
        &self.albedo
    }

    pub fn try_write(&mut self, x: usize, y: usize, depth: f32, normal: Vec3, albedo: Vec3) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let i = y * self.width + x;
        if depth < self.depth[i] {
            self.depth[i] = depth;
            self.normal[i] = normal;
            self.albedo[i] = albedo;
            true
        } else {
            false
        }
    }

    pub fn at(&self, x: usize, y: usize) -> Option<GBufferPixel> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = y * self.width + x;
        Some(GBufferPixel {
            depth: self.depth[i],
            normal: self.normal[i],
            albedo: self.albedo[i],
        })
    }
}
