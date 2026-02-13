use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct GBufferPixel {
    pub depth: f32,
    pub normal: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub ns: f32,
    pub ke: Vec3,
}

#[derive(Debug, Clone)]
pub struct GBuffer {
    width: usize,
    height: usize,
    depth: Vec<f32>,
    nx: Vec<f32>,
    ny: Vec<f32>,
    nz: Vec<f32>,
    kd: Vec<Vec3>,
    ks: Vec<Vec3>,
    ns: Vec<f32>,
    ke: Vec<Vec3>,
}

impl GBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        let mut g = Self {
            width,
            height,
            depth: vec![0.0; n],
            nx: vec![0.0; n],
            ny: vec![0.0; n],
            nz: vec![0.0; n],
            kd: vec![Vec3::ZERO; n],
            ks: vec![Vec3::ZERO; n],
            ns: vec![0.0; n],
            ke: vec![Vec3::ZERO; n],
        };
        g.clear();
        g
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self) {
        let n = self.width * self.height;
        self.depth.fill(1e9);
        self.nx.fill(0.0);
        self.ny.fill(0.0);
        self.nz.fill(0.0);
        self.kd.fill(Vec3::new(0.7, 0.7, 0.7));
        self.ks.fill(Vec3::ZERO);
        self.ns.fill(0.0);
        self.ke.fill(Vec3::ZERO);
        debug_assert_eq!(self.depth.len(), n);
        debug_assert_eq!(self.nx.len(), n);
        debug_assert_eq!(self.ny.len(), n);
        debug_assert_eq!(self.nz.len(), n);
        debug_assert_eq!(self.kd.len(), n);
        debug_assert_eq!(self.ks.len(), n);
        debug_assert_eq!(self.ns.len(), n);
        debug_assert_eq!(self.ke.len(), n);
    }

    fn idx(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.width || y >= self.height {
            None
        } else {
            Some(y * self.width + x)
        }
    }

    pub fn try_write(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
        normal: Vec3,
        kd: Vec3,
        ks: Vec3,
        ns: f32,
        ke: Vec3,
    ) -> bool {
        let Some(i) = self.idx(x, y) else {
            return false;
        };
        if depth < self.depth[i] {
            self.depth[i] = depth;
            self.nx[i] = normal.x;
            self.ny[i] = normal.y;
            self.nz[i] = normal.z;
            self.kd[i] = kd;
            self.ks[i] = ks;
            self.ns[i] = ns;
            self.ke[i] = ke;
            true
        } else {
            false
        }
    }

    pub fn at(&self, x: usize, y: usize) -> Option<GBufferPixel> {
        let i = self.idx(x, y)?;
        Some(GBufferPixel {
            depth: self.depth[i],
            normal: Vec3::new(self.nx[i], self.ny[i], self.nz[i]),
            kd: self.kd[i],
            ks: self.ks[i],
            ns: self.ns[i],
            ke: self.ke[i],
        })
    }

    pub fn depth_slice(&self) -> &[f32] {
        &self.depth
    }

    pub fn nx_slice(&self) -> &[f32] {
        &self.nx
    }

    pub fn ny_slice(&self) -> &[f32] {
        &self.ny
    }

    pub fn nz_slice(&self) -> &[f32] {
        &self.nz
    }

    pub fn albedo_slice(&self) -> &[Vec3] {
        &self.kd
    }

    pub fn kd_slice(&self) -> &[Vec3] {
        &self.kd
    }

    pub fn ks_slice(&self) -> &[Vec3] {
        &self.ks
    }

    pub fn ns_slice(&self) -> &[f32] {
        &self.ns
    }

    pub fn ke_slice(&self) -> &[Vec3] {
        &self.ke
    }
}
