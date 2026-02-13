use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Texture {
    pub fn from_rgba8(width: u32, height: u32, rgba8: Vec<u8>) -> Option<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if rgba8.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            rgba8,
        })
    }

    fn texel_index(&self, x: u32, y: u32) -> usize {
        ((y * self.width + x) as usize) * 4
    }

    pub fn sample_nearest(&self, uv: Vec2) -> Vec3 {
        if self.width == 0 || self.height == 0 {
            return Vec3::ZERO;
        }

        let u = uv.x - uv.x.floor();
        let v = uv.y - uv.y.floor();

        let fx = (u * self.width as f32).floor();
        let fy = (v * self.height as f32).floor();

        let x = (fx as i32).clamp(0, self.width as i32 - 1) as u32;
        let y = (fy as i32).clamp(0, self.height as i32 - 1) as u32;

        let i = self.texel_index(x, y);
        let r = self.rgba8[i] as f32 / 255.0;
        let g = self.rgba8[i + 1] as f32 / 255.0;
        let b = self.rgba8[i + 2] as f32 / 255.0;
        Vec3::new(r, g, b)
    }

    pub fn checkerboard_rgba8(width: u32, height: u32, cell_px: u32) -> Self {
        let cell_px = cell_px.max(1);
        let len = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let mut rgba8 = vec![0u8; len];

        for y in 0..height {
            for x in 0..width {
                let on = ((x / cell_px) + (y / cell_px)) & 1 == 0;
                let c = if on { 255u8 } else { 0u8 };
                let i = ((y * width + x) as usize) * 4;
                rgba8[i] = c;
                rgba8[i + 1] = c;
                rgba8[i + 2] = c;
                rgba8[i + 3] = 255u8;
            }
        }

        Self {
            width,
            height,
            rgba8,
        }
    }
}
