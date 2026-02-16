use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureWrapMode {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSampler {
    pub wrap_s: TextureWrapMode,
    pub wrap_t: TextureWrapMode,
    pub min_filter: TextureFilter,
    pub mag_filter: TextureFilter,
}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureFilter::Nearest,
            mag_filter: TextureFilter::Nearest,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub sampler: TextureSampler,
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
            sampler: TextureSampler::default(),
        })
    }

    fn texel_index(&self, x: u32, y: u32) -> usize {
        ((y * self.width + x) as usize) * 4
    }

    fn texel_rgb(&self, x: u32, y: u32) -> Vec3 {
        let i = self.texel_index(x, y);
        let r = self.rgba8[i] as f32 / 255.0;
        let g = self.rgba8[i + 1] as f32 / 255.0;
        let b = self.rgba8[i + 2] as f32 / 255.0;
        Vec3::new(r, g, b)
    }

    fn wrap_coord(c: f32, mode: TextureWrapMode) -> f32 {
        match mode {
            TextureWrapMode::Repeat => c.rem_euclid(1.0),
            TextureWrapMode::ClampToEdge => c.clamp(0.0, 1.0),
            TextureWrapMode::MirroredRepeat => {
                let t = c.rem_euclid(2.0);
                if t <= 1.0 {
                    t
                } else {
                    2.0 - t
                }
            }
        }
    }

    fn wrap_texel_index(i: i32, dim: u32, mode: TextureWrapMode) -> u32 {
        let dim_i = dim.max(1) as i32;
        match mode {
            TextureWrapMode::Repeat => i.rem_euclid(dim_i) as u32,
            TextureWrapMode::ClampToEdge | TextureWrapMode::MirroredRepeat => {
                i.clamp(0, dim_i - 1) as u32
            }
        }
    }

    fn sample_nearest_wrapped(&self, uv: Vec2) -> Vec3 {
        if self.width == 0 || self.height == 0 {
            return Vec3::ZERO;
        }

        let u = Self::wrap_coord(uv.x, self.sampler.wrap_s);
        let v = Self::wrap_coord(uv.y, self.sampler.wrap_t);

        let fx = (u * self.width as f32).floor();
        let fy = (v * self.height as f32).floor();

        let x = (fx as i32).clamp(0, self.width as i32 - 1) as u32;
        let y = (fy as i32).clamp(0, self.height as i32 - 1) as u32;

        self.texel_rgb(x, y)
    }

    fn sample_linear_wrapped(&self, uv: Vec2) -> Vec3 {
        if self.width == 0 || self.height == 0 {
            return Vec3::ZERO;
        }

        let u = Self::wrap_coord(uv.x, self.sampler.wrap_s);
        let v = Self::wrap_coord(uv.y, self.sampler.wrap_t);

        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let ix0 = Self::wrap_texel_index(x0, self.width, self.sampler.wrap_s);
        let iy0 = Self::wrap_texel_index(y0, self.height, self.sampler.wrap_t);
        let ix1 = Self::wrap_texel_index(x1, self.width, self.sampler.wrap_s);
        let iy1 = Self::wrap_texel_index(y1, self.height, self.sampler.wrap_t);

        let c00 = self.texel_rgb(ix0, iy0);
        let c10 = self.texel_rgb(ix1, iy0);
        let c01 = self.texel_rgb(ix0, iy1);
        let c11 = self.texel_rgb(ix1, iy1);

        let cx0 = c00.lerp(c10, tx);
        let cx1 = c01.lerp(c11, tx);
        cx0.lerp(cx1, ty)
    }

    pub fn sample_nearest(&self, uv: Vec2) -> Vec3 {
        self.sample_nearest_wrapped(uv)
    }

    pub fn sample(&self, uv: Vec2) -> Vec3 {
        let wants_linear = matches!(self.sampler.mag_filter, TextureFilter::Linear)
            || matches!(self.sampler.min_filter, TextureFilter::Linear);
        if wants_linear {
            self.sample_linear_wrapped(uv)
        } else {
            self.sample_nearest_wrapped(uv)
        }
    }

    pub fn with_sampler(mut self, sampler: TextureSampler) -> Self {
        self.sampler = sampler;
        self
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
            sampler: TextureSampler::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex2() -> Texture {
        Texture::from_rgba8(
            2,
            1,
            vec![
                255, 0, 0, 255, //
                0, 255, 0, 255, //
            ],
        )
        .unwrap()
    }

    #[test]
    fn sampler_wrap_modes_affect_coordinates() {
        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::ClampToEdge,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureFilter::Nearest,
            mag_filter: TextureFilter::Nearest,
        });
        let c = t.sample(Vec2::new(1.2, 0.0));
        assert!(c.y > 0.9 && c.x < 0.1);

        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::MirroredRepeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureFilter::Nearest,
            mag_filter: TextureFilter::Nearest,
        });
        let c = t.sample(Vec2::new(1.2, 0.0));
        assert!(c.y > 0.9 && c.x < 0.1);
    }

    #[test]
    fn linear_sampling_blends_adjacent_texels() {
        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureFilter::Linear,
            mag_filter: TextureFilter::Linear,
        });
        let c = t.sample(Vec2::new(0.99, 0.0));
        assert!(c.x > 0.2 && c.y > 0.2);
    }
}
