use glam::{Vec2, Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureWrapMode {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureMagFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureMinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSampler {
    pub wrap_s: TextureWrapMode,
    pub wrap_t: TextureWrapMode,
    pub min_filter: TextureMinFilter,
    pub mag_filter: TextureMagFilter,
}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureMinFilter::NearestMipmapLinear,
            mag_filter: TextureMagFilter::Linear,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSampleFootprint {
    pub ddx_uv: Vec2,
    pub ddy_uv: Vec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Debug, PartialEq)]
struct MipLevel {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub sampler: TextureSampler,
    mipmaps: Vec<MipLevel>,
}

impl Texture {
    pub fn from_rgba8(width: u32, height: u32, rgba8: Vec<u8>) -> Option<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if rgba8.len() != expected {
            return None;
        }
        let mipmaps = Self::build_mipmaps(width, height, &rgba8);
        Some(Self {
            width,
            height,
            rgba8,
            sampler: TextureSampler::default(),
            mipmaps,
        })
    }

    fn texel_index_for_level(level: &MipLevel, x: u32, y: u32) -> usize {
        ((y * level.width + x) as usize) * 4
    }

    fn texel_rgba_for_level(level: &MipLevel, x: u32, y: u32) -> Vec4 {
        let i = Self::texel_index_for_level(level, x, y);
        let r = level.rgba8[i] as f32 / 255.0;
        let g = level.rgba8[i + 1] as f32 / 255.0;
        let b = level.rgba8[i + 2] as f32 / 255.0;
        let a = level.rgba8[i + 3] as f32 / 255.0;
        Vec4::new(r, g, b, a)
    }

    fn wrap_coord(c: f32, mode: TextureWrapMode) -> f32 {
        match mode {
            TextureWrapMode::Repeat => c - c.floor(),
            TextureWrapMode::ClampToEdge => c.clamp(0.0, 1.0),
            TextureWrapMode::MirroredRepeat => {
                let whole = c.floor();
                let frac = c - whole;
                let odd = (whole as i64) & 1 != 0;
                if odd {
                    1.0 - frac
                } else {
                    frac
                }
            }
        }
    }

    fn wrap_texel_index(i: i32, dim: u32, mode: TextureWrapMode) -> u32 {
        let dim_i = dim.max(1) as i32;
        match mode {
            TextureWrapMode::Repeat => i.rem_euclid(dim_i) as u32,
            TextureWrapMode::ClampToEdge => i.clamp(0, dim_i - 1) as u32,
            TextureWrapMode::MirroredRepeat => {
                let period = dim_i.saturating_mul(2).max(1);
                let m = i.rem_euclid(period);
                if m < dim_i {
                    m as u32
                } else {
                    (period - 1 - m) as u32
                }
            }
        }
    }

    fn sample_nearest_wrapped_level(&self, uv: Vec2, level_idx: usize) -> Vec4 {
        let Some(level) = self.mipmap(level_idx) else {
            return Vec4::ZERO;
        };
        if level.width == 0 || level.height == 0 {
            return Vec4::ZERO;
        }

        let u = Self::wrap_coord(uv.x, self.sampler.wrap_s);
        let v = Self::wrap_coord(uv.y, self.sampler.wrap_t);

        let fx = (u * level.width as f32).floor();
        let fy = (v * level.height as f32).floor();

        let x = (fx as i32).clamp(0, level.width as i32 - 1) as u32;
        let y = (fy as i32).clamp(0, level.height as i32 - 1) as u32;
        Self::texel_rgba_for_level(level, x, y)
    }

    fn sample_linear_wrapped_level(&self, uv: Vec2, level_idx: usize) -> Vec4 {
        let Some(level) = self.mipmap(level_idx) else {
            return Vec4::ZERO;
        };
        if level.width == 0 || level.height == 0 {
            return Vec4::ZERO;
        }

        let u = Self::wrap_coord(uv.x, self.sampler.wrap_s);
        let v = Self::wrap_coord(uv.y, self.sampler.wrap_t);

        let x = u * level.width as f32 - 0.5;
        let y = v * level.height as f32 - 0.5;

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let ix0 = Self::wrap_texel_index(x0, level.width, self.sampler.wrap_s);
        let iy0 = Self::wrap_texel_index(y0, level.height, self.sampler.wrap_t);
        let ix1 = Self::wrap_texel_index(x1, level.width, self.sampler.wrap_s);
        let iy1 = Self::wrap_texel_index(y1, level.height, self.sampler.wrap_t);

        let c00 = Self::texel_rgba_for_level(level, ix0, iy0);
        let c10 = Self::texel_rgba_for_level(level, ix1, iy0);
        let c01 = Self::texel_rgba_for_level(level, ix0, iy1);
        let c11 = Self::texel_rgba_for_level(level, ix1, iy1);

        let cx0 = c00.lerp(c10, tx);
        let cx1 = c01.lerp(c11, tx);
        cx0.lerp(cx1, ty)
    }

    fn sample_level(&self, uv: Vec2, level_idx: usize, filter: LevelFilter) -> Vec4 {
        match filter {
            LevelFilter::Nearest => self.sample_nearest_wrapped_level(uv, level_idx),
            LevelFilter::Linear => self.sample_linear_wrapped_level(uv, level_idx),
        }
    }

    fn lod_from_footprint(&self, fp: TextureSampleFootprint) -> f32 {
        let tex_scale = Vec2::new(self.width.max(1) as f32, self.height.max(1) as f32);
        let gx = (fp.ddx_uv * tex_scale).length();
        let gy = (fp.ddy_uv * tex_scale).length();
        let rho = gx.max(gy).max(1e-8);
        rho.log2()
    }

    fn sample_mipmap_nearest_level(&self, uv: Vec2, lod: f32, filter: LevelFilter) -> Vec4 {
        let max_level = self.mipmap_count().saturating_sub(1);
        let lod_clamped = lod.clamp(0.0, max_level as f32);
        let level = (lod_clamped + 0.5).floor() as usize;
        self.sample_level(uv, level.min(max_level), filter)
    }

    fn sample_mipmap_linear_between_levels(&self, uv: Vec2, lod: f32, filter: LevelFilter) -> Vec4 {
        let max_level = self.mipmap_count().saturating_sub(1);
        let lod_clamped = lod.clamp(0.0, max_level as f32);
        let level0 = lod_clamped.floor() as usize;
        let level1 = (level0 + 1).min(max_level);
        if level0 == level1 {
            return self.sample_level(uv, level0, filter);
        }
        let t = lod_clamped - level0 as f32;
        self.sample_level(uv, level0, filter)
            .lerp(self.sample_level(uv, level1, filter), t)
    }

    fn sample_min_filter(&self, uv: Vec2, lod: f32) -> Vec4 {
        match self.sampler.min_filter {
            TextureMinFilter::Nearest => self.sample_level(uv, 0, LevelFilter::Nearest),
            TextureMinFilter::Linear => self.sample_level(uv, 0, LevelFilter::Linear),
            TextureMinFilter::NearestMipmapNearest => {
                self.sample_mipmap_nearest_level(uv, lod, LevelFilter::Nearest)
            }
            TextureMinFilter::LinearMipmapNearest => {
                self.sample_mipmap_nearest_level(uv, lod, LevelFilter::Linear)
            }
            TextureMinFilter::NearestMipmapLinear => {
                self.sample_mipmap_linear_between_levels(uv, lod, LevelFilter::Nearest)
            }
            TextureMinFilter::LinearMipmapLinear => {
                self.sample_mipmap_linear_between_levels(uv, lod, LevelFilter::Linear)
            }
        }
    }

    fn sample_mag_filter(&self, uv: Vec2) -> Vec4 {
        match self.sampler.mag_filter {
            TextureMagFilter::Nearest => self.sample_level(uv, 0, LevelFilter::Nearest),
            TextureMagFilter::Linear => self.sample_level(uv, 0, LevelFilter::Linear),
        }
    }

    fn sample_internal(&self, uv: Vec2, footprint: Option<TextureSampleFootprint>) -> Vec4 {
        match footprint {
            Some(fp) => {
                let lod = self.lod_from_footprint(fp);
                if lod > 0.0 {
                    self.sample_min_filter(uv, lod)
                } else {
                    self.sample_mag_filter(uv)
                }
            }
            None => self.sample_mag_filter(uv),
        }
    }

    fn build_mipmaps(width: u32, height: u32, base_rgba8: &[u8]) -> Vec<MipLevel> {
        let mut out = Vec::new();
        if width == 0 || height == 0 {
            return out;
        }
        out.push(MipLevel {
            width,
            height,
            rgba8: base_rgba8.to_vec(),
        });

        loop {
            let prev = out.last().expect("mip chain is non-empty");
            if prev.width == 1 && prev.height == 1 {
                break;
            }

            let next_w = (prev.width / 2).max(1);
            let next_h = (prev.height / 2).max(1);
            let mut next_rgba = vec![0u8; (next_w as usize) * (next_h as usize) * 4];

            for y in 0..next_h {
                for x in 0..next_w {
                    let sx = x.saturating_mul(2);
                    let sy = y.saturating_mul(2);
                    let x0 = sx.min(prev.width - 1);
                    let x1 = (sx + 1).min(prev.width - 1);
                    let y0 = sy.min(prev.height - 1);
                    let y1 = (sy + 1).min(prev.height - 1);
                    let taps = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
                    let mut acc = [0u32; 4];
                    for (tx, ty) in taps {
                        let i = (((ty * prev.width) + tx) as usize) * 4;
                        acc[0] += u32::from(prev.rgba8[i]);
                        acc[1] += u32::from(prev.rgba8[i + 1]);
                        acc[2] += u32::from(prev.rgba8[i + 2]);
                        acc[3] += u32::from(prev.rgba8[i + 3]);
                    }
                    let o = (((y * next_w) + x) as usize) * 4;
                    next_rgba[o] = ((acc[0] + 2) / 4) as u8;
                    next_rgba[o + 1] = ((acc[1] + 2) / 4) as u8;
                    next_rgba[o + 2] = ((acc[2] + 2) / 4) as u8;
                    next_rgba[o + 3] = ((acc[3] + 2) / 4) as u8;
                }
            }

            out.push(MipLevel {
                width: next_w,
                height: next_h,
                rgba8: next_rgba,
            });
        }

        out
    }

    fn mipmap(&self, level_idx: usize) -> Option<&MipLevel> {
        self.mipmaps.get(level_idx).or_else(|| self.mipmaps.first())
    }

    fn mipmap_count(&self) -> usize {
        self.mipmaps.len().max(1)
    }

    pub fn rebuild_mipmaps(&mut self) {
        self.mipmaps = Self::build_mipmaps(self.width, self.height, &self.rgba8);
    }

    pub fn sample_nearest(&self, uv: Vec2) -> Vec3 {
        self.sample_level(uv, 0, LevelFilter::Nearest).truncate()
    }

    pub fn sample(&self, uv: Vec2) -> Vec3 {
        self.sample_internal(uv, None).truncate()
    }

    pub fn sample_with_footprint(&self, uv: Vec2, footprint: TextureSampleFootprint) -> Vec3 {
        self.sample_internal(uv, Some(footprint)).truncate()
    }

    pub fn sample_rgba(&self, uv: Vec2) -> Vec4 {
        self.sample_internal(uv, None)
    }

    pub fn sample_rgba_with_footprint(&self, uv: Vec2, footprint: TextureSampleFootprint) -> Vec4 {
        self.sample_internal(uv, Some(footprint))
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

        Self::from_rgba8(width, height, rgba8).unwrap_or(Self {
            width,
            height,
            rgba8: Vec::new(),
            sampler: TextureSampler::default(),
            mipmaps: Vec::new(),
        })
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
            min_filter: TextureMinFilter::Nearest,
            mag_filter: TextureMagFilter::Nearest,
        });
        let c = t.sample(Vec2::new(1.2, 0.0));
        assert!(c.y > 0.9 && c.x < 0.1);

        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::MirroredRepeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureMinFilter::Nearest,
            mag_filter: TextureMagFilter::Nearest,
        });
        let c = t.sample(Vec2::new(-0.2, 0.0));
        assert!(c.x > 0.9 && c.y < 0.1);
    }

    #[test]
    fn linear_sampling_blends_adjacent_texels() {
        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureMinFilter::Linear,
            mag_filter: TextureMagFilter::Linear,
        });
        let c = t.sample(Vec2::new(0.99, 0.0));
        assert!(c.x > 0.2 && c.y > 0.2);
    }

    #[test]
    fn min_and_mag_filters_are_selected_independently() {
        let t = tex2().with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureMinFilter::LinearMipmapLinear,
            mag_filter: TextureMagFilter::Nearest,
        });
        let c = t.sample(Vec2::new(0.01, 0.0));
        assert!(c.x > 0.95 && c.y < 0.05);
    }

    #[test]
    fn mipmapped_minification_uses_coarser_levels() {
        let t = Texture::from_rgba8(
            4,
            1,
            vec![
                255, 0, 0, 255, //
                0, 255, 0, 255, //
                255, 0, 0, 255, //
                0, 255, 0, 255, //
            ],
        )
        .unwrap()
        .with_sampler(TextureSampler {
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
            min_filter: TextureMinFilter::LinearMipmapLinear,
            mag_filter: TextureMagFilter::Nearest,
        });
        let c = t.sample_with_footprint(
            Vec2::new(0.37, 0.0),
            TextureSampleFootprint {
                ddx_uv: Vec2::new(1.0, 0.0),
                ddy_uv: Vec2::new(0.0, 1.0),
            },
        );
        assert!((c.x - 0.5).abs() < 0.12);
        assert!((c.y - 0.5).abs() < 0.12);
    }
}
