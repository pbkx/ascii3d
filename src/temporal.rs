use crate::dither::Dither;

#[derive(Clone, Debug)]
pub struct TemporalConfig {
    pub enabled: bool,
    pub ema_alpha: f32,
    pub hysteresis: f32,
    pub anchored_dither: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ema_alpha: 0.35,
            hysteresis: 0.02,
            anchored_dither: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalState {
    width: usize,
    height: usize,
    luma_ema: Vec<f32>,
    glyph_idx: Vec<u16>,
}

impl Default for TemporalState {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalState {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            luma_ema: Vec::new(),
            glyph_idx: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.width = 0;
        self.height = 0;
        self.luma_ema.clear();
        self.glyph_idx.clear();
    }

    pub fn ensure_size(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let n = width.saturating_mul(height);
        self.luma_ema = vec![0.0; n];
        self.glyph_idx = vec![u16::MAX; n];
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        self.ensure_size(width, height);
    }

    pub(crate) fn step_index(
        &mut self,
        x: usize,
        y: usize,
        luma: f32,
        ramp_len: usize,
        quantizer: &mut TemporalQuantizer<'_>,
        cfg: &TemporalConfig,
    ) -> usize {
        self.step(x, y, luma, ramp_len, quantizer, cfg) as usize
    }


    pub(crate) fn step(
        &mut self,
        x: usize,
        y: usize,
        raw_luma: f32,
        ramp_len: usize,
        quantizer: &mut TemporalQuantizer<'_>,
        cfg: &TemporalConfig,
    ) -> u16 {
        let width = self.width;
        let idx = y * width + x;
        let raw_luma = raw_luma.clamp(0.0, 1.0);
        let prev_idx = self.glyph_idx[idx];
        let prev_ema = self.luma_ema[idx];

        let ema = if prev_idx == u16::MAX {
            raw_luma
        } else {
            let a = cfg.ema_alpha.clamp(0.0, 1.0);
            a * raw_luma + (1.0 - a) * prev_ema
        };

        let cand_idx = quantizer.quantize(x, y, ema, ramp_len);

        let out_idx = if prev_idx != u16::MAX && prev_idx != cand_idx {
            let diff = (ema - prev_ema).abs();
            let delta = if prev_idx > cand_idx {
                prev_idx - cand_idx
            } else {
                cand_idx - prev_idx
            };
            if delta <= 1 && diff < cfg.hysteresis.max(0.0) {
                prev_idx
            } else {
                cand_idx
            }
        } else {
            cand_idx
        };

        self.luma_ema[idx] = ema;
        self.glyph_idx[idx] = out_idx;
        out_idx
    }
}

pub(crate) enum TemporalQuantizer<'a> {
    Dither(&'a mut Dither),
    Anchored,
}

impl<'a> TemporalQuantizer<'a> {
    pub(crate) fn quantize(&mut self, x: usize, y: usize, luma: f32, ramp_len: usize) -> u16 {
        match self {
            Self::Dither(d) => d.quantize_index(luma, ramp_len, x, y) as u16,
            Self::Anchored => anchored_quantize(x, y, luma, ramp_len),
        }
    }
}

fn anchored_quantize(x: usize, y: usize, luma: f32, ramp_len: usize) -> u16 {
    if ramp_len <= 1 {
        return 0;
    }
    let max = (ramp_len - 1) as f32;
    let s = (luma.clamp(0.0, 1.0)) * max;
    let lo = s.floor() as i32;
    if lo as usize >= ramp_len - 1 {
        return (ramp_len - 1) as u16;
    }
    let frac = s - (lo as f32);
    let u = hash01(x as u32, y as u32);
    if frac > u {
        (lo + 1) as u16
    } else {
        lo as u16
    }
}

fn hash01(x: u32, y: u32) -> f32 {
    let mut v = x
        .wrapping_mul(0x9e3779b9)
        ^ y.wrapping_mul(0x85ebca6b)
        ^ 0x2c1b3c6d;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846ca68b);
    v ^= v >> 16;
    // Use the top 24 bits to get a stable [0,1) float without platform-specific quirks.
    let top24 = v >> 8;
    (top24 as f32) * (1.0 / 16_777_216.0)
}
