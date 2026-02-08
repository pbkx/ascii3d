use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DitherMode {
    None,
    Ordered,
    ErrorDiffusion,
    BlueNoise,
}

impl fmt::Display for DitherMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DitherMode::None => "none",
            DitherMode::Ordered => "ordered",
            DitherMode::ErrorDiffusion => "error_diffusion",
            DitherMode::BlueNoise => "blue_noise",
        };
        f.write_str(s)
    }
}

impl FromStr for DitherMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.trim().to_ascii_lowercase();
        match v.as_str() {
            "none" | "off" => Ok(DitherMode::None),
            "ordered" | "bayer" => Ok(DitherMode::Ordered),
            "error_diffusion" | "errordiffusion" | "floyd" | "fs" => Ok(DitherMode::ErrorDiffusion),
            "blue_noise" | "bluenoise" | "blue" => Ok(DitherMode::BlueNoise),
            _ => Err(format!("invalid dither mode: {s}")),
        }
    }
}

pub struct Dither {
    mode: DitherMode,
    width: usize,
    err_cur: Vec<f32>,
    err_next: Vec<f32>,
}

impl Dither {
    pub fn new(mode: DitherMode, width: usize) -> Self {
        let (err_cur, err_next) = if mode == DitherMode::ErrorDiffusion {
            (vec![0.0; width], vec![0.0; width])
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            mode,
            width,
            err_cur,
            err_next,
        }
    }

    pub fn mode(&self) -> DitherMode {
        self.mode
    }

    pub fn finish_row(&mut self) {
        if self.mode != DitherMode::ErrorDiffusion {
            return;
        }
        self.err_cur.clone_from(&self.err_next);
        for v in &mut self.err_next {
            *v = 0.0;
        }
    }

    pub fn quantize_index(&mut self, scalar: f32, levels: usize, x: usize, y: usize) -> usize {
        if levels <= 1 {
            return 0;
        }

        let t = scalar.clamp(0.0, 1.0);
        let max = (levels - 1) as f32;
        match self.mode {
            DitherMode::None => {
                let v = t * max + 0.5;
                (v.floor() as usize).min(levels - 1)
            }
            DitherMode::Ordered => {
                let v = t * max;
                let base = v.floor().clamp(0.0, max) as usize;
                let frac = v - base as f32;
                let thr = bayer8_threshold(x, y);
                if base + 1 < levels && frac > thr {
                    base + 1
                } else {
                    base
                }
            }
            DitherMode::BlueNoise => {
                let v = t * max;
                let base = v.floor().clamp(0.0, max) as usize;
                let frac = v - base as f32;
                let thr = ign_threshold(x, y);
                if base + 1 < levels && frac > thr {
                    base + 1
                } else {
                    base
                }
            }
            DitherMode::ErrorDiffusion => {
                let xi = x.min(self.width.saturating_sub(1));
                let v = t * max + self.err_cur.get(xi).copied().unwrap_or(0.0);
                let mut idx = (v + 0.5).floor() as isize;
                let lo = 0isize;
                let hi = (levels - 1) as isize;
                if idx < lo {
                    idx = lo;
                }
                if idx > hi {
                    idx = hi;
                }
                let q = idx as f32;
                let err = v - q;

                if xi < self.err_cur.len() {
                    self.err_cur[xi] = 0.0;
                }

                if xi + 1 < self.err_cur.len() {
                    self.err_cur[xi + 1] += err * (7.0 / 16.0);
                }
                if xi > 0 && xi - 1 < self.err_next.len() {
                    self.err_next[xi - 1] += err * (3.0 / 16.0);
                }
                if xi < self.err_next.len() {
                    self.err_next[xi] += err * (5.0 / 16.0);
                }
                if xi + 1 < self.err_next.len() {
                    self.err_next[xi + 1] += err * (1.0 / 16.0);
                }

                idx as usize
            }
        }
    }
}

fn bayer8_threshold(x: usize, y: usize) -> f32 {
    const BAYER8: [[u8; 8]; 8] = [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ];
    let v = BAYER8[y & 7][x & 7] as f32;
    (v + 0.5) / 64.0
}

fn ign_threshold(x: usize, y: usize) -> f32 {
    let fx = x as f32;
    let fy = y as f32;
    let a = (fx * 0.06711056 + fy * 0.00583715).fract();
    (52.9829189 * a).fract()
}
