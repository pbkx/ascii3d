use crate::{targets::Cell, types::Rgb8};

#[derive(Clone, Debug)]
pub enum GlyphMode {
    AsciiRamp(AsciiRamp),
    HalfBlock,
}

impl Default for GlyphMode {
    fn default() -> Self {
        GlyphMode::AsciiRamp(AsciiRamp::basic())
    }
}

impl GlyphMode {
    pub fn cell_from_scalar(&self, t: f32, depth: f32) -> Cell {
        match self {
            GlyphMode::AsciiRamp(r) => Cell::new(r.map_scalar_to_char(t), Rgb8::BLACK, Rgb8::BLACK, depth),
            GlyphMode::HalfBlock => Cell::new(' ', Rgb8::BLACK, Rgb8::BLACK, depth),
        }
    }

    pub fn with_ascii_ramp(self, ramp: AsciiRamp) -> Self {
        GlyphMode::AsciiRamp(ramp)
    }

    pub fn ascii_ramp(&self) -> Option<&AsciiRamp> {
        match self {
            GlyphMode::AsciiRamp(r) => Some(r),
            GlyphMode::HalfBlock => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AsciiRamp {
    bytes: Vec<u8>,
    gamma: f32,
}

impl AsciiRamp {
    pub fn new(chars: &str) -> Self {
        AsciiRamp {
            bytes: chars.as_bytes().to_vec(),
            gamma: 1.0,
        }
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = sanitize_gamma(gamma);
        self
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn map_scalar_to_char(&self, t: f32) -> char {
        self.map_scalar_to_byte(t) as char
    }

    pub fn map_scalar_to_byte(&self, t: f32) -> u8 {
        if self.bytes.is_empty() {
            return b' ';
        }
        let mut tt = t.clamp(0.0, 1.0);
        let g = self.gamma;
        if g != 1.0 {
            tt = tt.powf(g);
        }
        let last = self.bytes.len().saturating_sub(1);
        let i = ((tt * (last as f32)) + 0.5) as usize;
        self.bytes[i.min(last)]
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "smooth" => Some(AsciiRamp::smooth()),
            "basic" => Some(AsciiRamp::basic()),
            "alt" => Some(AsciiRamp::alt()),
            "classic" => Some(AsciiRamp::classic()),
            "tiny" => Some(AsciiRamp::tiny()),
            _ => None,
        }
    }

    pub fn from_arg(arg: &str) -> Self {
        AsciiRamp::from_name(arg).unwrap_or_else(|| AsciiRamp::new(arg))
    }

    pub fn smooth() -> Self {
        AsciiRamp::new(r#" .`^",:;Il!i><~+_-?][}{1)(|\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"#)
    }

    pub fn basic() -> Self {
        AsciiRamp::new(" .:-=+*#%@")
    }

    pub fn alt() -> Self {
        AsciiRamp::new(" .,:;i1tfLCG08@")
    }

    pub fn classic() -> Self {
        AsciiRamp::new(r#" .'`^",:;Il!i><~+_-?][}{1)(|\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"#)
    }

    pub fn tiny() -> Self {
        AsciiRamp::new(" .:-=+*#%@")
    }
}

fn sanitize_gamma(gamma: f32) -> f32 {
    if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        1.0
    }
}
