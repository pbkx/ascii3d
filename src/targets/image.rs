pub struct ImageTarget {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

impl ImageTarget {
    pub fn new(width: usize, height: usize) -> Self {
        let mut out = Self {
            width,
            height,
            rgba: vec![0u8; width.saturating_mul(height).saturating_mul(4)],
        };
        out.clear_rgba(0, 0, 0, 255);
        out
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear_rgba(&mut self, r: u8, g: u8, b: u8, a: u8) {
        for px in self.rgba.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
    }

    pub fn set_rgba(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8, a: u8) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let idx = (y * self.width + x) * 4;
        self.rgba[idx] = r;
        self.rgba[idx + 1] = g;
        self.rgba[idx + 2] = b;
        self.rgba[idx + 3] = a;
        true
    }

    pub fn get_rgba(&self, x: usize, y: usize) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 4;
        Some([
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ])
    }

    pub fn as_rgba_slice(&self) -> &[u8] {
        &self.rgba
    }

    pub fn hash64(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        fn mix(h: &mut u64, b: u8) {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
        for b in self.width.to_le_bytes() {
            mix(&mut h, b);
        }
        for b in self.height.to_le_bytes() {
            mix(&mut h, b);
        }
        for &b in &self.rgba {
            mix(&mut h, b);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::ImageTarget;

    #[test]
    fn image_hash_is_deterministic() {
        let mut img = ImageTarget::new(4, 3);
        img.set_rgba(0, 0, 1, 2, 3, 4);
        img.set_rgba(3, 2, 9, 8, 7, 6);
        let h1 = img.hash64();
        let h2 = img.hash64();
        assert_eq!(h1, h2);
    }
}
