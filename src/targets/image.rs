pub struct ImageTarget {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

#[cfg(feature = "png")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PngWriteError {
    DimensionTooLarge,
}

#[cfg(feature = "png")]
fn append_chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let mut crc = 0xFFFF_FFFFu32;
    for &b in &kind {
        crc = crc32_update(crc, b);
    }
    for &b in data {
        crc = crc32_update(crc, b);
    }
    crc = !crc;
    out.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(feature = "png")]
fn crc32_update(mut crc: u32, b: u8) -> u32 {
    crc ^= b as u32;
    for _ in 0..8 {
        let mask = (crc & 1).wrapping_neg();
        crc = (crc >> 1) ^ (0xEDB8_8320u32 & mask);
    }
    crc
}

#[cfg(feature = "png")]
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + b as u32) % MOD;
        s2 = (s2 + s1) % MOD;
    }
    (s2 << 16) | s1
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

    #[cfg(feature = "png")]
    pub fn write_png_to_vec(&self) -> Result<Vec<u8>, PngWriteError> {
        let width = u32::try_from(self.width).map_err(|_| PngWriteError::DimensionTooLarge)?;
        let height = u32::try_from(self.height).map_err(|_| PngWriteError::DimensionTooLarge)?;

        let mut raw = Vec::with_capacity(self.height.saturating_mul(1 + 4 * self.width));
        for y in 0..self.height {
            raw.push(0u8);
            let row = &self.rgba[(y * self.width * 4)..((y + 1) * self.width * 4)];
            raw.extend_from_slice(row);
        }

        let mut zlib = Vec::new();
        zlib.push(0x78);
        zlib.push(0x01);

        let mut i = 0;
        while i < raw.len() {
            let remaining = raw.len() - i;
            let chunk_len = remaining.min(65535);
            let bfinal = if i + chunk_len == raw.len() { 1u8 } else { 0u8 };
            zlib.push(bfinal);
            let len = chunk_len as u16;
            zlib.extend_from_slice(&len.to_le_bytes());
            zlib.extend_from_slice(&(!len).to_le_bytes());
            zlib.extend_from_slice(&raw[i..(i + chunk_len)]);
            i += chunk_len;
        }

        let ad = adler32(&raw);
        zlib.extend_from_slice(&ad.to_be_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(8u8);
        ihdr.push(6u8);
        ihdr.push(0u8);
        ihdr.push(0u8);
        ihdr.push(0u8);

        append_chunk(&mut out, *b"IHDR", &ihdr);
        append_chunk(&mut out, *b"IDAT", &zlib);
        append_chunk(&mut out, *b"IEND", &[]);

        Ok(out)
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
