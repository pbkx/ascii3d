use crate::types::Rgb8;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub fg: Rgb8,
    pub bg: Rgb8,
    pub depth_bits: u32,
}

impl Cell {
    pub fn new(ch: char, fg: Rgb8, bg: Rgb8, depth: f32) -> Self {
        Self {
            ch,
            fg,
            bg,
            depth_bits: depth.to_bits(),
        }
    }

    pub fn depth(self) -> f32 {
        f32::from_bits(self.depth_bits)
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: Rgb8::BLACK,
            bg: Rgb8::BLACK,
            depth_bits: f32::INFINITY.to_bits(),
        }
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cell")
            .field("ch", &self.ch)
            .field("fg", &self.fg)
            .field("bg", &self.bg)
            .field("depth", &self.depth())
            .finish()
    }
}

#[derive(Clone)]
pub struct BufferTarget {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl BufferTarget {
    pub fn new(width: usize, height: usize) -> Self {
        let len = width.saturating_mul(height);
        Self {
            width,
            height,
            cells: vec![Cell::default(); len],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_slice(&self) -> &[Cell] {
        &self.cells
    }

    pub fn as_mut_slice(&mut self) -> &mut [Cell] {
        &mut self.cells
    }

    pub fn clear(&mut self, cell: Cell) {
        self.cells.fill(cell);
    }

    pub fn get(&self, x: usize, y: usize) -> Option<Cell> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.cells[self.idx(x, y)])
    }

    pub fn cell(&self, x: usize, y: usize) -> Option<Cell> {
        self.get(x, y)
    }

    pub fn set(&mut self, x: usize, y: usize, cell: Cell) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let idx = self.idx(x, y);
        self.cells[idx] = cell;
        true
    }

    pub fn hash64(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        h = fnv1a_u64(h, &(self.width as u64).to_le_bytes());
        h = fnv1a_u64(h, &(self.height as u64).to_le_bytes());

        for c in &self.cells {
            let ch = c.ch as u32;
            h = fnv1a_u64(h, &ch.to_le_bytes());
            h = fnv1a_u64(h, &c.fg.to_le_bytes());
            h = fnv1a_u64(h, &c.bg.to_le_bytes());
            h = fnv1a_u64(h, &c.depth_bits.to_le_bytes());
        }

        h
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

fn fnv1a_u64(mut h: u64, bytes: &[u8]) -> u64 {
    let prime = 0x0100_0000_01b3_u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(prime);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{BufferTarget, Cell};
    use crate::types::Rgb8;

    #[test]
    fn buffer_hash_is_deterministic() {
        let mut b = BufferTarget::new(4, 3);
        b.set(
            0,
            0,
            Cell::new('A', Rgb8::new(1, 2, 3), Rgb8::new(4, 5, 6), 0.5),
        );
        b.set(
            3,
            2,
            Cell::new('Z', Rgb8::new(255, 0, 17), Rgb8::BLACK, 42.0),
        );
        b.set(
            2,
            1,
            Cell::new('█', Rgb8::new(7, 3, 9), Rgb8::new(1, 0, 0), 1.25),
        );

        let h = b.hash64();
        assert_eq!(h, 11_525_354_658_919_025_284_u64);
    }

    #[test]
    fn out_of_bounds_set_returns_false() {
        let mut b = BufferTarget::new(2, 2);
        assert!(!b.set(2, 0, Cell::new('X', Rgb8::BLACK, Rgb8::BLACK, 0.0)));
        assert!(!b.set(0, 2, Cell::new('X', Rgb8::BLACK, Rgb8::BLACK, 0.0)));
    }
}
