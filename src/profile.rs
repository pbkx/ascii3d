use crate::targets::{BufferTarget, Cell};
use crate::types::Rgb8;
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    pub width: usize,
    pub height: usize,
    pub tile_binning: bool,
    pub triangles_submitted: u64,

    pub total: Duration,
    pub target_resize: Duration,
    pub target_clear: Duration,
    pub gbuffer_alloc: Duration,
    pub raster: Duration,
    pub map: Duration,
}

impl RenderStats {
    pub fn overlay_text(&self) -> String {
        fn ms(d: Duration) -> f32 {
            d.as_secs_f32() * 1000.0
        }

        format!(
            "raster={}x{} tri={} tiled={} total={:.2}ms resize={:.2}ms clear={:.2}ms gbuf={:.2}ms raster={:.2}ms map={:.2}ms",
            self.width,
            self.height,
            self.triangles_submitted,
            if self.tile_binning { 1 } else { 0 },
            ms(self.total),
            ms(self.target_resize),
            ms(self.target_clear),
            ms(self.gbuffer_alloc),
            ms(self.raster),
            ms(self.map),
        )
    }

    pub fn write_overlay(&self, target: &mut BufferTarget, x: usize, y: usize) {
        let fg = Rgb8::new(255, 255, 255);
        let bg = Rgb8::new(0, 0, 0);
        for (i, ch) in self.overlay_text().chars().enumerate() {
            let _ = target.set(x + i, y, Cell::new(ch, fg, bg, 0.0));
        }
    }
}
