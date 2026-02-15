use glam::Vec3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FramePassId {
    RasterizeGBuffer,
    Shade,
    ToneMap,
    Contrast,
    EdgeEnhance,
    ResolveTarget,
}

#[derive(Clone, Debug)]
pub struct FramePass {
    pub id: FramePassId,
    pub deps: Vec<FramePassId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostProcessSettings {
    pub tone_map: bool,
    pub contrast: f32,
    pub edge_enhance: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            tone_map: false,
            contrast: 1.0,
            edge_enhance: 0.0,
        }
    }
}

impl PostProcessSettings {
    pub fn has_any_post_pass(self) -> bool {
        self.tone_map || (self.contrast - 1.0).abs() > 1e-6 || self.edge_enhance.abs() > 1e-6
    }
}

#[derive(Clone, Debug)]
pub struct FrameGraph {
    passes: Vec<FramePass>,
}

impl FrameGraph {
    pub fn new(settings: PostProcessSettings) -> Self {
        let mut passes = vec![
            FramePass {
                id: FramePassId::RasterizeGBuffer,
                deps: Vec::new(),
            },
            FramePass {
                id: FramePassId::Shade,
                deps: vec![FramePassId::RasterizeGBuffer],
            },
        ];

        let mut prev = FramePassId::Shade;

        if settings.tone_map {
            passes.push(FramePass {
                id: FramePassId::ToneMap,
                deps: vec![prev],
            });
            prev = FramePassId::ToneMap;
        }

        if (settings.contrast - 1.0).abs() > 1e-6 {
            passes.push(FramePass {
                id: FramePassId::Contrast,
                deps: vec![prev],
            });
            prev = FramePassId::Contrast;
        }

        if settings.edge_enhance.abs() > 1e-6 {
            passes.push(FramePass {
                id: FramePassId::EdgeEnhance,
                deps: vec![prev],
            });
            prev = FramePassId::EdgeEnhance;
        }

        passes.push(FramePass {
            id: FramePassId::ResolveTarget,
            deps: vec![prev],
        });

        Self { passes }
    }

    pub fn passes(&self) -> &[FramePass] {
        &self.passes
    }
}

pub fn tone_map_reinhard(rgb: Vec3) -> Vec3 {
    let c = rgb.max(Vec3::ZERO);
    c / (Vec3::ONE + c)
}

pub fn apply_tone_map_reinhard(buf: &mut [Vec3]) {
    for px in buf.iter_mut() {
        *px = tone_map_reinhard(*px);
    }
}

pub fn apply_contrast(buf: &mut [Vec3], contrast: f32) {
    for px in buf.iter_mut() {
        *px = ((*px - Vec3::splat(0.5)) * contrast + Vec3::splat(0.5)).clamp(Vec3::ZERO, Vec3::ONE);
    }
}

pub fn apply_edge_enhance(buf: &mut [Vec3], width: usize, height: usize, strength: f32) {
    if width == 0 || height == 0 || strength.abs() <= 1e-6 {
        return;
    }

    let src = buf.to_vec();
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let c = src[i];

            let xn = x.saturating_sub(1);
            let xp = (x + 1).min(width - 1);
            let yn = y.saturating_sub(1);
            let yp = (y + 1).min(height - 1);

            let l = src[y * width + xn];
            let r = src[y * width + xp];
            let u = src[yn * width + x];
            let d = src[yp * width + x];

            let lap = l + r + u + d - c * 4.0;
            let out = c - lap * strength;
            buf[i] = out.clamp(Vec3::ZERO, Vec3::ONE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framegraph_builds_dependencies_in_order() {
        let fg = FrameGraph::new(PostProcessSettings {
            tone_map: true,
            contrast: 1.25,
            edge_enhance: 0.5,
        });
        let p = fg.passes();
        assert_eq!(p[0].id, FramePassId::RasterizeGBuffer);
        assert_eq!(p[1].id, FramePassId::Shade);
        assert_eq!(p[2].id, FramePassId::ToneMap);
        assert_eq!(p[3].id, FramePassId::Contrast);
        assert_eq!(p[4].id, FramePassId::EdgeEnhance);
        assert_eq!(p[5].id, FramePassId::ResolveTarget);
        assert_eq!(p[5].deps, vec![FramePassId::EdgeEnhance]);
    }
}
