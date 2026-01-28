use crate::Scene;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 40,
        }
    }
}

#[derive(Debug)]
pub struct Renderer {
    config: RendererConfig,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> RendererConfig {
        self.config
    }

    pub fn render(&mut self, _scene: &Scene) {

    }
}
