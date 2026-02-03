#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic)]
pub mod camera;
pub mod debug;
pub mod glyph;
pub mod gbuffer;
pub mod io;
pub mod light;
pub mod material;
pub mod mesh;
pub mod prelude;
pub mod raster;
pub mod renderer;
pub mod scene;
pub mod shader;
pub mod targets;
pub mod transform;
pub mod types;

pub use crate::{
    camera::{Camera, Projection},
    debug::DebugView,
    gbuffer::GBuffer,
    glyph::{AsciiRamp, GlyphMode},
    light::Light,
    material::Material,
    mesh::Mesh,
    renderer::{Renderer, RendererConfig},
    scene::Scene,
    shader::{BuiltinShader, LambertShader, ShadeSample, Shader, ShaderId, UnlitShader},
    transform::Transform,
    types::Rgb8,
};