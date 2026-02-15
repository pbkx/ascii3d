#![forbid(unsafe_code)]
#![warn(clippy::all)]
#![allow(
    clippy::pedantic,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::explicit_iter_loop,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::match_same_arms,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_errors_doc,
    clippy::missing_fields_in_debug,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps,
    clippy::unreadable_literal,
    clippy::verbose_bit_mask,
    clippy::wrap_ok,
)]
pub mod bvh;
pub mod camera;
pub mod debug;
pub mod dither;
pub mod framegraph;
pub mod glyph;
pub mod gbuffer;
pub mod io;
pub mod light;
pub mod material;
pub mod texture;
pub mod mesh;
pub mod profile;
pub mod prelude;
pub mod raster;
pub mod renderer;
pub mod scene;
pub mod shader;
pub mod targets;
pub mod temporal;
pub mod transform;
pub mod types;

pub use crate::{
    bvh::{Aabb, MeshBvh, MeshHit, Ray, SceneBvh, SceneHit},
    camera::{Camera, Projection},
    debug::DebugView,
    dither::DitherMode,
    framegraph::{FrameGraph, FramePass, FramePassId, PostProcessSettings},
    gbuffer::GBuffer,
    glyph::{AsciiRamp, GlyphMode},
    light::Light,
    material::Material,
    mesh::Mesh,
    profile::RenderStats,
    texture::{Texture, TextureHandle},
    renderer::{Renderer, RendererConfig},
    scene::Scene,
    shader::{BuiltinShader, LambertShader, ShadeSample, Shader, ShaderId, UnlitShader},
    temporal::TemporalConfig,
    transform::Transform,
    types::Rgb8,
};
