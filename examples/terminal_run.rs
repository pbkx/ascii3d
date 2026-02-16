#[cfg(feature = "terminal")]
use ascii3d::{
    prelude::*,
    targets::{
        buffer::BufferTarget,
        terminal::{ColorMode, TerminalGuard, TerminalPresenter, TerminalPresenterConfig},
    },
};

#[cfg(feature = "terminal")]
use glam::{Quat, Vec3};

#[cfg(feature = "terminal")]
use std::{io, thread, time::Duration};

#[cfg(not(feature = "terminal"))]
fn main() {}

#[cfg(feature = "terminal")]
fn main() {
    let (view, ramp, color_mode) = parse_cli();
    let _guard = TerminalGuard::new().ok();

    let mut scene = Scene::new();
    scene.add_light(Light::directional(Vec3::new(0.3, 0.7, 1.0), Vec3::ONE, 1.0));
    scene.add_object(
        Mesh::unit_triangle(),
        Transform {
            translation: Vec3::new(0.0, 0.0, 2.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        Material::new(Vec3::ONE),
    );

    let renderer = Renderer::new(
        RendererConfig::new(80, 40)
            .with_debug_view(view)
            .with_ascii_ramp(ramp),
    );
    let mut target = BufferTarget::new(80, 40);
    renderer.render(&scene, &mut target);

    let mut presenter =
        TerminalPresenter::with_config(80, 40, TerminalPresenterConfig { color_mode });
    let _ = presenter.present(&mut io::stdout(), &target);
    thread::sleep(Duration::from_millis(1500));
}

#[cfg(feature = "terminal")]
fn parse_cli() -> (DebugView, AsciiRamp, ColorMode) {
    let mut view = DebugView::default();
    let mut ramp = AsciiRamp::basic();
    let mut gamma = 1.0f32;
    let mut color_mode = ColorMode::Auto;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        let a = args[i].as_str();
        if let Some(rest) = a.strip_prefix("--view=") {
            if let Some(v) = DebugView::parse(rest) {
                view = v;
            }
        } else if a == "--view" && i + 1 < args.len() {
            if let Some(v) = DebugView::parse(&args[i + 1]) {
                view = v;
            }
            i += 1;
        } else if let Some(rest) = a.strip_prefix("--ramp=") {
            ramp = AsciiRamp::from_arg(rest);
        } else if a == "--ramp" && i + 1 < args.len() {
            ramp = AsciiRamp::from_arg(&args[i + 1]);
            i += 1;
        } else if let Some(rest) = a.strip_prefix("--gamma=") {
            if let Ok(g) = rest.parse::<f32>() {
                gamma = g;
            }
        } else if a == "--gamma" && i + 1 < args.len() {
            if let Ok(g) = args[i + 1].parse::<f32>() {
                gamma = g;
            }
            i += 1;
        } else if let Some(rest) = a.strip_prefix("--color=") {
            if let Some(m) = parse_color_mode(rest) {
                color_mode = m;
            }
        } else if a == "--color" && i + 1 < args.len() {
            if let Some(m) = parse_color_mode(&args[i + 1]) {
                color_mode = m;
            }
            i += 1;
        }
        i += 1;
    }

    (view, ramp.with_gamma(gamma), color_mode)
}

#[cfg(feature = "terminal")]
fn parse_color_mode(s: &str) -> Option<ColorMode> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Some(ColorMode::Auto),
        "truecolor" => Some(ColorMode::Truecolor),
        "ansi256" => Some(ColorMode::Ansi256),
        "mono" => Some(ColorMode::Mono),
        _ => None,
    }
}
