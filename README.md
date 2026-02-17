# ascii3d

[![crates.io](https://img.shields.io/badge/crates.io-not%20published%20yet-orange)](#)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

library api docs are generated via rustdoc from this crate

terminal output backend is integrated in this crate behind feature `terminal`

gltf loading, trs animation, and skinning are integrated behind feature `gltf`

default behavior is deterministic for testing via output hashes

# install instructions

install the `ascii3d` crate by adding it as a dependency in `Cargo.toml`.

### linux

use path dependency in your Cargo.toml
```toml
[dependencies]
ascii3d = { path = "../ascii3d" }
```

### windows

use path dependency in your Cargo.toml
```toml
[dependencies]
ascii3d = { path = "../ascii3d" }
```

for terminal rendering, use a modern terminal like windows terminal or alacritty

### macos

install developer tools via ```xcode-select --install```

use path dependency in your Cargo.toml
```toml
[dependencies]
ascii3d = { path = "../ascii3d" }
```

# build instructions

if build fails due to missing rust toolchain pieces, install/update rustup and set stable as default toolchain.

### linux

dependencies are: rust stable toolchain (`cargo`, `rustc`)

```
git clone <this-repo-url>
cd ascii3d
cargo build
cargo test --all-features
```

### windows

have cargo and stable toolchain installed

```
git clone <this-repo-url>
cd ascii3d
cargo build
cargo test --all-features
```

### macos

install developer tools via ```xcode-select --install```

```
git clone <this-repo-url>
cd ascii3d
cargo build
cargo test --all-features
```

# usage

```
Usage: ascii3d (library crate)
Core public types:
- Renderer
- RendererConfig
- Scene
- Camera
- Mesh
- Material
- Light
- Transform

Core targets:
- BufferTarget (cell buffer)
- ImageTarget (rgba buffer)
- TerminalPresenter (feature="terminal")

Core io:
- io::load_obj_with_mtl(...)
- io::load_stl(...)
- io::load_gltf(...) (feature="gltf")
- io::load_gltf_at_time(...) (feature="gltf")
- io::load_gltf_str(...) (feature="gltf")
- io::load_gltf_str_at_time(...) (feature="gltf")

RendererConfig controls:
- with_size(width,height)
- with_shader_id(...)
- with_debug_view(Final|Depth|Normals|Albedo)
- with_glyph_mode(AsciiRamp|HalfBlock)
- with_dither_mode(None|Ordered|ErrorDiffusion|BlueNoise)
- with_temporal_enabled(...)
- with_temporal_config(...)
- with_tile_binning(...)
- with_tone_map(...)
- with_contrast(...)
- with_edge_enhance(...)

Render entry points:
- render(scene, &mut BufferTarget)
- render_image(scene, &mut ImageTarget)
- render_with_stats(scene, &mut BufferTarget)
- render_image_with_stats(scene, &mut ImageTarget)

Determinism:
- BufferTarget::hash64() for deterministic cell snapshots
- ImageTarget::hash64() for deterministic rgba snapshots
- stable render behavior is verified by tests under src/*
```

- framegraph pass ordering is explicit and deterministic
- gltf trs animation playback is integrated (translation/rotation/scale channels)
- gltf skinning is integrated (JOINTS_0, WEIGHTS_0, inverse bind matrices)
- material workflow includes albedo/base-color mapping support
- terminal modes support mono/ansi256/truecolor paths
- png writing is available on `ImageTarget` with feature `png`
- rayon tiling path is available with feature `rayon`

Order of Pipeline:
- scene traversal and transform composition
- optional gltf animation sampling by time
- optional gltf skinning deformation
- rasterization to gbuffer
- shading/debug mapping
- optional post process (tone map, contrast, edge enhance)
- glyph resolve and target output

Feature Flags:
- `terminal`: terminal backend/presenter
- `image`: texture decode support through image crate
- `png`: png write path on image target
- `gltf`: gltf io + animation + skinning paths
- `rayon`: parallel tiled rendering

Debug Views:
- `Final`
- `Depth`
- `Normals`
- `Albedo`

Built-in shader ids:
- `Lambert`
- `Unlit`

Glyph modes:
- `AsciiRamp`
- `HalfBlock`

# basic example usage

`examples/basic_usage.rs` demonstrates:
- creating a scene with `Mesh::unit_triangle()`
- rendering to `BufferTarget` with `Renderer::render`
- rendering to `ImageTarget` with `Renderer::render_image`
- verifying deterministic output with `hash64()`

run:

```
cargo run --example basic_usage
```

what you should see:
- one `buffer hash64: ...` line
- one `image  hash64: ...` line

the exact hash values are deterministic for a fixed codebase/build path.

### more advanced usage

`examples/advanced_usage.rs` demonstrates:
- adding directional lighting to `Scene`
- building materials with diffuse/specular terms
- rendering multiple meshes with different transforms
- enabling dither, tile binning, tone map, contrast, and edge enhance
- collecting `RenderStats` from `render_image_with_stats`

run:

```
cargo run --example advanced_usage
```

what you should see:
- one renderer timing/triangle summary line from `stats.overlay_text()`
- one deterministic image hash line
