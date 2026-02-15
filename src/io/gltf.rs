use crate::{
    io::texture::{load_texture_rgba8_from_bytes, TextureIoError},
    Material, Mesh, Scene, Transform,
};
use crate::texture::{Texture, TextureHandle};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use serde::Deserialize;
use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub enum GltfError {
    Io,
    Json,
    UnsupportedVersion,
    UnsupportedBinaryGltf,
    UnsupportedDataUri,
    DecodeBase64,
    MissingBasePath,
    InvalidIndex,
    BufferOutOfBounds,
    UnsupportedAccessor,
    MissingAttribute,
    InvalidNodeGraph,
    InvalidTransform,
    UnsupportedPrimitiveMode,
    MissingImageSource,
    TextureIo,
    ImageFeatureDisabled,
    UnsupportedAnimationPath,
    UnsupportedInterpolation,
    InvalidAnimation,
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => write!(f, "io error"),
            Self::Json => write!(f, "json parse error"),
            Self::UnsupportedVersion => write!(f, "unsupported gltf version"),
            Self::UnsupportedBinaryGltf => write!(f, "binary gltf/glb is not supported"),
            Self::UnsupportedDataUri => write!(f, "unsupported data uri"),
            Self::DecodeBase64 => write!(f, "base64 decode failed"),
            Self::MissingBasePath => write!(f, "missing base path for external buffer uri"),
            Self::InvalidIndex => write!(f, "invalid index"),
            Self::BufferOutOfBounds => write!(f, "buffer out of bounds"),
            Self::UnsupportedAccessor => write!(f, "unsupported accessor"),
            Self::MissingAttribute => write!(f, "missing required mesh attribute"),
            Self::InvalidNodeGraph => write!(f, "invalid node graph"),
            Self::InvalidTransform => write!(f, "invalid transform"),
            Self::UnsupportedPrimitiveMode => write!(f, "unsupported primitive mode"),
            Self::MissingImageSource => write!(f, "missing image source"),
            Self::TextureIo => write!(f, "texture decode failed"),
            Self::ImageFeatureDisabled => write!(f, "feature `image` is disabled"),
            Self::UnsupportedAnimationPath => write!(f, "unsupported animation path"),
            Self::UnsupportedInterpolation => write!(f, "unsupported animation interpolation"),
            Self::InvalidAnimation => write!(f, "invalid animation data"),
        }
    }
}

impl Error for GltfError {}

#[derive(Deserialize)]
struct GltfRoot {
    asset: AssetDef,
    #[serde(default)]
    buffers: Vec<BufferDef>,
    #[serde(rename = "bufferViews", default)]
    buffer_views: Vec<BufferViewDef>,
    #[serde(default)]
    accessors: Vec<AccessorDef>,
    #[serde(default)]
    meshes: Vec<MeshDef>,
    #[serde(default)]
    materials: Vec<MaterialDef>,
    #[serde(default)]
    textures: Vec<TextureDef>,
    #[serde(default)]
    images: Vec<ImageDef>,
    #[serde(default)]
    nodes: Vec<NodeDef>,
    #[serde(default)]
    scenes: Vec<SceneDef>,
    #[serde(default)]
    scene: Option<usize>,
    #[serde(default)]
    animations: Vec<AnimationDef>,
}

#[derive(Deserialize)]
struct AssetDef {
    version: String,
}

#[derive(Deserialize)]
struct BufferDef {
    #[serde(default)]
    uri: Option<String>,
    #[serde(rename = "byteLength")]
    byte_length: usize,
}

#[derive(Deserialize)]
struct BufferViewDef {
    buffer: usize,
    #[serde(rename = "byteOffset", default)]
    byte_offset: usize,
    #[serde(rename = "byteLength")]
    byte_length: usize,
    #[serde(rename = "byteStride", default)]
    byte_stride: Option<usize>,
}

#[derive(Deserialize)]
struct AccessorDef {
    #[serde(rename = "bufferView", default)]
    buffer_view: Option<usize>,
    #[serde(rename = "byteOffset", default)]
    byte_offset: usize,
    #[serde(rename = "componentType")]
    component_type: u32,
    count: usize,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    sparse: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct MeshDef {
    primitives: Vec<PrimitiveDef>,
}

#[derive(Deserialize)]
struct PrimitiveDef {
    attributes: HashMap<String, usize>,
    #[serde(default)]
    indices: Option<usize>,
    #[serde(default)]
    material: Option<usize>,
    #[serde(default)]
    mode: Option<u32>,
}

#[derive(Deserialize)]
struct MaterialDef {
    #[serde(rename = "pbrMetallicRoughness", default)]
    pbr_metallic_roughness: Option<PbrMetallicRoughnessDef>,
}

#[derive(Deserialize)]
struct PbrMetallicRoughnessDef {
    #[serde(rename = "baseColorFactor", default)]
    base_color_factor: Option<[f32; 4]>,
    #[serde(rename = "baseColorTexture", default)]
    base_color_texture: Option<TextureInfoDef>,
}

#[derive(Deserialize)]
struct TextureInfoDef {
    index: usize,
    #[serde(rename = "texCoord", default)]
    tex_coord: Option<usize>,
}

#[derive(Deserialize)]
struct TextureDef {
    #[serde(default)]
    source: Option<usize>,
}

#[derive(Deserialize)]
struct ImageDef {
    #[serde(default)]
    uri: Option<String>,
    #[serde(rename = "bufferView", default)]
    buffer_view: Option<usize>,
}

#[derive(Deserialize)]
struct NodeDef {
    #[serde(default)]
    mesh: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    #[serde(default)]
    matrix: Option<[f32; 16]>,
    #[serde(default)]
    translation: Option<[f32; 3]>,
    #[serde(default)]
    rotation: Option<[f32; 4]>,
    #[serde(default)]
    scale: Option<[f32; 3]>,
}

#[derive(Deserialize)]
struct SceneDef {
    #[serde(default)]
    nodes: Vec<usize>,
}

#[derive(Deserialize)]
struct AnimationDef {
    #[serde(default)]
    samplers: Vec<AnimationSamplerDef>,
    #[serde(default)]
    channels: Vec<AnimationChannelDef>,
}

#[derive(Deserialize)]
struct AnimationSamplerDef {
    input: usize,
    output: usize,
    #[serde(default)]
    interpolation: Option<String>,
}

#[derive(Deserialize)]
struct AnimationChannelDef {
    sampler: usize,
    target: AnimationTargetDef,
}

#[derive(Deserialize)]
struct AnimationTargetDef {
    #[serde(default)]
    node: Option<usize>,
    path: String,
}

struct AccessorLayout<'a> {
    accessor: &'a AccessorDef,
    data: &'a [u8],
    start: usize,
    stride: usize,
    elem_size: usize,
    count: usize,
    view_end: usize,
}

#[derive(Clone)]
struct MaterialBinding {
    material: Material,
    texcoord_set: usize,
}

#[derive(Clone)]
struct LoadedPrimitive {
    mesh: Mesh,
    material: Material,
}

#[derive(Clone, Copy)]
enum AnimationInterpolation {
    Linear,
    Step,
    CubicSpline,
}

fn component_size(component_type: u32) -> Option<usize> {
    match component_type {
        5120 | 5121 => Some(1),
        5122 | 5123 => Some(2),
        5125 | 5126 => Some(4),
        _ => None,
    }
}

fn accessor_components(kind: &str) -> Option<usize> {
    match kind {
        "SCALAR" => Some(1),
        "VEC2" => Some(2),
        "VEC3" => Some(3),
        "VEC4" => Some(4),
        "MAT2" => Some(4),
        "MAT3" => Some(9),
        "MAT4" => Some(16),
        _ => None,
    }
}

fn checked_add(a: usize, b: usize) -> Result<usize, GltfError> {
    a.checked_add(b).ok_or(GltfError::BufferOutOfBounds)
}

fn checked_mul(a: usize, b: usize) -> Result<usize, GltfError> {
    a.checked_mul(b).ok_or(GltfError::BufferOutOfBounds)
}

fn accessor_layout<'a>(
    root: &'a GltfRoot,
    buffers: &'a [Vec<u8>],
    accessor_index: usize,
) -> Result<AccessorLayout<'a>, GltfError> {
    let accessor = root.accessors.get(accessor_index).ok_or(GltfError::InvalidIndex)?;
    if accessor.sparse.is_some() {
        return Err(GltfError::UnsupportedAccessor);
    }
    let view_index = accessor.buffer_view.ok_or(GltfError::UnsupportedAccessor)?;
    let view = root.buffer_views.get(view_index).ok_or(GltfError::InvalidIndex)?;
    let data = buffers.get(view.buffer).ok_or(GltfError::InvalidIndex)?.as_slice();

    let components = accessor_components(&accessor.kind).ok_or(GltfError::UnsupportedAccessor)?;
    let component_size = component_size(accessor.component_type).ok_or(GltfError::UnsupportedAccessor)?;
    let elem_size = checked_mul(components, component_size)?;
    let stride = view.byte_stride.unwrap_or(elem_size);
    if stride < elem_size {
        return Err(GltfError::BufferOutOfBounds);
    }

    let view_start = view.byte_offset;
    let view_end = checked_add(view.byte_offset, view.byte_length)?;
    if view_end > data.len() {
        return Err(GltfError::BufferOutOfBounds);
    }

    let start = checked_add(view_start, accessor.byte_offset)?;
    if start > view_end {
        return Err(GltfError::BufferOutOfBounds);
    }

    if accessor.count > 0 {
        let last_index = accessor.count - 1;
        let last_stride = checked_mul(last_index, stride)?;
        let last_start = checked_add(start, last_stride)?;
        let last_end = checked_add(last_start, elem_size)?;
        if last_end > view_end {
            return Err(GltfError::BufferOutOfBounds);
        }
    }

    Ok(AccessorLayout {
        accessor,
        data,
        start,
        stride,
        elem_size,
        count: accessor.count,
        view_end,
    })
}

fn accessor_elem<'a>(layout: &'a AccessorLayout<'_>, index: usize) -> Result<&'a [u8], GltfError> {
    if index >= layout.count {
        return Err(GltfError::BufferOutOfBounds);
    }
    let off = checked_add(layout.start, checked_mul(index, layout.stride)?)?;
    let end = checked_add(off, layout.elem_size)?;
    if end > layout.view_end || end > layout.data.len() {
        return Err(GltfError::BufferOutOfBounds);
    }
    Ok(&layout.data[off..end])
}

fn read_u16_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_accessor_scalar_f32(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<f32>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "SCALAR" || layout.accessor.component_type != 5126 {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    for i in 0..layout.count {
        let e = accessor_elem(&layout, i)?;
        out.push(read_f32_le(&e[0..4]));
    }
    Ok(out)
}

fn read_accessor_vec3(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<Vec3>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "VEC3" || layout.accessor.component_type != 5126 {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    for i in 0..layout.count {
        let e = accessor_elem(&layout, i)?;
        out.push(Vec3::new(
            read_f32_le(&e[0..4]),
            read_f32_le(&e[4..8]),
            read_f32_le(&e[8..12]),
        ));
    }
    Ok(out)
}

fn read_accessor_vec2(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<Vec2>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "VEC2" || layout.accessor.component_type != 5126 {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    for i in 0..layout.count {
        let e = accessor_elem(&layout, i)?;
        out.push(Vec2::new(read_f32_le(&e[0..4]), read_f32_le(&e[4..8])));
    }
    Ok(out)
}

fn read_accessor_vec4(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<Vec4>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "VEC4" || layout.accessor.component_type != 5126 {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    for i in 0..layout.count {
        let e = accessor_elem(&layout, i)?;
        out.push(Vec4::new(
            read_f32_le(&e[0..4]),
            read_f32_le(&e[4..8]),
            read_f32_le(&e[8..12]),
            read_f32_le(&e[12..16]),
        ));
    }
    Ok(out)
}

fn read_accessor_indices(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<u32>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "SCALAR" {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    match layout.accessor.component_type {
        5121 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(u32::from(e[0]));
            }
        }
        5123 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(u32::from(read_u16_le(e)));
            }
        }
        5125 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(read_u32_le(e));
            }
        }
        _ => return Err(GltfError::UnsupportedAccessor),
    }
    Ok(out)
}

fn primitive_to_mesh(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    primitive: &PrimitiveDef,
    texcoord_set: usize,
    require_uv: bool,
) -> Result<Mesh, GltfError> {
    if let Some(mode) = primitive.mode {
        if mode != 4 {
            return Err(GltfError::UnsupportedPrimitiveMode);
        }
    }

    let pos_accessor = *primitive.attributes.get("POSITION").ok_or(GltfError::MissingAttribute)?;
    let positions = read_accessor_vec3(root, buffers, pos_accessor)?;

    let normals = if let Some(n) = primitive.attributes.get("NORMAL") {
        read_accessor_vec3(root, buffers, *n)?
    } else {
        Vec::new()
    };

    let uv_semantic = format!("TEXCOORD_{texcoord_set}");
    let uvs = if let Some(uv) = primitive.attributes.get(&uv_semantic) {
        read_accessor_vec2(root, buffers, *uv)?
    } else if require_uv {
        return Err(GltfError::MissingAttribute);
    } else {
        Vec::new()
    };

    let index_data = if let Some(idx_accessor) = primitive.indices {
        read_accessor_indices(root, buffers, idx_accessor)?
    } else {
        (0..positions.len()).map(|v| v as u32).collect()
    };

    if index_data.len() % 3 != 0 {
        return Err(GltfError::UnsupportedPrimitiveMode);
    }

    for &ix in &index_data {
        if (ix as usize) >= positions.len() {
            return Err(GltfError::InvalidIndex);
        }
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.uvs = uvs;
    mesh.indices = index_data
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    if mesh.normals.is_empty() {
        mesh.ensure_normals();
    }

    Ok(mesh)
}

fn quat_from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Result<Quat, GltfError> {
    let q = Quat::from_xyzw(x, y, z, w);
    if !q.is_finite() || q.length_squared() <= 0.0 {
        return Err(GltfError::InvalidTransform);
    }
    Ok(q.normalize())
}

fn node_base_trs(node: &NodeDef) -> Result<(Vec3, Quat, Vec3), GltfError> {
    if let Some(arr) = node.matrix {
        let m = Mat4::from_cols_array(&arr);
        if !m.is_finite() {
            return Err(GltfError::InvalidTransform);
        }
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
            return Err(GltfError::InvalidTransform);
        }
        return Ok((translation, rotation.normalize(), scale));
    }

    let t = node.translation.unwrap_or([0.0, 0.0, 0.0]);
    let r = node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let s = node.scale.unwrap_or([1.0, 1.0, 1.0]);

    Ok((
        Vec3::new(t[0], t[1], t[2]),
        quat_from_xyzw(r[0], r[1], r[2], r[3])?,
        Vec3::new(s[0], s[1], s[2]),
    ))
}

fn node_local_matrix(node: &NodeDef) -> Result<Mat4, GltfError> {
    let (t, r, s) = node_base_trs(node)?;
    let m = Mat4::from_translation(t) * Mat4::from_quat(r) * Mat4::from_scale(s);
    if !m.is_finite() {
        return Err(GltfError::InvalidTransform);
    }
    Ok(m)
}

fn world_to_transform(world: Mat4) -> Result<Transform, GltfError> {
    if !world.is_finite() {
        return Err(GltfError::InvalidTransform);
    }
    let (scale, rotation, translation) = world.to_scale_rotation_translation();
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
        return Err(GltfError::InvalidTransform);
    }
    Ok(Transform::new(translation, rotation.normalize(), scale))
}

fn default_material_binding() -> MaterialBinding {
    let mut material = Material::default();
    material.kd = Vec3::ONE;
    material.alpha = 1.0;
    material.map_kd = None;
    material.map_kd_path = None;
    MaterialBinding {
        material,
        texcoord_set: 0,
    }
}

fn srgb_to_linear_u8(v: u8) -> u8 {
    let s = (v as f32) / 255.0;
    let l = if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    };
    ((l.clamp(0.0, 1.0) * 255.0) + 0.5).floor() as u8
}

fn linearize_base_color_texture(tex: &mut Texture) {
    for px in tex.rgba8.chunks_exact_mut(4) {
        px[0] = srgb_to_linear_u8(px[0]);
        px[1] = srgb_to_linear_u8(px[1]);
        px[2] = srgb_to_linear_u8(px[2]);
    }
}

fn view_bytes<'a>(root: &'a GltfRoot, buffers: &'a [Vec<u8>], view_index: usize) -> Result<&'a [u8], GltfError> {
    let view = root.buffer_views.get(view_index).ok_or(GltfError::InvalidIndex)?;
    let data = buffers.get(view.buffer).ok_or(GltfError::InvalidIndex)?.as_slice();
    let start = view.byte_offset;
    let end = checked_add(start, view.byte_length)?;
    if end > data.len() {
        return Err(GltfError::BufferOutOfBounds);
    }
    Ok(&data[start..end])
}

fn decode_data_uri(uri: &str) -> Result<Vec<u8>, GltfError> {
    if !uri.starts_with("data:") {
        return Err(GltfError::UnsupportedDataUri);
    }
    let marker = ";base64,";
    let idx = uri.find(marker).ok_or(GltfError::UnsupportedDataUri)?;
    let encoded = &uri[(idx + marker.len())..];
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| GltfError::DecodeBase64)
}

fn load_buffer_bytes(buffer: &BufferDef, base_dir: Option<&Path>) -> Result<Vec<u8>, GltfError> {
    let Some(uri) = &buffer.uri else {
        return Err(GltfError::UnsupportedBinaryGltf);
    };

    let bytes = if uri.starts_with("data:") {
        decode_data_uri(uri)?
    } else {
        let base = base_dir.ok_or(GltfError::MissingBasePath)?;
        let path = base.join(uri);
        fs::read(path).map_err(|_| GltfError::Io)?
    };

    if bytes.len() < buffer.byte_length {
        return Err(GltfError::BufferOutOfBounds);
    }
    Ok(bytes)
}

fn load_image_bytes(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    image_index: usize,
) -> Result<Vec<u8>, GltfError> {
    let image = root.images.get(image_index).ok_or(GltfError::InvalidIndex)?;

    if let Some(uri) = &image.uri {
        if uri.starts_with("data:") {
            return decode_data_uri(uri);
        }
        let base = base_dir.ok_or(GltfError::MissingBasePath)?;
        let path = base.join(uri);
        return fs::read(path).map_err(|_| GltfError::Io);
    }

    if let Some(view_index) = image.buffer_view {
        return Ok(view_bytes(root, buffers, view_index)?.to_vec());
    }

    Err(GltfError::MissingImageSource)
}

fn map_texture_io_error(err: TextureIoError) -> GltfError {
    match err {
        TextureIoError::ImageFeatureDisabled => GltfError::ImageFeatureDisabled,
        TextureIoError::Io | TextureIoError::Decode | TextureIoError::Invalid => GltfError::TextureIo,
    }
}

fn ensure_base_color_texture_handle(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    scene: &mut Scene,
    texture_cache: &mut [Option<TextureHandle>],
    texture_index: usize,
) -> Result<TextureHandle, GltfError> {
    let Some(slot) = texture_cache.get_mut(texture_index) else {
        return Err(GltfError::InvalidIndex);
    };

    if let Some(handle) = *slot {
        return Ok(handle);
    }

    let tex_def = root.textures.get(texture_index).ok_or(GltfError::InvalidIndex)?;
    let image_index = tex_def.source.ok_or(GltfError::MissingImageSource)?;
    let bytes = load_image_bytes(root, buffers, base_dir, image_index)?;
    let mut texture = load_texture_rgba8_from_bytes(&bytes).map_err(map_texture_io_error)?;
    linearize_base_color_texture(&mut texture);
    let handle = scene.add_texture(texture);
    *slot = Some(handle);
    Ok(handle)
}

fn build_material_bindings(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    scene: &mut Scene,
) -> Result<Vec<MaterialBinding>, GltfError> {
    let mut texture_cache: Vec<Option<TextureHandle>> = vec![None; root.textures.len()];
    let mut out = Vec::with_capacity(root.materials.len());

    for material in &root.materials {
        let mut binding = default_material_binding();

        if let Some(pbr) = &material.pbr_metallic_roughness {
            let factor = pbr.base_color_factor.unwrap_or([1.0, 1.0, 1.0, 1.0]);
            binding.material.kd = Vec3::new(
                factor[0].clamp(0.0, 1.0),
                factor[1].clamp(0.0, 1.0),
                factor[2].clamp(0.0, 1.0),
            );
            binding.material.alpha = factor[3].clamp(0.0, 1.0);

            if let Some(tex_info) = &pbr.base_color_texture {
                let handle = ensure_base_color_texture_handle(
                    root,
                    buffers,
                    base_dir,
                    scene,
                    &mut texture_cache,
                    tex_info.index,
                )?;
                binding.material.map_kd = Some(handle);
                binding.texcoord_set = tex_info.tex_coord.unwrap_or(0);
            }
        }

        out.push(binding);
    }

    Ok(out)
}

fn parse_interpolation(s: Option<&str>) -> Result<AnimationInterpolation, GltfError> {
    match s.unwrap_or("LINEAR") {
        "LINEAR" => Ok(AnimationInterpolation::Linear),
        "STEP" => Ok(AnimationInterpolation::Step),
        "CUBICSPLINE" => Ok(AnimationInterpolation::CubicSpline),
        _ => Err(GltfError::UnsupportedInterpolation),
    }
}

fn validate_keyframe_times(times: &[f32]) -> Result<(), GltfError> {
    if times.is_empty() {
        return Err(GltfError::InvalidAnimation);
    }
    if !times[0].is_finite() || times[0] < 0.0 {
        return Err(GltfError::InvalidAnimation);
    }
    for i in 1..times.len() {
        if !times[i].is_finite() || times[i] <= times[i - 1] {
            return Err(GltfError::InvalidAnimation);
        }
    }
    Ok(())
}

fn find_segment(times: &[f32], time: f32) -> Result<(usize, usize, f32, f32), GltfError> {
    if times.is_empty() {
        return Err(GltfError::InvalidAnimation);
    }
    if times.len() == 1 {
        return Ok((0, 0, 0.0, 0.0));
    }
    if time <= times[0] {
        return Ok((0, 0, 0.0, 0.0));
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return Ok((last, last, 0.0, 0.0));
    }

    let mut lo = 0usize;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if time < times[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let dt = times[hi] - times[lo];
    let t = if dt > 0.0 {
        ((time - times[lo]) / dt).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok((lo, hi, t, dt.max(0.0)))
}

fn sample_vec3_channel(
    times: &[f32],
    values: &[Vec3],
    interpolation: AnimationInterpolation,
    time: f32,
) -> Result<Vec3, GltfError> {
    validate_keyframe_times(times)?;
    let (i0, i1, t, dt) = find_segment(times, time)?;

    match interpolation {
        AnimationInterpolation::Step => {
            if values.len() != times.len() {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 >= values.len() {
                return Err(GltfError::InvalidAnimation);
            }
            Ok(values[i0])
        }
        AnimationInterpolation::Linear => {
            if values.len() != times.len() {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 >= values.len() || i1 >= values.len() {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 == i1 {
                Ok(values[i0])
            } else {
                Ok(values[i0].lerp(values[i1], t))
            }
        }
        AnimationInterpolation::CubicSpline => {
            if times.len() < 2 {
                return Err(GltfError::InvalidAnimation);
            }
            let need = times.len().checked_mul(3).ok_or(GltfError::InvalidAnimation)?;
            if values.len() != need {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 == i1 {
                return Ok(values[i0 * 3 + 1]);
            }

            let v0 = values[i0 * 3 + 1];
            let out_tan = values[i0 * 3 + 2];
            let in_tan = values[i1 * 3];
            let v1 = values[i1 * 3 + 1];

            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;

            Ok(v0 * h00 + out_tan * (h10 * dt) + v1 * h01 + in_tan * (h11 * dt))
        }
    }
}

fn sample_quat_channel(
    times: &[f32],
    values: &[Vec4],
    interpolation: AnimationInterpolation,
    time: f32,
) -> Result<Quat, GltfError> {
    validate_keyframe_times(times)?;
    let (i0, i1, t, dt) = find_segment(times, time)?;

    match interpolation {
        AnimationInterpolation::Step => {
            if values.len() != times.len() {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 >= values.len() {
                return Err(GltfError::InvalidAnimation);
            }
            quat_from_xyzw(values[i0].x, values[i0].y, values[i0].z, values[i0].w)
        }
        AnimationInterpolation::Linear => {
            if values.len() != times.len() {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 >= values.len() || i1 >= values.len() {
                return Err(GltfError::InvalidAnimation);
            }
            let q0 = quat_from_xyzw(values[i0].x, values[i0].y, values[i0].z, values[i0].w)?;
            if i0 == i1 {
                return Ok(q0);
            }
            let mut q1 = quat_from_xyzw(values[i1].x, values[i1].y, values[i1].z, values[i1].w)?;
            if q0.dot(q1) < 0.0 {
                q1 = -q1;
            }
            Ok(q0.slerp(q1, t).normalize())
        }
        AnimationInterpolation::CubicSpline => {
            if times.len() < 2 {
                return Err(GltfError::InvalidAnimation);
            }
            let need = times.len().checked_mul(3).ok_or(GltfError::InvalidAnimation)?;
            if values.len() != need {
                return Err(GltfError::InvalidAnimation);
            }
            if i0 == i1 {
                let v = values[i0 * 3 + 1];
                return quat_from_xyzw(v.x, v.y, v.z, v.w);
            }

            let v0 = values[i0 * 3 + 1];
            let out_tan = values[i0 * 3 + 2];
            let in_tan = values[i1 * 3];
            let v1 = values[i1 * 3 + 1];

            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;

            let q = v0 * h00 + out_tan * (h10 * dt) + v1 * h01 + in_tan * (h11 * dt);
            quat_from_xyzw(q.x, q.y, q.z, q.w)
        }
    }
}

fn sample_animation_locals(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    animation_index: usize,
    time_seconds: f32,
) -> Result<Vec<Mat4>, GltfError> {
    let anim = root.animations.get(animation_index).ok_or(GltfError::InvalidIndex)?;
    let time = if time_seconds.is_finite() { time_seconds } else { 0.0 };

    let mut translations = Vec::with_capacity(root.nodes.len());
    let mut rotations = Vec::with_capacity(root.nodes.len());
    let mut scales = Vec::with_capacity(root.nodes.len());
    for node in &root.nodes {
        let (t, r, s) = node_base_trs(node)?;
        translations.push(t);
        rotations.push(r);
        scales.push(s);
    }

    for channel in &anim.channels {
        let sampler = anim.samplers.get(channel.sampler).ok_or(GltfError::InvalidIndex)?;
        let node_index = channel.target.node.ok_or(GltfError::InvalidAnimation)?;
        if node_index >= root.nodes.len() {
            return Err(GltfError::InvalidIndex);
        }
        if root.nodes[node_index].matrix.is_some() {
            return Err(GltfError::InvalidAnimation);
        }
        let interpolation = parse_interpolation(sampler.interpolation.as_deref())?;
        let input_times = read_accessor_scalar_f32(root, buffers, sampler.input)?;

        match channel.target.path.as_str() {
            "translation" => {
                let output = read_accessor_vec3(root, buffers, sampler.output)?;
                translations[node_index] =
                    sample_vec3_channel(&input_times, &output, interpolation, time)?;
            }
            "scale" => {
                let output = read_accessor_vec3(root, buffers, sampler.output)?;
                scales[node_index] = sample_vec3_channel(&input_times, &output, interpolation, time)?;
            }
            "rotation" => {
                let output = read_accessor_vec4(root, buffers, sampler.output)?;
                rotations[node_index] = sample_quat_channel(&input_times, &output, interpolation, time)?;
            }
            _ => return Err(GltfError::UnsupportedAnimationPath),
        }
    }

    let mut locals = Vec::with_capacity(root.nodes.len());
    for i in 0..root.nodes.len() {
        locals.push(
            Mat4::from_translation(translations[i])
                * Mat4::from_quat(rotations[i])
                * Mat4::from_scale(scales[i]),
        );
    }

    Ok(locals)
}

fn build_scene(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    animated_locals: Option<&[Mat4]>,
) -> Result<Scene, GltfError> {
    let mut scene = Scene::new();
    let materials = build_material_bindings(root, buffers, base_dir, &mut scene)?;
    let default_binding = default_material_binding();

    let mut per_mesh_primitives: Vec<Vec<LoadedPrimitive>> = Vec::with_capacity(root.meshes.len());
    for mesh in &root.meshes {
        let mut prims = Vec::with_capacity(mesh.primitives.len());
        for p in &mesh.primitives {
            let binding = if let Some(material_index) = p.material {
                materials.get(material_index).ok_or(GltfError::InvalidIndex)?
            } else {
                &default_binding
            };

            let mesh = primitive_to_mesh(
                root,
                buffers,
                p,
                binding.texcoord_set,
                binding.material.map_kd.is_some(),
            )?;

            prims.push(LoadedPrimitive {
                mesh,
                material: binding.material.clone(),
            });
        }
        per_mesh_primitives.push(prims);
    }

    let mut visiting = vec![false; root.nodes.len()];

    fn walk(
        node_index: usize,
        root: &GltfRoot,
        per_mesh_primitives: &[Vec<LoadedPrimitive>],
        animated_locals: Option<&[Mat4]>,
        parent_world: Mat4,
        visiting: &mut [bool],
        scene: &mut Scene,
    ) -> Result<(), GltfError> {
        let node = root.nodes.get(node_index).ok_or(GltfError::InvalidIndex)?;
        if visiting[node_index] {
            return Err(GltfError::InvalidNodeGraph);
        }
        visiting[node_index] = true;

        let local = if let Some(locals) = animated_locals {
            *locals.get(node_index).ok_or(GltfError::InvalidIndex)?
        } else {
            node_local_matrix(node)?
        };
        let world = parent_world * local;

        if let Some(mesh_index) = node.mesh {
            let primitives = per_mesh_primitives.get(mesh_index).ok_or(GltfError::InvalidIndex)?;
            let xf = world_to_transform(world)?;
            for prim in primitives {
                let _ = scene.add_object(prim.mesh.clone(), xf, prim.material.clone());
            }
        }

        for &child in &node.children {
            walk(
                child,
                root,
                per_mesh_primitives,
                animated_locals,
                world,
                visiting,
                scene,
            )?;
        }

        visiting[node_index] = false;
        Ok(())
    }

    if root.scenes.is_empty() {
        for i in 0..root.nodes.len() {
            walk(
                i,
                root,
                &per_mesh_primitives,
                animated_locals,
                Mat4::IDENTITY,
                &mut visiting,
                &mut scene,
            )?;
        }
        return Ok(scene);
    }

    let scene_index = root.scene.unwrap_or(0);
    let selected = root.scenes.get(scene_index).ok_or(GltfError::InvalidIndex)?;
    for &node in &selected.nodes {
        walk(
            node,
            root,
            &per_mesh_primitives,
            animated_locals,
            Mat4::IDENTITY,
            &mut visiting,
            &mut scene,
        )?;
    }
    Ok(scene)
}

fn parse_and_load(
    src: &str,
    base_dir: Option<&Path>,
    animation_sample: Option<(usize, f32)>,
) -> Result<Scene, GltfError> {
    let root: GltfRoot = serde_json::from_str(src).map_err(|_| GltfError::Json)?;
    if root.asset.version != "2.0" {
        return Err(GltfError::UnsupportedVersion);
    }

    let mut buffers = Vec::with_capacity(root.buffers.len());
    for b in &root.buffers {
        buffers.push(load_buffer_bytes(b, base_dir)?);
    }

    let animated_locals = if let Some((animation_index, time_seconds)) = animation_sample {
        Some(sample_animation_locals(
            &root,
            &buffers,
            animation_index,
            time_seconds,
        )?)
    } else {
        None
    };

    build_scene(&root, &buffers, base_dir, animated_locals.as_deref())
}

pub fn load_gltf(path: impl AsRef<Path>) -> Result<Scene, GltfError> {
    let path = path.as_ref();
    let src = fs::read_to_string(path).map_err(|_| GltfError::Io)?;
    let base_dir: PathBuf = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parse_and_load(&src, Some(&base_dir), None)
}

pub fn load_gltf_at_time(
    path: impl AsRef<Path>,
    animation_index: usize,
    time_seconds: f32,
) -> Result<Scene, GltfError> {
    let path = path.as_ref();
    let src = fs::read_to_string(path).map_err(|_| GltfError::Io)?;
    let base_dir: PathBuf = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parse_and_load(
        &src,
        Some(&base_dir),
        Some((animation_index, time_seconds)),
    )
}

pub fn load_gltf_str(src: &str) -> Result<Scene, GltfError> {
    parse_and_load(src, None, None)
}

pub fn load_gltf_str_at_time(
    src: &str,
    animation_index: usize,
    time_seconds: f32,
) -> Result<Scene, GltfError> {
    parse_and_load(src, None, Some((animation_index, time_seconds)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera, DebugView, Projection, Renderer, RendererConfig};

    fn tiny_gltf_inline() -> String {
        let mut bytes = Vec::new();

        let positions = [
            -0.5f32, -0.5, 0.0, //
            0.5, -0.5, 0.0, //
            0.0, 0.5, 0.0,
        ];
        for v in positions {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let indices = [0u16, 1u16, 2u16];
        for i in indices {
            bytes.extend_from_slice(&i.to_le_bytes());
        }

        use base64::Engine as _;
        let payload = base64::engine::general_purpose::STANDARD.encode(bytes);

        format!(
            r#"{{
"asset":{{"version":"2.0"}},
"buffers":[{{"uri":"data:application/octet-stream;base64,{payload}","byteLength":42}}],
"bufferViews":[
  {{"buffer":0,"byteOffset":0,"byteLength":36}},
  {{"buffer":0,"byteOffset":36,"byteLength":6}}
],
"accessors":[
  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
  {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}
],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],
"nodes":[
  {{"children":[1],"translation":[0.25,0.0,0.0]}},
  {{"mesh":0,"translation":[0.25,0.0,0.0]}}
],
"scenes":[{{"nodes":[0]}}],
"scene":0
}}"#
        )
    }

    fn tiny_textured_gltf_inline() -> String {
        let mut geom = Vec::new();

        let positions = [
            -1.0f32, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            1.0, -1.0, 0.0, //
            -1.0, -1.0, 0.0,
        ];
        for v in positions {
            geom.extend_from_slice(&v.to_le_bytes());
        }

        let uvs = [
            0.0f32, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            0.0, 1.0,
        ];
        for v in uvs {
            geom.extend_from_slice(&v.to_le_bytes());
        }

        let indices = [0u16, 1u16, 2u16, 0u16, 2u16, 3u16];
        for i in indices {
            geom.extend_from_slice(&i.to_le_bytes());
        }

        let mut image_rgba = image::RgbaImage::new(2, 2);
        image_rgba.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        image_rgba.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
        image_rgba.put_pixel(0, 1, image::Rgba([0, 0, 255, 255]));
        image_rgba.put_pixel(1, 1, image::Rgba([255, 255, 255, 255]));

        let mut png_bytes = Vec::new();
        {
            let dyn_img = image::DynamicImage::ImageRgba8(image_rgba);
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            dyn_img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
        }

        use base64::Engine as _;
        let geom_payload = base64::engine::general_purpose::STANDARD.encode(geom);
        let image_payload = base64::engine::general_purpose::STANDARD.encode(png_bytes);

        format!(
            r#"{{
"asset":{{"version":"2.0"}},
"buffers":[{{"uri":"data:application/octet-stream;base64,{geom_payload}","byteLength":92}}],
"bufferViews":[
  {{"buffer":0,"byteOffset":0,"byteLength":48}},
  {{"buffer":0,"byteOffset":48,"byteLength":32}},
  {{"buffer":0,"byteOffset":80,"byteLength":12}}
],
"accessors":[
  {{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"}},
  {{"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"}},
  {{"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}}
],
"images":[{{"uri":"data:image/png;base64,{image_payload}"}}],
"textures":[{{"source":0}}],
"materials":[{{
  "pbrMetallicRoughness":{{
    "baseColorFactor":[0.5,1.0,1.0,1.0],
    "baseColorTexture":{{"index":0,"texCoord":0}}
  }}
}}],
"meshes":[{{"primitives":[{{
  "attributes":{{"POSITION":0,"TEXCOORD_0":1}},
  "indices":2,
  "material":0
}}]}}],
"nodes":[{{"mesh":0}}],
"scenes":[{{"nodes":[0]}}],
"scene":0
}}"#
        )
    }

    fn tiny_animated_gltf_inline() -> String {
        let mut bytes = Vec::new();

        let positions = [
            -0.25f32, -0.25, 0.0, //
            0.25, -0.25, 0.0, //
            0.0, 0.25, 0.0,
        ];
        for v in positions {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let times = [0.0f32, 1.0f32];
        for t in times {
            bytes.extend_from_slice(&t.to_le_bytes());
        }

        let translations = [
            -0.45f32, 0.0, 0.0, //
            0.45, 0.0, 0.0,
        ];
        for v in translations {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let indices = [0u16, 1u16, 2u16];
        for i in indices {
            bytes.extend_from_slice(&i.to_le_bytes());
        }

        use base64::Engine as _;
        let payload = base64::engine::general_purpose::STANDARD.encode(bytes);

        format!(
            r#"{{
"asset":{{"version":"2.0"}},
"buffers":[{{"uri":"data:application/octet-stream;base64,{payload}","byteLength":74}}],
"bufferViews":[
  {{"buffer":0,"byteOffset":0,"byteLength":36}},
  {{"buffer":0,"byteOffset":36,"byteLength":8}},
  {{"buffer":0,"byteOffset":44,"byteLength":24}},
  {{"buffer":0,"byteOffset":68,"byteLength":6}}
],
"accessors":[
  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
  {{"bufferView":1,"componentType":5126,"count":2,"type":"SCALAR"}},
  {{"bufferView":2,"componentType":5126,"count":2,"type":"VEC3"}},
  {{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}}
],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":3}}]}}],
"nodes":[{{"mesh":0}}],
"animations":[{{
  "samplers":[{{"input":1,"output":2,"interpolation":"LINEAR"}}],
  "channels":[{{"sampler":0,"target":{{"node":0,"path":"translation"}}}}]
}}],
"scenes":[{{"nodes":[0]}}],
"scene":0
}}"#
        )
    }

    #[test]
    fn gltf_inline_loads_mesh_nodes_and_transforms() {
        let gltf = tiny_gltf_inline();
        let scene = load_gltf_str(&gltf).unwrap();
        assert_eq!(scene.object_count(), 1);

        let mut saw = false;
        for (mesh, _mat, xf) in scene.iter_objects() {
            saw = true;
            assert_eq!(mesh.indices.len(), 1);
            assert_eq!(mesh.positions.len(), 3);
            assert!((xf.translation.x - 0.5).abs() < 1e-6);
        }
        assert!(saw);
    }

    #[test]
    fn gltf_inline_renders_deterministically() {
        let gltf = tiny_gltf_inline();
        let scene = load_gltf_str(&gltf).unwrap();
        let renderer = Renderer::new(RendererConfig::default().with_size(64, 32));
        let mut img = crate::targets::ImageTarget::new(64, 32);
        let empty = img.hash64();
        renderer.render_image(&scene, &mut img);
        let h1 = img.hash64();
        assert_ne!(h1, empty);
        renderer.render_image(&scene, &mut img);
        let h2 = img.hash64();
        assert_eq!(h1, h2);
    }

    #[test]
    fn gltf_textured_material_maps_and_renders_snapshot() {
        let gltf = tiny_textured_gltf_inline();
        let mut scene = load_gltf_str(&gltf).unwrap();
        assert_eq!(scene.object_count(), 1);

        let mut saw = false;
        for (_mesh, mat, _xf) in scene.iter_objects() {
            saw = true;
            assert!((mat.kd.x - 0.5).abs() < 1e-6);
            assert!((mat.kd.y - 1.0).abs() < 1e-6);
            assert!((mat.kd.z - 1.0).abs() < 1e-6);
            assert!((mat.alpha - 1.0).abs() < 1e-6);
            assert!(mat.map_kd.is_some());
        }
        assert!(saw);

        scene.camera = Camera::new(
            Transform::IDENTITY,
            Projection::Orthographic {
                half_height: 1.0,
                near: -10.0,
                far: 10.0,
            },
        );

        let renderer = Renderer::new(
            RendererConfig::default()
                .with_size(32, 32)
                .with_debug_view(DebugView::Albedo),
        );

        let mut img = crate::targets::ImageTarget::new(32, 32);
        renderer.render_image(&scene, &mut img);

        const EXPECTED: u64 = 17_340_975_637_531_158_088;
        assert_eq!(img.hash64(), EXPECTED);
    }

    #[test]
    fn gltf_animation_trs_frames_differ_and_are_deterministic() {
        let gltf = tiny_animated_gltf_inline();

        let mut scene_a = load_gltf_str_at_time(&gltf, 0, 0.0).unwrap();
        let mut scene_b = load_gltf_str_at_time(&gltf, 0, 1.0).unwrap();
        let mut scene_a_repeat = load_gltf_str_at_time(&gltf, 0, 0.0).unwrap();

        for scene in [&mut scene_a, &mut scene_b, &mut scene_a_repeat] {
            scene.camera = Camera::new(
                Transform::IDENTITY,
                Projection::Orthographic {
                    half_height: 1.0,
                    near: -10.0,
                    far: 10.0,
                },
            );
        }

        let mut x_a = None;
        for (_mesh, _mat, xf) in scene_a.iter_objects() {
            x_a = Some(xf.translation.x);
        }
        let mut x_b = None;
        for (_mesh, _mat, xf) in scene_b.iter_objects() {
            x_b = Some(xf.translation.x);
        }
        assert!((x_a.unwrap() + 0.45).abs() < 1e-6);
        assert!((x_b.unwrap() - 0.45).abs() < 1e-6);

        let renderer = Renderer::new(
            RendererConfig::default()
                .with_size(64, 32)
                .with_debug_view(DebugView::Albedo),
        );

        let mut img_a = crate::targets::ImageTarget::new(64, 32);
        let mut img_b = crate::targets::ImageTarget::new(64, 32);
        let mut img_a_repeat = crate::targets::ImageTarget::new(64, 32);

        renderer.render_image(&scene_a, &mut img_a);
        renderer.render_image(&scene_b, &mut img_b);
        renderer.render_image(&scene_a_repeat, &mut img_a_repeat);

        let h_a = img_a.hash64();
        let h_b = img_b.hash64();
        let h_a_repeat = img_a_repeat.hash64();

        assert_eq!(h_a, h_a_repeat);
        assert_ne!(h_a, h_b);
    }
}

