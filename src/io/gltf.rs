use crate::texture::{
    Texture, TextureHandle, TextureMagFilter, TextureMinFilter, TextureSampler, TextureWrapMode,
};
use crate::{
    io::texture::{load_texture_rgba8_from_bytes_raw, TextureIoError},
    AlphaMode, Camera, Light, Material, Mesh, Projection, Scene, Transform,
};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use serde::Deserialize;
use std::{
    collections::HashMap,
    error::Error,
    fmt, fs,
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
    UnsupportedMaterialFeature,
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
            Self::UnsupportedMaterialFeature => write!(f, "unsupported material feature"),
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
    samplers: Vec<SamplerDef>,
    #[serde(default)]
    images: Vec<ImageDef>,
    #[serde(default)]
    skins: Vec<SkinDef>,
    #[serde(default)]
    nodes: Vec<NodeDef>,
    #[serde(default)]
    scenes: Vec<SceneDef>,
    #[serde(default)]
    scene: Option<usize>,
    #[serde(default)]
    animations: Vec<AnimationDef>,
    #[serde(default)]
    cameras: Vec<CameraDef>,
    #[serde(default)]
    extensions: Option<RootExtensionsDef>,
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
    normalized: bool,
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
    #[serde(rename = "alphaMode", default)]
    alpha_mode: Option<String>,
    #[serde(rename = "alphaCutoff", default)]
    alpha_cutoff: Option<f32>,
    #[serde(rename = "doubleSided", default)]
    double_sided: Option<bool>,
    #[serde(rename = "emissiveFactor", default)]
    emissive_factor: Option<[f32; 3]>,
    #[serde(rename = "normalTexture", default)]
    normal_texture: Option<NormalTextureInfoDef>,
    #[serde(rename = "occlusionTexture", default)]
    occlusion_texture: Option<OcclusionTextureInfoDef>,
    #[serde(rename = "emissiveTexture", default)]
    emissive_texture: Option<TextureInfoDef>,
}

#[derive(Clone, Deserialize)]
struct NormalTextureInfoDef {
    index: usize,
    #[serde(rename = "texCoord", default)]
    tex_coord: Option<usize>,
    #[serde(default)]
    scale: Option<f32>,
    #[serde(default)]
    extensions: Option<TextureInfoExtensionsDef>,
}

#[derive(Clone, Deserialize)]
struct OcclusionTextureInfoDef {
    index: usize,
    #[serde(rename = "texCoord", default)]
    tex_coord: Option<usize>,
    #[serde(default)]
    strength: Option<f32>,
    #[serde(default)]
    extensions: Option<TextureInfoExtensionsDef>,
}

#[derive(Deserialize)]
struct PbrMetallicRoughnessDef {
    #[serde(rename = "baseColorFactor", default)]
    base_color_factor: Option<[f32; 4]>,
    #[serde(rename = "baseColorTexture", default)]
    base_color_texture: Option<TextureInfoDef>,
    #[serde(rename = "metallicFactor", default)]
    metallic_factor: Option<f32>,
    #[serde(rename = "roughnessFactor", default)]
    roughness_factor: Option<f32>,
    #[serde(rename = "metallicRoughnessTexture", default)]
    metallic_roughness_texture: Option<TextureInfoDef>,
}

#[derive(Clone, Deserialize)]
struct TextureInfoDef {
    index: usize,
    #[serde(rename = "texCoord", default)]
    tex_coord: Option<usize>,
    #[serde(default)]
    extensions: Option<TextureInfoExtensionsDef>,
}

#[derive(Clone, Deserialize)]
struct TextureInfoExtensionsDef {
    #[serde(rename = "KHR_texture_transform", default)]
    texture_transform: Option<TextureTransformDef>,
}

#[derive(Clone, Deserialize)]
struct TextureTransformDef {
    #[serde(default)]
    offset: Option<[f32; 2]>,
    #[serde(default)]
    rotation: Option<f32>,
    #[serde(default)]
    scale: Option<[f32; 2]>,
    #[serde(rename = "texCoord", default)]
    tex_coord: Option<usize>,
}

#[derive(Deserialize)]
struct TextureDef {
    #[serde(default)]
    source: Option<usize>,
    #[serde(default)]
    sampler: Option<usize>,
}

#[derive(Deserialize)]
struct SamplerDef {
    #[serde(rename = "magFilter", default)]
    mag_filter: Option<u32>,
    #[serde(rename = "minFilter", default)]
    min_filter: Option<u32>,
    #[serde(rename = "wrapS", default)]
    wrap_s: Option<u32>,
    #[serde(rename = "wrapT", default)]
    wrap_t: Option<u32>,
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
    skin: Option<usize>,
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
    #[serde(default)]
    camera: Option<usize>,
    #[serde(default)]
    extensions: Option<NodeExtensionsDef>,
}

#[derive(Deserialize)]
struct NodeExtensionsDef {
    #[serde(rename = "KHR_lights_punctual", default)]
    lights: Option<NodeLightRefDef>,
}

#[derive(Deserialize)]
struct NodeLightRefDef {
    light: usize,
}

#[derive(Deserialize)]
struct RootExtensionsDef {
    #[serde(rename = "KHR_lights_punctual", default)]
    lights: Option<RootLightsDef>,
}

#[derive(Deserialize)]
struct RootLightsDef {
    #[serde(default)]
    lights: Vec<PunctualLightDef>,
}

#[derive(Deserialize)]
struct PunctualLightDef {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    color: Option<[f32; 3]>,
    #[serde(default)]
    intensity: Option<f32>,
}

#[derive(Deserialize)]
struct CameraDef {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    perspective: Option<PerspectiveCameraDef>,
    #[serde(default)]
    orthographic: Option<OrthographicCameraDef>,
}

#[derive(Deserialize)]
struct PerspectiveCameraDef {
    yfov: f32,
    znear: f32,
    #[serde(default)]
    zfar: Option<f32>,
}

#[derive(Deserialize)]
struct OrthographicCameraDef {
    #[serde(rename = "xmag")]
    xmag: f32,
    ymag: f32,
    znear: f32,
    zfar: f32,
}

#[derive(Deserialize)]
struct SkinDef {
    #[serde(rename = "inverseBindMatrices", default)]
    inverse_bind_matrices: Option<usize>,
    #[serde(default)]
    joints: Vec<usize>,
    #[serde(default)]
    skeleton: Option<usize>,
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
}

#[derive(Clone)]
struct LoadedPrimitive {
    mesh: Mesh,
    material: Material,
    skinning: Option<PrimitiveSkinningData>,
}

#[derive(Clone)]
struct PrimitiveSkinningData {
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
}

#[derive(Clone)]
struct LoadedSkin {
    joints: Vec<usize>,
    inverse_bind_matrices: Vec<Mat4>,
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
    let accessor = root
        .accessors
        .get(accessor_index)
        .ok_or(GltfError::InvalidIndex)?;
    if accessor.sparse.is_some() {
        return Err(GltfError::UnsupportedAccessor);
    }
    let view_index = accessor.buffer_view.ok_or(GltfError::UnsupportedAccessor)?;
    let view = root
        .buffer_views
        .get(view_index)
        .ok_or(GltfError::InvalidIndex)?;
    let data = buffers
        .get(view.buffer)
        .ok_or(GltfError::InvalidIndex)?
        .as_slice();

    let components = accessor_components(&accessor.kind).ok_or(GltfError::UnsupportedAccessor)?;
    let component_size =
        component_size(accessor.component_type).ok_or(GltfError::UnsupportedAccessor)?;
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
    if layout.accessor.kind != "VEC2" {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    match layout.accessor.component_type {
        5126 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(Vec2::new(read_f32_le(&e[0..4]), read_f32_le(&e[4..8])));
            }
        }
        5121 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(Vec2::new(f32::from(e[0]) / 255.0, f32::from(e[1]) / 255.0));
            }
        }
        5123 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push(Vec2::new(
                    f32::from(read_u16_le(&e[0..2])) / 65535.0,
                    f32::from(read_u16_le(&e[2..4])) / 65535.0,
                ));
            }
        }
        _ => return Err(GltfError::UnsupportedAccessor),
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

fn read_accessor_color4(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<Vec4>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    let comps = match layout.accessor.kind.as_str() {
        "VEC3" => 3usize,
        "VEC4" => 4usize,
        _ => return Err(GltfError::UnsupportedAccessor),
    };
    let mut out = Vec::with_capacity(layout.count);
    match layout.accessor.component_type {
        5126 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                let x = read_f32_le(&e[0..4]);
                let y = read_f32_le(&e[4..8]);
                let z = read_f32_le(&e[8..12]);
                let w = if comps == 4 {
                    read_f32_le(&e[12..16])
                } else {
                    1.0
                };
                out.push(Vec4::new(x, y, z, w));
            }
        }
        5121 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                let x = f32::from(e[0]) / 255.0;
                let y = f32::from(e[1]) / 255.0;
                let z = f32::from(e[2]) / 255.0;
                let w = if comps == 4 {
                    f32::from(e[3]) / 255.0
                } else {
                    1.0
                };
                out.push(Vec4::new(x, y, z, w));
            }
        }
        5123 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                let x = f32::from(read_u16_le(&e[0..2])) / 65535.0;
                let y = f32::from(read_u16_le(&e[2..4])) / 65535.0;
                let z = f32::from(read_u16_le(&e[4..6])) / 65535.0;
                let w = if comps == 4 {
                    f32::from(read_u16_le(&e[6..8])) / 65535.0
                } else {
                    1.0
                };
                out.push(Vec4::new(x, y, z, w));
            }
        }
        _ => return Err(GltfError::UnsupportedAccessor),
    }
    Ok(out)
}

fn read_accessor_mat4(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<Mat4>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "MAT4" || layout.accessor.component_type != 5126 {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    for i in 0..layout.count {
        let e = accessor_elem(&layout, i)?;
        let mut cols = [0.0f32; 16];
        for j in 0..16 {
            let off = j * 4;
            cols[j] = read_f32_le(&e[off..(off + 4)]);
        }
        out.push(Mat4::from_cols_array(&cols));
    }
    Ok(out)
}

fn read_accessor_joint_indices(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<[u16; 4]>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "VEC4" {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    match layout.accessor.component_type {
        5121 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push([
                    u16::from(e[0]),
                    u16::from(e[1]),
                    u16::from(e[2]),
                    u16::from(e[3]),
                ]);
            }
        }
        5123 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push([
                    read_u16_le(&e[0..2]),
                    read_u16_le(&e[2..4]),
                    read_u16_le(&e[4..6]),
                    read_u16_le(&e[6..8]),
                ]);
            }
        }
        _ => return Err(GltfError::UnsupportedAccessor),
    }
    Ok(out)
}

fn read_accessor_joint_weights(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    accessor_index: usize,
) -> Result<Vec<[f32; 4]>, GltfError> {
    let layout = accessor_layout(root, buffers, accessor_index)?;
    if layout.accessor.kind != "VEC4" {
        return Err(GltfError::UnsupportedAccessor);
    }
    let mut out = Vec::with_capacity(layout.count);
    match layout.accessor.component_type {
        5126 => {
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push([
                    read_f32_le(&e[0..4]),
                    read_f32_le(&e[4..8]),
                    read_f32_le(&e[8..12]),
                    read_f32_le(&e[12..16]),
                ]);
            }
        }
        5121 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push([
                    f32::from(e[0]) / 255.0,
                    f32::from(e[1]) / 255.0,
                    f32::from(e[2]) / 255.0,
                    f32::from(e[3]) / 255.0,
                ]);
            }
        }
        5123 => {
            if !layout.accessor.normalized {
                return Err(GltfError::UnsupportedAccessor);
            }
            for i in 0..layout.count {
                let e = accessor_elem(&layout, i)?;
                out.push([
                    f32::from(read_u16_le(&e[0..2])) / 65535.0,
                    f32::from(read_u16_le(&e[2..4])) / 65535.0,
                    f32::from(read_u16_le(&e[4..6])) / 65535.0,
                    f32::from(read_u16_le(&e[6..8])) / 65535.0,
                ]);
            }
        }
        _ => return Err(GltfError::UnsupportedAccessor),
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
    required_uv_sets: &[usize],
) -> Result<Mesh, GltfError> {
    if let Some(mode) = primitive.mode {
        if mode != 4 {
            return Err(GltfError::UnsupportedPrimitiveMode);
        }
    }

    let pos_accessor = *primitive
        .attributes
        .get("POSITION")
        .ok_or(GltfError::MissingAttribute)?;
    let positions = read_accessor_vec3(root, buffers, pos_accessor)?;

    let normals = if let Some(n) = primitive.attributes.get("NORMAL") {
        read_accessor_vec3(root, buffers, *n)?
    } else {
        Vec::new()
    };

    let uvs = if let Some(uv) = primitive.attributes.get("TEXCOORD_0") {
        read_accessor_vec2(root, buffers, *uv)?
    } else {
        Vec::new()
    };
    let uvs1 = if let Some(uv) = primitive.attributes.get("TEXCOORD_1") {
        read_accessor_vec2(root, buffers, *uv)?
    } else {
        Vec::new()
    };
    for &set in required_uv_sets {
        match set {
            0 if uvs.is_empty() => return Err(GltfError::MissingAttribute),
            1 if uvs1.is_empty() => return Err(GltfError::MissingAttribute),
            0 | 1 => {}
            _ => return Err(GltfError::MissingAttribute),
        }
    }

    let colors = if let Some(c0) = primitive.attributes.get("COLOR_0") {
        read_accessor_color4(root, buffers, *c0)?
    } else {
        Vec::new()
    };
    let tangents = if let Some(tg) = primitive.attributes.get("TANGENT") {
        read_accessor_vec4(root, buffers, *tg)?
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
    mesh.uvs1 = uvs1;
    mesh.colors = colors;
    mesh.tangents = tangents;
    mesh.indices = index_data
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    if !mesh.colors.is_empty() && mesh.colors.len() != mesh.positions.len() {
        return Err(GltfError::UnsupportedAccessor);
    }
    if !mesh.tangents.is_empty() && mesh.tangents.len() != mesh.positions.len() {
        return Err(GltfError::UnsupportedAccessor);
    }

    if mesh.normals.is_empty() {
        mesh.ensure_normals();
    }

    Ok(mesh)
}

fn primitive_skinning_data(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    primitive: &PrimitiveDef,
    vertex_count: usize,
) -> Result<Option<PrimitiveSkinningData>, GltfError> {
    let joints_accessor = primitive.attributes.get("JOINTS_0").copied();
    let weights_accessor = primitive.attributes.get("WEIGHTS_0").copied();

    let (Some(joints_ix), Some(weights_ix)) = (joints_accessor, weights_accessor) else {
        if joints_accessor.is_some() || weights_accessor.is_some() {
            return Err(GltfError::MissingAttribute);
        }
        return Ok(None);
    };

    let joints = read_accessor_joint_indices(root, buffers, joints_ix)?;
    let weights = read_accessor_joint_weights(root, buffers, weights_ix)?;
    if joints.len() != vertex_count || weights.len() != vertex_count {
        return Err(GltfError::InvalidAnimation);
    }

    Ok(Some(PrimitiveSkinningData { joints, weights }))
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

fn apply_world_to_mesh(mesh: &mut Mesh, world: Mat4) -> Result<(), GltfError> {
    if !world.is_finite() {
        return Err(GltfError::InvalidTransform);
    }
    let normal_m = Mat3::from_mat4(world).inverse().transpose();
    let tangent_m = if normal_m.is_finite() {
        normal_m
    } else {
        Mat3::from_mat4(world)
    };
    for p in &mut mesh.positions {
        *p = (world * p.extend(1.0)).truncate();
    }
    if !mesh.normals.is_empty() {
        for n in &mut mesh.normals {
            let nn = if normal_m.is_finite() {
                (normal_m * *n).normalize_or_zero()
            } else {
                (Mat3::from_mat4(world) * *n).normalize_or_zero()
            };
            *n = nn;
        }
    }
    if !mesh.tangents.is_empty() {
        for t in &mut mesh.tangents {
            let tt = (tangent_m * t.truncate()).normalize_or_zero();
            let handedness = if t.w < 0.0 { -1.0 } else { 1.0 };
            *t = Vec4::new(tt.x, tt.y, tt.z, handedness);
        }
    }
    Ok(())
}

fn default_material_binding() -> MaterialBinding {
    let mut material = Material::default();
    material.kd = Vec3::ONE;
    material.alpha = 1.0;
    material.alpha_mode = AlphaMode::Opaque;
    material.alpha_cutoff = 0.5;
    material.double_sided = false;
    material.map_kd = None;
    material.map_kd_path = None;
    material.map_kd_texcoord_set = 0;
    material.map_kd_uv_transform = Mat3::IDENTITY;
    material.map_normal = None;
    material.map_normal_texcoord_set = 0;
    material.map_normal_uv_transform = Mat3::IDENTITY;
    material.map_normal_scale = 1.0;
    material.map_occlusion = None;
    material.map_occlusion_texcoord_set = 0;
    material.map_occlusion_uv_transform = Mat3::IDENTITY;
    material.map_occlusion_strength = 1.0;
    material.map_emissive = None;
    material.map_emissive_texcoord_set = 0;
    material.map_emissive_uv_transform = Mat3::IDENTITY;
    material.map_metallic_roughness = None;
    material.map_metallic_roughness_texcoord_set = 0;
    material.map_metallic_roughness_uv_transform = Mat3::IDENTITY;
    material.metallic = 1.0;
    material.roughness = 1.0;
    material.pbr_metallic_roughness = true;
    MaterialBinding { material }
}

fn parse_alpha_mode(s: Option<&str>) -> AlphaMode {
    match s.unwrap_or("OPAQUE") {
        "MASK" => AlphaMode::Mask,
        "BLEND" => AlphaMode::Blend,
        _ => AlphaMode::Opaque,
    }
}

fn update_material_spec_from_metal_roughness(material: &mut Material) {
    let metallic = material.metallic.clamp(0.0, 1.0);
    let roughness = material.roughness.clamp(0.045, 1.0);
    material.ks = Vec3::splat(0.04).lerp(material.kd, metallic);
    material.ns = ((2.0 / (roughness * roughness)) - 2.0).clamp(1.0, 1024.0);
}

fn texture_uv_transform(tex_info: &TextureInfoDef) -> (usize, Mat3) {
    let mut texcoord = tex_info.tex_coord.unwrap_or(0);
    let mut offset = Vec2::ZERO;
    let mut scale = Vec2::ONE;
    let mut rotation = 0.0f32;
    if let Some(ext) = tex_info
        .extensions
        .as_ref()
        .and_then(|e| e.texture_transform.as_ref())
    {
        if let Some(uv_set) = ext.tex_coord {
            texcoord = uv_set;
        }
        if let Some(v) = ext.offset {
            offset = Vec2::new(v[0], v[1]);
        }
        if let Some(v) = ext.scale {
            scale = Vec2::new(v[0], v[1]);
        }
        if let Some(v) = ext.rotation {
            rotation = v;
        }
    }

    let c = rotation.cos();
    let s = rotation.sin();
    let transform = Mat3::from_cols(
        Vec3::new(scale.x * c, scale.x * s, 0.0),
        Vec3::new(-scale.y * s, scale.y * c, 0.0),
        Vec3::new(offset.x, offset.y, 1.0),
    );
    (texcoord, transform)
}

fn texture_uv_transform_normal(tex_info: &NormalTextureInfoDef) -> (usize, Mat3) {
    let as_info = TextureInfoDef {
        index: tex_info.index,
        tex_coord: tex_info.tex_coord,
        extensions: tex_info.extensions.clone(),
    };
    texture_uv_transform(&as_info)
}

fn texture_uv_transform_occlusion(tex_info: &OcclusionTextureInfoDef) -> (usize, Mat3) {
    let as_info = TextureInfoDef {
        index: tex_info.index,
        tex_coord: tex_info.tex_coord,
        extensions: tex_info.extensions.clone(),
    };
    texture_uv_transform(&as_info)
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

fn linearize_srgb_texture(tex: &mut Texture) {
    for px in tex.rgba8.chunks_exact_mut(4) {
        px[0] = srgb_to_linear_u8(px[0]);
        px[1] = srgb_to_linear_u8(px[1]);
        px[2] = srgb_to_linear_u8(px[2]);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextureColorSpace {
    Linear,
    Srgb,
}

fn view_bytes<'a>(
    root: &'a GltfRoot,
    buffers: &'a [Vec<u8>],
    view_index: usize,
) -> Result<&'a [u8], GltfError> {
    let view = root
        .buffer_views
        .get(view_index)
        .ok_or(GltfError::InvalidIndex)?;
    let data = buffers
        .get(view.buffer)
        .ok_or(GltfError::InvalidIndex)?
        .as_slice();
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
    let image = root
        .images
        .get(image_index)
        .ok_or(GltfError::InvalidIndex)?;

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
        TextureIoError::Io | TextureIoError::Decode | TextureIoError::Invalid => {
            GltfError::TextureIo
        }
    }
}

fn parse_wrap_mode(v: Option<u32>) -> TextureWrapMode {
    match v.unwrap_or(10_497) {
        33_071 => TextureWrapMode::ClampToEdge,
        33_648 => TextureWrapMode::MirroredRepeat,
        _ => TextureWrapMode::Repeat,
    }
}

fn parse_mag_filter(v: Option<u32>) -> TextureMagFilter {
    match v.unwrap_or(9_729) {
        9_728 => TextureMagFilter::Nearest,
        _ => TextureMagFilter::Linear,
    }
}

fn parse_min_filter(v: Option<u32>) -> TextureMinFilter {
    match v.unwrap_or(9_987) {
        9_728 => TextureMinFilter::Nearest,
        9_729 => TextureMinFilter::Linear,
        9_984 => TextureMinFilter::NearestMipmapNearest,
        9_985 => TextureMinFilter::LinearMipmapNearest,
        9_986 => TextureMinFilter::NearestMipmapLinear,
        _ => TextureMinFilter::LinearMipmapLinear,
    }
}

fn texture_sampler_for_texture(
    root: &GltfRoot,
    texture_index: usize,
) -> Result<TextureSampler, GltfError> {
    let tex_def = root
        .textures
        .get(texture_index)
        .ok_or(GltfError::InvalidIndex)?;

    let sampler_def = match tex_def.sampler {
        Some(s) => Some(root.samplers.get(s).ok_or(GltfError::InvalidIndex)?),
        None => None,
    };

    let sampler = TextureSampler {
        wrap_s: parse_wrap_mode(sampler_def.and_then(|s| s.wrap_s)),
        wrap_t: parse_wrap_mode(sampler_def.and_then(|s| s.wrap_t)),
        min_filter: parse_min_filter(sampler_def.and_then(|s| s.min_filter)),
        mag_filter: parse_mag_filter(sampler_def.and_then(|s| s.mag_filter)),
    };
    Ok(sampler)
}

fn ensure_texture_handle(
    root: &GltfRoot,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    scene: &mut Scene,
    texture_cache: &mut [Option<TextureHandle>],
    texture_index: usize,
    color_space: TextureColorSpace,
) -> Result<TextureHandle, GltfError> {
    let Some(slot) = texture_cache.get_mut(texture_index) else {
        return Err(GltfError::InvalidIndex);
    };

    if let Some(handle) = *slot {
        return Ok(handle);
    }

    let tex_def = root
        .textures
        .get(texture_index)
        .ok_or(GltfError::InvalidIndex)?;
    let sampler = texture_sampler_for_texture(root, texture_index)?;
    let image_index = tex_def.source.ok_or(GltfError::MissingImageSource)?;
    let bytes = load_image_bytes(root, buffers, base_dir, image_index)?;
    let mut texture = load_texture_rgba8_from_bytes_raw(&bytes).map_err(map_texture_io_error)?;
    if color_space == TextureColorSpace::Srgb {
        linearize_srgb_texture(&mut texture);
    }
    texture.rebuild_mipmaps();
    texture.sampler = sampler;
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
    let mut texture_cache_linear: Vec<Option<TextureHandle>> = vec![None; root.textures.len()];
    let mut texture_cache_srgb: Vec<Option<TextureHandle>> = vec![None; root.textures.len()];
    let mut out = Vec::with_capacity(root.materials.len());

    for material in &root.materials {
        let mut binding = default_material_binding();
        binding.material.alpha_mode = parse_alpha_mode(material.alpha_mode.as_deref());
        binding.material.alpha_cutoff = material.alpha_cutoff.unwrap_or(0.5).clamp(0.0, 1.0);
        binding.material.double_sided = material.double_sided.unwrap_or(false);
        if let Some(ke) = material.emissive_factor {
            binding.material.ke = Vec3::new(
                ke[0].clamp(0.0, 1.0),
                ke[1].clamp(0.0, 1.0),
                ke[2].clamp(0.0, 1.0),
            );
        }
        binding.material.metallic = 1.0;
        binding.material.roughness = 1.0;

        if let Some(pbr) = &material.pbr_metallic_roughness {
            let factor = pbr.base_color_factor.unwrap_or([1.0, 1.0, 1.0, 1.0]);
            binding.material.kd = Vec3::new(
                factor[0].clamp(0.0, 1.0),
                factor[1].clamp(0.0, 1.0),
                factor[2].clamp(0.0, 1.0),
            );
            binding.material.alpha = factor[3].clamp(0.0, 1.0);

            binding.material.metallic = pbr.metallic_factor.unwrap_or(1.0).clamp(0.0, 1.0);
            binding.material.roughness = pbr.roughness_factor.unwrap_or(1.0).clamp(0.045, 1.0);
            update_material_spec_from_metal_roughness(&mut binding.material);

            if let Some(tex_info) = &pbr.base_color_texture {
                let handle = ensure_texture_handle(
                    root,
                    buffers,
                    base_dir,
                    scene,
                    &mut texture_cache_srgb,
                    tex_info.index,
                    TextureColorSpace::Srgb,
                )?;
                binding.material.map_kd = Some(handle);
                let (texcoord_set, uv_transform) = texture_uv_transform(tex_info);
                binding.material.map_kd_texcoord_set = texcoord_set;
                binding.material.map_kd_uv_transform = uv_transform;
            }

            if let Some(tex_info) = &pbr.metallic_roughness_texture {
                let handle = ensure_texture_handle(
                    root,
                    buffers,
                    base_dir,
                    scene,
                    &mut texture_cache_linear,
                    tex_info.index,
                    TextureColorSpace::Linear,
                )?;
                binding.material.map_metallic_roughness = Some(handle);
                let (texcoord_set, uv_transform) = texture_uv_transform(tex_info);
                binding.material.map_metallic_roughness_texcoord_set = texcoord_set;
                binding.material.map_metallic_roughness_uv_transform = uv_transform;
            }
        }

        if let Some(tex_info) = &material.normal_texture {
            let handle = ensure_texture_handle(
                root,
                buffers,
                base_dir,
                scene,
                &mut texture_cache_linear,
                tex_info.index,
                TextureColorSpace::Linear,
            )?;
            binding.material.map_normal = Some(handle);
            let (texcoord_set, uv_transform) = texture_uv_transform_normal(tex_info);
            binding.material.map_normal_texcoord_set = texcoord_set;
            binding.material.map_normal_uv_transform = uv_transform;
            binding.material.map_normal_scale = tex_info.scale.unwrap_or(1.0).max(0.0);
        }

        if let Some(tex_info) = &material.occlusion_texture {
            let handle = ensure_texture_handle(
                root,
                buffers,
                base_dir,
                scene,
                &mut texture_cache_linear,
                tex_info.index,
                TextureColorSpace::Linear,
            )?;
            binding.material.map_occlusion = Some(handle);
            let (texcoord_set, uv_transform) = texture_uv_transform_occlusion(tex_info);
            binding.material.map_occlusion_texcoord_set = texcoord_set;
            binding.material.map_occlusion_uv_transform = uv_transform;
            binding.material.map_occlusion_strength = tex_info.strength.unwrap_or(1.0).clamp(0.0, 1.0);
        }

        if let Some(tex_info) = &material.emissive_texture {
            let handle = ensure_texture_handle(
                root,
                buffers,
                base_dir,
                scene,
                &mut texture_cache_srgb,
                tex_info.index,
                TextureColorSpace::Srgb,
            )?;
            binding.material.map_emissive = Some(handle);
            let (texcoord_set, uv_transform) = texture_uv_transform(tex_info);
            binding.material.map_emissive_texcoord_set = texcoord_set;
            binding.material.map_emissive_uv_transform = uv_transform;
        }

        out.push(binding);
    }

    Ok(out)
}

fn load_skins(root: &GltfRoot, buffers: &[Vec<u8>]) -> Result<Vec<LoadedSkin>, GltfError> {
    let mut out = Vec::with_capacity(root.skins.len());
    for skin in &root.skins {
        if skin.joints.is_empty() {
            return Err(GltfError::InvalidAnimation);
        }
        if let Some(skeleton) = skin.skeleton {
            if skeleton >= root.nodes.len() {
                return Err(GltfError::InvalidIndex);
            }
        }
        for &joint in &skin.joints {
            if joint >= root.nodes.len() {
                return Err(GltfError::InvalidIndex);
            }
        }

        let inverse_bind_matrices = if let Some(accessor_index) = skin.inverse_bind_matrices {
            let mats = read_accessor_mat4(root, buffers, accessor_index)?;
            if mats.len() != skin.joints.len() {
                return Err(GltfError::InvalidAnimation);
            }
            mats
        } else {
            vec![Mat4::IDENTITY; skin.joints.len()]
        };

        out.push(LoadedSkin {
            joints: skin.joints.clone(),
            inverse_bind_matrices,
        });
    }
    Ok(out)
}

fn node_parent_map(root: &GltfRoot) -> Result<Vec<Option<usize>>, GltfError> {
    let mut parent: Vec<Option<usize>> = vec![None; root.nodes.len()];
    for (node_index, node) in root.nodes.iter().enumerate() {
        for &child in &node.children {
            if child >= root.nodes.len() {
                return Err(GltfError::InvalidIndex);
            }
            if parent[child].is_some() {
                return Err(GltfError::InvalidNodeGraph);
            }
            parent[child] = Some(node_index);
        }
    }
    Ok(parent)
}

fn implicit_scene_root_nodes(root: &GltfRoot) -> Result<Vec<usize>, GltfError> {
    let parent = node_parent_map(root)?;
    Ok((0..parent.len()).filter(|&i| parent[i].is_none()).collect())
}

fn compute_node_world_matrices(root: &GltfRoot, locals: &[Mat4]) -> Result<Vec<Mat4>, GltfError> {
    if locals.len() != root.nodes.len() {
        return Err(GltfError::InvalidIndex);
    }

    let parent = node_parent_map(root)?;

    let mut world = vec![Mat4::IDENTITY; root.nodes.len()];
    let mut done = vec![false; root.nodes.len()];
    let mut visiting = vec![false; root.nodes.len()];

    fn resolve_world(
        node_index: usize,
        parent: &[Option<usize>],
        locals: &[Mat4],
        world: &mut [Mat4],
        done: &mut [bool],
        visiting: &mut [bool],
    ) -> Result<Mat4, GltfError> {
        if done[node_index] {
            return Ok(world[node_index]);
        }
        if visiting[node_index] {
            return Err(GltfError::InvalidNodeGraph);
        }

        visiting[node_index] = true;
        let node_world = if let Some(parent_index) = parent[node_index] {
            resolve_world(parent_index, parent, locals, world, done, visiting)? * locals[node_index]
        } else {
            locals[node_index]
        };
        if !node_world.is_finite() {
            return Err(GltfError::InvalidTransform);
        }
        world[node_index] = node_world;
        done[node_index] = true;
        visiting[node_index] = false;
        Ok(node_world)
    }

    for i in 0..root.nodes.len() {
        let _ = resolve_world(i, &parent, locals, &mut world, &mut done, &mut visiting)?;
    }

    Ok(world)
}

fn skin_joint_matrices(
    node_index: usize,
    skin: &LoadedSkin,
    world_matrices: &[Mat4],
) -> Result<Vec<Mat4>, GltfError> {
    let node_world = *world_matrices
        .get(node_index)
        .ok_or(GltfError::InvalidIndex)?;
    let inv_node_world = node_world.inverse();
    if !inv_node_world.is_finite() {
        return Err(GltfError::InvalidTransform);
    }

    let mut out = Vec::with_capacity(skin.joints.len());
    for (joint_slot, &joint_node) in skin.joints.iter().enumerate() {
        let joint_world = *world_matrices
            .get(joint_node)
            .ok_or(GltfError::InvalidIndex)?;
        let ibm = *skin
            .inverse_bind_matrices
            .get(joint_slot)
            .ok_or(GltfError::InvalidAnimation)?;
        out.push(inv_node_world * joint_world * ibm);
    }
    Ok(out)
}

fn skin_mesh_in_place(
    mesh: &mut Mesh,
    skinning: &PrimitiveSkinningData,
    joint_matrices: &[Mat4],
) -> Result<(), GltfError> {
    if mesh.positions.len() != skinning.joints.len()
        || mesh.positions.len() != skinning.weights.len()
    {
        return Err(GltfError::InvalidAnimation);
    }
    if mesh.normals.len() != mesh.positions.len() {
        return Err(GltfError::InvalidAnimation);
    }
    if !mesh.tangents.is_empty() && mesh.tangents.len() != mesh.positions.len() {
        return Err(GltfError::InvalidAnimation);
    }

    let mut normal_mats = Vec::with_capacity(joint_matrices.len());
    for m in joint_matrices {
        let lin = Mat3::from_mat4(*m);
        let inv_t = lin.inverse().transpose();
        if inv_t.is_finite() {
            normal_mats.push(inv_t);
        } else {
            normal_mats.push(lin);
        }
    }

    for i in 0..mesh.positions.len() {
        let joints = skinning.joints[i];
        let raw_weights = skinning.weights[i];
        let mut weights = [
            raw_weights[0].max(0.0),
            raw_weights[1].max(0.0),
            raw_weights[2].max(0.0),
            raw_weights[3].max(0.0),
        ];
        let sum = weights[0] + weights[1] + weights[2] + weights[3];
        if sum <= 0.0 {
            continue;
        }
        let inv_sum = 1.0 / sum;
        for w in &mut weights {
            *w *= inv_sum;
        }

        let src_pos = mesh.positions[i].extend(1.0);
        let src_nrm = mesh.normals[i];
        let src_tangent = if mesh.tangents.is_empty() {
            None
        } else {
            Some(mesh.tangents[i])
        };

        let mut out_pos = Vec4::ZERO;
        let mut out_nrm = Vec3::ZERO;
        let mut out_tangent = Vec3::ZERO;
        for k in 0..4 {
            let w = weights[k];
            if w <= 0.0 {
                continue;
            }
            let joint_slot = usize::from(joints[k]);
            let jm = *joint_matrices
                .get(joint_slot)
                .ok_or(GltfError::InvalidIndex)?;
            let nm = *normal_mats.get(joint_slot).ok_or(GltfError::InvalidIndex)?;
            out_pos += (jm * src_pos) * w;
            out_nrm += (nm * src_nrm) * w;
            if let Some(t) = src_tangent {
                out_tangent += (nm * t.truncate()) * w;
            }
        }

        mesh.positions[i] = out_pos.truncate();
        let dst_normal = if out_nrm.length_squared() > 0.0 {
            out_nrm.normalize()
        } else {
            src_nrm
        };
        mesh.normals[i] = dst_normal;
        if let Some(src_tangent) = src_tangent {
            let mut t = if out_tangent.length_squared() > 0.0 {
                out_tangent.normalize()
            } else {
                src_tangent.truncate().normalize_or_zero()
            };
            t = (t - dst_normal * dst_normal.dot(t)).normalize_or_zero();
            if t.length_squared() <= 1e-12 {
                let fallback_axis = if dst_normal.z.abs() < 0.999 {
                    Vec3::Z
                } else {
                    Vec3::Y
                };
                t = dst_normal.cross(fallback_axis).normalize_or_zero();
            }
            let handedness = if src_tangent.w < 0.0 { -1.0 } else { 1.0 };
            mesh.tangents[i] = Vec4::new(t.x, t.y, t.z, handedness);
        }
    }

    Ok(())
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
            let need = times
                .len()
                .checked_mul(3)
                .ok_or(GltfError::InvalidAnimation)?;
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
            let need = times
                .len()
                .checked_mul(3)
                .ok_or(GltfError::InvalidAnimation)?;
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
    let anim = root
        .animations
        .get(animation_index)
        .ok_or(GltfError::InvalidIndex)?;
    let time = if time_seconds.is_finite() {
        time_seconds
    } else {
        0.0
    };

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
        let sampler = anim
            .samplers
            .get(channel.sampler)
            .ok_or(GltfError::InvalidIndex)?;
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
                scales[node_index] =
                    sample_vec3_channel(&input_times, &output, interpolation, time)?;
            }
            "rotation" => {
                let output = read_accessor_vec4(root, buffers, sampler.output)?;
                rotations[node_index] =
                    sample_quat_channel(&input_times, &output, interpolation, time)?;
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

fn camera_projection_from_def(def: &CameraDef) -> Option<Projection> {
    match def.kind.as_str() {
        "perspective" => {
            let p = def.perspective.as_ref()?;
            if !(p.yfov.is_finite() && p.znear.is_finite() && p.znear > 0.0) {
                return None;
            }
            let far = p.zfar.unwrap_or(1000.0);
            if !(far.is_finite() && far > p.znear) {
                return None;
            }
            Some(Projection::Perspective {
                fov_y_radians: p.yfov,
                near: p.znear,
                far,
            })
        }
        "orthographic" => {
            let o = def.orthographic.as_ref()?;
            if !(o.xmag.is_finite()
                && o.xmag > 0.0
                && o.ymag.is_finite()
                && o.ymag > 0.0
                && o.znear.is_finite()
                && o.zfar.is_finite()
                && o.zfar > o.znear)
            {
                return None;
            }
            Some(Projection::Orthographic {
                half_width: Some(o.xmag),
                half_height: o.ymag,
                near: o.znear,
                far: o.zfar,
            })
        }
        _ => None,
    }
}

fn apply_cameras_and_lights(
    scene: &mut Scene,
    root: &GltfRoot,
    world_matrices: &[Mat4],
    selected_scene_nodes: Option<&[usize]>,
) {
    let candidate_nodes: Vec<usize> = if let Some(nodes) = selected_scene_nodes {
        let mut out = Vec::new();
        let mut stack: Vec<usize> = nodes.to_vec();
        while let Some(n) = stack.pop() {
            if out.contains(&n) {
                continue;
            }
            out.push(n);
            if let Some(node) = root.nodes.get(n) {
                for &c in &node.children {
                    stack.push(c);
                }
            }
        }
        out
    } else {
        (0..root.nodes.len()).collect()
    };

    for &node_index in &candidate_nodes {
        let Some(node) = root.nodes.get(node_index) else {
            continue;
        };
        let Some(camera_index) = node.camera else {
            continue;
        };
        let Some(camera_def) = root.cameras.get(camera_index) else {
            continue;
        };
        let Some(projection) = camera_projection_from_def(camera_def) else {
            continue;
        };
        let Some(world) = world_matrices.get(node_index).copied() else {
            continue;
        };
        if let Ok(transform) = world_to_transform(world) {
            scene.camera = Camera::new(transform, projection);
            break;
        }
    }

    let Some(root_lights) = root
        .extensions
        .as_ref()
        .and_then(|e| e.lights.as_ref())
        .map(|l| &l.lights)
    else {
        return;
    };

    for &node_index in &candidate_nodes {
        let Some(node) = root.nodes.get(node_index) else {
            continue;
        };
        let Some(light_ref) = node
            .extensions
            .as_ref()
            .and_then(|e| e.lights.as_ref())
            .map(|r| r.light)
        else {
            continue;
        };
        let Some(light_def) = root_lights.get(light_ref) else {
            continue;
        };
        let Some(world) = world_matrices.get(node_index).copied() else {
            continue;
        };
        let color_arr = light_def.color.unwrap_or([1.0, 1.0, 1.0]);
        let color =
            Vec3::new(color_arr[0], color_arr[1], color_arr[2]).clamp(Vec3::ZERO, Vec3::ONE);
        let intensity = light_def.intensity.unwrap_or(1.0).max(0.0);

        match light_def.kind.as_str() {
            "directional" => {
                let dir = (world * Vec4::new(0.0, 0.0, -1.0, 0.0))
                    .truncate()
                    .normalize_or_zero();
                if dir.length_squared() > 0.0 {
                    scene.add_light(Light::directional(dir, color, intensity));
                }
            }
            "point" => {
                let pos = (world * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
                scene.add_light(Light::point(pos, color, intensity));
            }
            _ => {}
        }
    }
}

fn material_required_uv_sets(material: &Material) -> Vec<usize> {
    let mut out = Vec::new();
    if material.map_kd.is_some() {
        out.push(material.map_kd_texcoord_set);
    }
    if material.map_normal.is_some() {
        out.push(material.map_normal_texcoord_set);
    }
    if material.map_occlusion.is_some() {
        out.push(material.map_occlusion_texcoord_set);
    }
    if material.map_emissive.is_some() {
        out.push(material.map_emissive_texcoord_set);
    }
    if material.map_metallic_roughness.is_some() {
        out.push(material.map_metallic_roughness_texcoord_set);
    }
    out.sort_unstable();
    out.dedup();
    out
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
                materials
                    .get(material_index)
                    .ok_or(GltfError::InvalidIndex)?
            } else {
                &default_binding
            };

            let required_uv_sets = material_required_uv_sets(&binding.material);
            let mesh = primitive_to_mesh(root, buffers, p, &required_uv_sets)?;
            let skinning = primitive_skinning_data(root, buffers, p, mesh.positions.len())?;

            prims.push(LoadedPrimitive {
                mesh,
                material: binding.material.clone(),
                skinning,
            });
        }
        per_mesh_primitives.push(prims);
    }

    let locals = if let Some(values) = animated_locals {
        if values.len() != root.nodes.len() {
            return Err(GltfError::InvalidIndex);
        }
        values.to_vec()
    } else {
        let mut mats = Vec::with_capacity(root.nodes.len());
        for node in &root.nodes {
            mats.push(node_local_matrix(node)?);
        }
        mats
    };
    let world_matrices = compute_node_world_matrices(root, &locals)?;
    let loaded_skins = load_skins(root, buffers)?;

    let mut visiting = vec![false; root.nodes.len()];

    fn walk(
        node_index: usize,
        root: &GltfRoot,
        per_mesh_primitives: &[Vec<LoadedPrimitive>],
        world_matrices: &[Mat4],
        loaded_skins: &[LoadedSkin],
        locals: &[Mat4],
        parent_world: Mat4,
        visiting: &mut [bool],
        scene: &mut Scene,
    ) -> Result<(), GltfError> {
        let node = root.nodes.get(node_index).ok_or(GltfError::InvalidIndex)?;
        if visiting[node_index] {
            return Err(GltfError::InvalidNodeGraph);
        }
        visiting[node_index] = true;

        let local = *locals.get(node_index).ok_or(GltfError::InvalidIndex)?;
        let world = parent_world * local;

        if let Some(mesh_index) = node.mesh {
            let primitives = per_mesh_primitives
                .get(mesh_index)
                .ok_or(GltfError::InvalidIndex)?;

            if let Some(skin_index) = node.skin {
                let skin = loaded_skins
                    .get(skin_index)
                    .ok_or(GltfError::InvalidIndex)?;
                let joint_matrices = skin_joint_matrices(node_index, skin, world_matrices)?;
                for prim in primitives {
                    let skinning = prim.skinning.as_ref().ok_or(GltfError::MissingAttribute)?;
                    let mut mesh = prim.mesh.clone();
                    skin_mesh_in_place(&mut mesh, skinning, &joint_matrices)?;
                    apply_world_to_mesh(&mut mesh, world)?;
                    let _ = scene.add_object(mesh, Transform::IDENTITY, prim.material.clone());
                }
            } else {
                for prim in primitives {
                    let mut mesh = prim.mesh.clone();
                    apply_world_to_mesh(&mut mesh, world)?;
                    let _ = scene.add_object(mesh, Transform::IDENTITY, prim.material.clone());
                }
            }
        }

        for &child in &node.children {
            walk(
                child,
                root,
                per_mesh_primitives,
                world_matrices,
                loaded_skins,
                locals,
                world,
                visiting,
                scene,
            )?;
        }

        visiting[node_index] = false;
        Ok(())
    }

    if root.scenes.is_empty() {
        let root_nodes = implicit_scene_root_nodes(root)?;
        for &i in &root_nodes {
            walk(
                i,
                root,
                &per_mesh_primitives,
                &world_matrices,
                &loaded_skins,
                &locals,
                Mat4::IDENTITY,
                &mut visiting,
                &mut scene,
            )?;
        }
        apply_cameras_and_lights(&mut scene, root, &world_matrices, Some(&root_nodes));
        return Ok(scene);
    }

    let scene_index = root.scene.unwrap_or(0);
    let selected = root
        .scenes
        .get(scene_index)
        .ok_or(GltfError::InvalidIndex)?;
    for &node in &selected.nodes {
        walk(
            node,
            root,
            &per_mesh_primitives,
            &world_matrices,
            &loaded_skins,
            &locals,
            Mat4::IDENTITY,
            &mut visiting,
            &mut scene,
        )?;
    }
    apply_cameras_and_lights(&mut scene, root, &world_matrices, Some(&selected.nodes));
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
    parse_and_load(&src, Some(&base_dir), Some((animation_index, time_seconds)))
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
    use crate::texture::{TextureMagFilter, TextureMinFilter, TextureWrapMode};
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

    fn tiny_gltf_inline_no_scene() -> String {
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
]
}}"#
        )
    }

    fn tiny_gltf_ortho_camera_inline() -> String {
        r#"{
"asset":{"version":"2.0"},
"cameras":[{"type":"orthographic","orthographic":{"xmag":2.0,"ymag":1.0,"znear":0.1,"zfar":10.0}}],
"nodes":[{"camera":0}],
"scenes":[{"nodes":[0]}],
"scene":0
}"#
        .to_string()
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
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
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

    fn tiny_textured_gltf_inline_with_sampler() -> String {
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
            -0.25f32, 0.0, //
            1.25, 0.0, //
            1.25, 1.0, //
            -0.25, 1.0,
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
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
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
"samplers":[{{"wrapS":33071,"wrapT":33648,"magFilter":9728,"minFilter":9987}}],
"textures":[{{"sampler":0,"source":0}}],
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

    fn tiny_textured_gltf_inline_with_normalized_integer_uvs() -> String {
        let mut geom = Vec::new();

        let positions = [
            -0.5f32, -0.5, 0.0, //
            0.5, -0.5, 0.0, //
            0.0, 0.5, 0.0,
        ];
        for v in positions {
            geom.extend_from_slice(&v.to_le_bytes());
        }

        let uvs0 = [[0u8, 0u8], [255u8, 0u8], [0u8, 255u8]];
        for uv in uvs0 {
            geom.extend_from_slice(&uv);
        }

        let uvs1 = [[0u16, 0u16], [u16::MAX, 0u16], [0u16, u16::MAX]];
        for uv in uvs1 {
            geom.extend_from_slice(&uv[0].to_le_bytes());
            geom.extend_from_slice(&uv[1].to_le_bytes());
        }

        let indices = [0u16, 1u16, 2u16];
        for i in indices {
            geom.extend_from_slice(&i.to_le_bytes());
        }

        use base64::Engine as _;
        let geom_payload = base64::engine::general_purpose::STANDARD.encode(geom);

        format!(
            r#"{{
"asset":{{"version":"2.0"}},
"buffers":[{{"uri":"data:application/octet-stream;base64,{geom_payload}","byteLength":60}}],
"bufferViews":[
  {{"buffer":0,"byteOffset":0,"byteLength":36}},
  {{"buffer":0,"byteOffset":36,"byteLength":6}},
  {{"buffer":0,"byteOffset":42,"byteLength":12}},
  {{"buffer":0,"byteOffset":54,"byteLength":6}}
],
"accessors":[
  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
  {{"bufferView":1,"componentType":5121,"count":3,"type":"VEC2","normalized":true}},
  {{"bufferView":2,"componentType":5123,"count":3,"type":"VEC2","normalized":true}},
  {{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}}
],
"meshes":[{{"primitives":[{{
  "attributes":{{"POSITION":0,"TEXCOORD_0":1,"TEXCOORD_1":2}},
  "indices":3
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

    fn tiny_skinned_gltf_inline() -> String {
        let mut bytes = Vec::new();

        let positions = [
            0.0f32, 2.0, 0.0, //
            0.2, 2.0, 0.0, //
            0.0, 2.2, 0.0,
        ];
        for v in positions {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let joints = [
            [1u8, 0u8, 0u8, 0u8],
            [1u8, 0u8, 0u8, 0u8],
            [1u8, 0u8, 0u8, 0u8],
        ];
        for j in joints {
            bytes.extend_from_slice(&j);
        }

        let weights = [
            1.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ];
        for w in weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        let indices = [0u16, 1u16, 2u16];
        for i in indices {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        bytes.extend_from_slice(&[0u8, 0u8]);

        let ibm0 = Mat4::IDENTITY.to_cols_array();
        for v in ibm0 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let ibm1 = Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)).to_cols_array();
        for v in ibm1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let times = [0.0f32, 1.0f32];
        for t in times {
            bytes.extend_from_slice(&t.to_le_bytes());
        }

        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rotations = [
            0.0f32, 0.0, 0.0, 1.0, //
            0.0, 0.0, s, s,
        ];
        for r in rotations {
            bytes.extend_from_slice(&r.to_le_bytes());
        }

        use base64::Engine as _;
        let payload = base64::engine::general_purpose::STANDARD.encode(bytes);

        format!(
            r#"{{
"asset":{{"version":"2.0"}},
"buffers":[{{"uri":"data:application/octet-stream;base64,{payload}","byteLength":272}}],
"bufferViews":[
  {{"buffer":0,"byteOffset":0,"byteLength":36}},
  {{"buffer":0,"byteOffset":36,"byteLength":12}},
  {{"buffer":0,"byteOffset":48,"byteLength":48}},
  {{"buffer":0,"byteOffset":96,"byteLength":6}},
  {{"buffer":0,"byteOffset":104,"byteLength":128}},
  {{"buffer":0,"byteOffset":232,"byteLength":8}},
  {{"buffer":0,"byteOffset":240,"byteLength":32}}
],
"accessors":[
  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
  {{"bufferView":1,"componentType":5121,"count":3,"type":"VEC4"}},
  {{"bufferView":2,"componentType":5126,"count":3,"type":"VEC4"}},
  {{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}},
  {{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"}},
  {{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR"}},
  {{"bufferView":6,"componentType":5126,"count":2,"type":"VEC4"}}
],
"meshes":[{{"primitives":[{{
  "attributes":{{"POSITION":0,"JOINTS_0":1,"WEIGHTS_0":2}},
  "indices":3
}}]}}],
"skins":[{{"inverseBindMatrices":4,"joints":[0,1],"skeleton":0}}],
"nodes":[
  {{"children":[1,2]}},
  {{"translation":[0.0,1.0,0.0]}},
  {{"mesh":0,"skin":0}}
],
"animations":[{{
  "samplers":[{{"input":5,"output":6,"interpolation":"LINEAR"}}],
  "channels":[{{"sampler":0,"target":{{"node":1,"path":"rotation"}}}}]
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
            assert!(xf.translation.length() < 1e-6);
            let centroid = (mesh.positions[0] + mesh.positions[1] + mesh.positions[2]) / 3.0;
            assert!((centroid.x - 0.5).abs() < 1e-6);
        }
        assert!(saw);
    }

    #[test]
    fn gltf_inline_without_scene_uses_only_implicit_roots() {
        let gltf = tiny_gltf_inline_no_scene();
        let scene = load_gltf_str(&gltf).unwrap();
        assert_eq!(scene.object_count(), 1);

        let (mesh, _mat, xf) = scene.iter_objects().next().unwrap();
        assert!(xf.translation.length() < 1e-6);
        let centroid = (mesh.positions[0] + mesh.positions[1] + mesh.positions[2]) / 3.0;
        assert!((centroid.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gltf_orthographic_camera_respects_xmag() {
        let gltf = tiny_gltf_ortho_camera_inline();
        let scene = load_gltf_str(&gltf).unwrap();

        match scene.camera.projection {
            Projection::Orthographic {
                half_width,
                half_height,
                near,
                far,
            } => {
                assert_eq!(half_width, Some(2.0));
                assert!((half_height - 1.0).abs() < 1e-6);
                assert!((near - 0.1).abs() < 1e-6);
                assert!((far - 10.0).abs() < 1e-6);
            }
            _ => panic!("expected orthographic projection"),
        }

        let p = scene.camera.projection_matrix(5.0);
        assert!((p.x_axis.x - 0.5).abs() < 1e-6);
        assert!((p.y_axis.y - 1.0).abs() < 1e-6);
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
                half_width: None,
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

        const EXPECTED: u64 = 14_200_097_390_887_458_853;
        assert_eq!(img.hash64(), EXPECTED);
    }

    #[test]
    fn gltf_base_color_sampler_wrap_and_filter_are_applied() {
        let gltf = tiny_textured_gltf_inline_with_sampler();
        let scene = load_gltf_str(&gltf).unwrap();
        let (_mesh, mat, _xf) = scene.iter_objects().next().unwrap();
        let h = mat.map_kd.expect("expected baseColor texture");
        let tex = scene.texture(h).unwrap();
        assert_eq!(tex.sampler.wrap_s, TextureWrapMode::ClampToEdge);
        assert_eq!(tex.sampler.wrap_t, TextureWrapMode::MirroredRepeat);
        assert_eq!(tex.sampler.mag_filter, TextureMagFilter::Nearest);
        assert_eq!(tex.sampler.min_filter, TextureMinFilter::LinearMipmapLinear);
    }

    #[test]
    fn gltf_texcoord_accepts_normalized_unsigned_integer_accessors() {
        let gltf = tiny_textured_gltf_inline_with_normalized_integer_uvs();
        let scene = load_gltf_str(&gltf).unwrap();
        assert_eq!(scene.object_count(), 1);
        let (mesh, _mat, _xf) = scene.iter_objects().next().unwrap();

        assert_eq!(mesh.uvs.len(), 3);
        assert_eq!(mesh.uvs1.len(), 3);
        assert!((mesh.uvs[0].x - 0.0).abs() < 1e-6);
        assert!((mesh.uvs[0].y - 0.0).abs() < 1e-6);
        assert!((mesh.uvs[1].x - 1.0).abs() < 1e-6);
        assert!((mesh.uvs[2].y - 1.0).abs() < 1e-6);
        assert!((mesh.uvs1[1].x - 1.0).abs() < 1e-6);
        assert!((mesh.uvs1[2].y - 1.0).abs() < 1e-6);
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
                    half_width: None,
                    half_height: 1.0,
                    near: -10.0,
                    far: 10.0,
                },
            );
        }

        let mut x_a = None;
        for (mesh, _mat, xf) in scene_a.iter_objects() {
            assert!(xf.translation.length() < 1e-6);
            let mut c = Vec3::ZERO;
            for p in &mesh.positions {
                c += *p;
            }
            x_a = Some(c.x / mesh.positions.len() as f32);
        }
        let mut x_b = None;
        for (mesh, _mat, xf) in scene_b.iter_objects() {
            assert!(xf.translation.length() < 1e-6);
            let mut c = Vec3::ZERO;
            for p in &mesh.positions {
                c += *p;
            }
            x_b = Some(c.x / mesh.positions.len() as f32);
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

    #[test]
    fn gltf_skinning_math_matches_expected_vertices() {
        let gltf = tiny_skinned_gltf_inline();
        let scene_t0 = load_gltf_str_at_time(&gltf, 0, 0.0).unwrap();
        let scene_t1 = load_gltf_str_at_time(&gltf, 0, 1.0).unwrap();

        assert_eq!(scene_t0.object_count(), 1);
        assert_eq!(scene_t1.object_count(), 1);

        let (mesh_t0, _mat_t0, xf_t0) = scene_t0.iter_objects().next().unwrap();
        let (mesh_t1, _mat_t1, xf_t1) = scene_t1.iter_objects().next().unwrap();

        assert!(xf_t0.translation.length() < 1e-6);
        assert!(xf_t1.translation.length() < 1e-6);

        assert!((mesh_t0.positions[0].x - 0.0).abs() < 1e-5);
        assert!((mesh_t0.positions[0].y - 2.0).abs() < 1e-5);
        assert!((mesh_t1.positions[0].x + 1.0).abs() < 1e-5);
        assert!((mesh_t1.positions[0].y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_world_to_mesh_rotates_tangent_basis() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![Vec3::new(1.0, 0.0, 0.0)];
        mesh.normals = vec![Vec3::Z];
        mesh.tangents = vec![Vec4::new(1.0, 0.0, 0.0, -1.0)];

        let world = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        apply_world_to_mesh(&mut mesh, world).unwrap();

        assert!((mesh.positions[0] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        assert!((mesh.normals[0] - Vec3::Z).length() < 1e-5);
        assert!((mesh.tangents[0].truncate() - Vec3::Y).length() < 1e-5);
        assert!((mesh.tangents[0].w + 1.0).abs() < 1e-6);
    }

    #[test]
    fn skinning_updates_tangent_basis() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![Vec3::new(1.0, 0.0, 0.0)];
        mesh.normals = vec![Vec3::Z];
        mesh.tangents = vec![Vec4::new(1.0, 0.0, 0.0, -1.0)];

        let skinning = PrimitiveSkinningData {
            joints: vec![[0, 0, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]],
        };
        let joint_matrices = vec![Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)];

        skin_mesh_in_place(&mut mesh, &skinning, &joint_matrices).unwrap();

        assert!((mesh.positions[0] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        assert!((mesh.normals[0] - Vec3::Z).length() < 1e-5);
        assert!((mesh.tangents[0].truncate() - Vec3::Y).length() < 1e-5);
        assert!((mesh.tangents[0].w + 1.0).abs() < 1e-6);
    }

    #[test]
    fn gltf_skinning_animation_renders_deterministically() {
        let gltf = tiny_skinned_gltf_inline();
        let mut scene_a = load_gltf_str_at_time(&gltf, 0, 0.0).unwrap();
        let mut scene_b = load_gltf_str_at_time(&gltf, 0, 1.0).unwrap();
        let mut scene_a_repeat = load_gltf_str_at_time(&gltf, 0, 0.0).unwrap();

        for scene in [&mut scene_a, &mut scene_b, &mut scene_a_repeat] {
            scene.camera = Camera::new(
                Transform::IDENTITY,
                Projection::Orthographic {
                    half_width: None,
                    half_height: 2.0,
                    near: -10.0,
                    far: 10.0,
                },
            );
        }

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

        assert_eq!(img_a.hash64(), img_a_repeat.hash64());
        assert_ne!(img_a.hash64(), img_b.hash64());
    }
}
