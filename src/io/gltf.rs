use crate::{Material, Mesh, Scene, Transform};
use glam::{Mat4, Quat, Vec2, Vec3};
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
    nodes: Vec<NodeDef>,
    #[serde(default)]
    scenes: Vec<SceneDef>,
    #[serde(default)]
    scene: Option<usize>,
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
    mode: Option<u32>,
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

struct AccessorLayout<'a> {
    accessor: &'a AccessorDef,
    data: &'a [u8],
    start: usize,
    stride: usize,
    elem_size: usize,
    count: usize,
    view_end: usize,
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

    let uvs = if let Some(uv) = primitive.attributes.get("TEXCOORD_0") {
        read_accessor_vec2(root, buffers, *uv)?
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

fn node_local_matrix(node: &NodeDef) -> Result<Mat4, GltfError> {
    let m = if let Some(arr) = node.matrix {
        Mat4::from_cols_array(&arr)
    } else {
        let t = node.translation.unwrap_or([0.0, 0.0, 0.0]);
        let r = node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let s = node.scale.unwrap_or([1.0, 1.0, 1.0]);

        let trans = Vec3::new(t[0], t[1], t[2]);
        let rot = Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize();
        let scale = Vec3::new(s[0], s[1], s[2]);

        Mat4::from_translation(trans) * Mat4::from_quat(rot) * Mat4::from_scale(scale)
    };
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

fn build_scene(root: &GltfRoot, buffers: &[Vec<u8>]) -> Result<Scene, GltfError> {
    let mut per_mesh_primitives: Vec<Vec<Mesh>> = Vec::with_capacity(root.meshes.len());
    for mesh in &root.meshes {
        let mut prims = Vec::with_capacity(mesh.primitives.len());
        for p in &mesh.primitives {
            prims.push(primitive_to_mesh(root, buffers, p)?);
        }
        per_mesh_primitives.push(prims);
    }

    let mut scene = Scene::new();
    let mut visiting = vec![false; root.nodes.len()];

    fn walk(
        node_index: usize,
        root: &GltfRoot,
        per_mesh_primitives: &[Vec<Mesh>],
        parent_world: Mat4,
        visiting: &mut [bool],
        scene: &mut Scene,
    ) -> Result<(), GltfError> {
        let node = root.nodes.get(node_index).ok_or(GltfError::InvalidIndex)?;
        if visiting[node_index] {
            return Err(GltfError::InvalidNodeGraph);
        }
        visiting[node_index] = true;

        let local = node_local_matrix(node)?;
        let world = parent_world * local;

        if let Some(mesh_index) = node.mesh {
            let primitives = per_mesh_primitives.get(mesh_index).ok_or(GltfError::InvalidIndex)?;
            let xf = world_to_transform(world)?;
            for mesh in primitives {
                let _ = scene.add_object(mesh.clone(), xf, Material::default());
            }
        }

        for &child in &node.children {
            walk(child, root, per_mesh_primitives, world, visiting, scene)?;
        }

        visiting[node_index] = false;
        Ok(())
    }

    if root.scenes.is_empty() {
        for i in 0..root.nodes.len() {
            walk(i, root, &per_mesh_primitives, Mat4::IDENTITY, &mut visiting, &mut scene)?;
        }
        return Ok(scene);
    }

    let scene_index = root.scene.unwrap_or(0);
    let selected = root.scenes.get(scene_index).ok_or(GltfError::InvalidIndex)?;
    for &node in &selected.nodes {
        walk(node, root, &per_mesh_primitives, Mat4::IDENTITY, &mut visiting, &mut scene)?;
    }
    Ok(scene)
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

fn parse_and_load(src: &str, base_dir: Option<&Path>) -> Result<Scene, GltfError> {
    let root: GltfRoot = serde_json::from_str(src).map_err(|_| GltfError::Json)?;
    if root.asset.version != "2.0" {
        return Err(GltfError::UnsupportedVersion);
    }

    let mut buffers = Vec::with_capacity(root.buffers.len());
    for b in &root.buffers {
        buffers.push(load_buffer_bytes(b, base_dir)?);
    }

    build_scene(&root, &buffers)
}

pub fn load_gltf(path: impl AsRef<Path>) -> Result<Scene, GltfError> {
    let path = path.as_ref();
    let src = fs::read_to_string(path).map_err(|_| GltfError::Io)?;
    let base_dir: PathBuf = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parse_and_load(&src, Some(&base_dir))
}

pub fn load_gltf_str(src: &str) -> Result<Scene, GltfError> {
    parse_and_load(src, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Renderer, RendererConfig};

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
}
