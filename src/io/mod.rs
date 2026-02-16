#[cfg(feature = "gltf")]
pub mod gltf;
pub mod mtl;
pub mod obj;
pub mod stl;
pub mod texture;

#[cfg(feature = "gltf")]
pub use gltf::{load_gltf, load_gltf_at_time, load_gltf_str, load_gltf_str_at_time, GltfError};
pub use mtl::{parse_mtl, MtlError, MtlLibrary};
pub use obj::{
    load_obj_with_mtl, load_obj_with_mtl_mesh_materials, load_obj_with_mtl_str,
    load_obj_with_mtl_str_mesh_materials, LoadedObj, ObjError,
};
pub use stl::{load_stl, load_stl_str, StlError};
pub use texture::{load_texture_rgba8, load_texture_rgba8_from_bytes, TextureIoError};
