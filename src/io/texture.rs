use crate::texture::Texture;
use std::{
    error::Error,
    fmt,
    fs,
    path::Path,
};

#[derive(Clone, Debug)]
pub enum TextureIoError {
    Io,
    ImageFeatureDisabled,
    Decode,
    Invalid,
}

impl fmt::Display for TextureIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => write!(f, "io error"),
            Self::ImageFeatureDisabled => write!(f, "feature `image` is disabled"),
            Self::Decode => write!(f, "failed to decode image"),
            Self::Invalid => write!(f, "invalid texture"),
        }
    }
}

impl Error for TextureIoError {}

pub fn load_texture_rgba8(path: impl AsRef<Path>) -> Result<Texture, TextureIoError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|_| TextureIoError::Io)?;
    load_texture_rgba8_from_bytes(&bytes)
}

pub fn load_texture_rgba8_from_bytes(bytes: &[u8]) -> Result<Texture, TextureIoError> {
    #[cfg(feature = "image")]
    {
        let img = image::load_from_memory(bytes).map_err(|_| TextureIoError::Decode)?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let w = u32::try_from(w).map_err(|_| TextureIoError::Invalid)?;
        let h = u32::try_from(h).map_err(|_| TextureIoError::Invalid)?;
        Texture::from_rgba8(w, h, rgba.into_raw()).ok_or(TextureIoError::Invalid)
    }
    #[cfg(not(feature = "image"))]
    {
        let _ = bytes;
        Err(TextureIoError::ImageFeatureDisabled)
    }
}
