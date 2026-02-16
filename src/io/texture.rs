use crate::texture::Texture;
use std::{error::Error, fmt, fs, path::Path};

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
    let mut tex = load_texture_rgba8_raw(path)?;
    linearize_rgb_in_place(&mut tex);
    Ok(tex)
}

pub fn load_texture_rgba8_raw(path: impl AsRef<Path>) -> Result<Texture, TextureIoError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|_| TextureIoError::Io)?;
    load_texture_rgba8_from_bytes_raw(&bytes)
}

pub fn load_texture_rgba8_from_bytes(bytes: &[u8]) -> Result<Texture, TextureIoError> {
    let mut tex = load_texture_rgba8_from_bytes_raw(bytes)?;
    linearize_rgb_in_place(&mut tex);
    Ok(tex)
}

pub fn load_texture_rgba8_from_bytes_raw(bytes: &[u8]) -> Result<Texture, TextureIoError> {
    #[cfg(feature = "image")]
    {
        let img = image::load_from_memory(bytes).map_err(|_| TextureIoError::Decode)?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Texture::from_rgba8(w, h, rgba.into_raw()).ok_or(TextureIoError::Invalid)
    }
    #[cfg(not(feature = "image"))]
    {
        let _ = bytes;
        Err(TextureIoError::ImageFeatureDisabled)
    }
}

fn srgb_to_linear_u8(v: u8) -> u8 {
    let srgb = v as f32 / 255.0;
    let lin = if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    };
    ((lin.clamp(0.0, 1.0) * 255.0) + 0.5).floor() as u8
}

fn linearize_rgb_in_place(tex: &mut Texture) {
    for px in tex.rgba8.chunks_exact_mut(4) {
        px[0] = srgb_to_linear_u8(px[0]);
        px[1] = srgb_to_linear_u8(px[1]);
        px[2] = srgb_to_linear_u8(px[2]);
    }
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_raw_preserves_encoded_values() {
        let mut img = image::RgbaImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgba([128, 64, 32, 255]));
        let mut bytes = Vec::new();
        {
            let dyn_img = image::DynamicImage::ImageRgba8(img);
            let mut cursor = std::io::Cursor::new(&mut bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
        }
        let tex = load_texture_rgba8_from_bytes_raw(&bytes).unwrap();
        assert_eq!(tex.rgba8[0], 128);
        assert_eq!(tex.rgba8[1], 64);
        assert_eq!(tex.rgba8[2], 32);
    }

    #[test]
    fn from_bytes_linearizes_rgb_channels() {
        let mut img = image::RgbaImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgba([128, 64, 32, 255]));
        let mut bytes = Vec::new();
        {
            let dyn_img = image::DynamicImage::ImageRgba8(img);
            let mut cursor = std::io::Cursor::new(&mut bytes);
            dyn_img
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
        }
        let tex = load_texture_rgba8_from_bytes(&bytes).unwrap();
        assert!(tex.rgba8[0] < 128);
        assert!(tex.rgba8[1] < 64);
        assert!(tex.rgba8[2] < 32);
        assert_eq!(tex.rgba8[3], 255);
    }
}
