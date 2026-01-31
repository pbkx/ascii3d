pub mod buffer;
pub mod image;
#[cfg(feature = "terminal")]
pub mod terminal;

pub use buffer::{BufferTarget, Cell};
pub use image::ImageTarget;
#[cfg(feature = "terminal")]
pub use terminal::{TerminalPresenter, TerminalTarget};
