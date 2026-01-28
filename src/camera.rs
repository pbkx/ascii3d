use glam::Mat4;

use crate::Transform;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective {
        fov_y_radians: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        half_height: f32,
        near: f32,
        far: f32,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::Perspective {
            fov_y_radians: 60.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub transform: Transform,
    pub projection: Projection,
}

impl Camera {
    pub fn new(transform: Transform, projection: Projection) -> Self {
        Self {
            transform,
            projection,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.transform.to_mat4().inverse()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Transform::IDENTITY,
            projection: Projection::default(),
        }
    }
}
