use glam::{Mat4, Vec3, Vec4};

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

impl Projection {
    pub fn near(&self) -> f32 {
        match self {
            Projection::Perspective { near, .. } => *near,
            Projection::Orthographic { near, .. } => *near,
        }
    }

    pub fn far(&self) -> f32 {
        match self {
            Projection::Perspective { far, .. } => *far,
            Projection::Orthographic { far, .. } => *far,
        }
    }

    pub fn matrix(&self, aspect: f32) -> Mat4 {
        match self {
            Projection::Perspective { fov_y_radians, near, far } => {
                let f = 1.0 / (0.5 * *fov_y_radians).tan();
                let a = aspect;
                let n = *near;
                let fa = *far;

                
                let m00 = f / a;
                let m11 = f;
                let m22 = (fa + n) / (n - fa);
                let m23 = (2.0 * fa * n) / (n - fa);
                let m32 = -1.0;

                Mat4::from_cols(
                    Vec4::new(m00, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, m11, 0.0, 0.0),
                    Vec4::new(0.0, 0.0, m22, m32),
                    Vec4::new(0.0, 0.0, m23, 0.0),
                )
            }
            Projection::Orthographic { half_height, near, far } => {
                let hh = *half_height;
                let hw = hh * aspect;

                let l = -hw;
                let r = hw;
                let b = -hh;
                let t = hh;

                let n = *near;
                let fa = *far;

                let m00 = 2.0 / (r - l);
                let m11 = 2.0 / (t - b);
                let m22 = -2.0 / (fa - n);

                let m03 = -(r + l) / (r - l);
                let m13 = -(t + b) / (t - b);
                let m23 = -(fa + n) / (fa - n);

                Mat4::from_cols(
                    Vec4::new(m00, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, m11, 0.0, 0.0),
                    Vec4::new(0.0, 0.0, m22, 0.0),
                    Vec4::new(m03, m13, m23, 1.0),
                )
            }
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

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        self.projection.matrix(aspect)
    }

    pub fn near(&self) -> f32 {
        self.projection.near()
    }

    pub fn far(&self) -> f32 {
        self.projection.far()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)),
            projection: Projection::default(),
        }
    }
}
