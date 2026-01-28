use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    Directional {
        direction: Vec3,
        color: Vec3,
        intensity: f32,
    },
    Point {
        position: Vec3,
        color: Vec3,
        intensity: f32,
    },
}

impl Light {
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self::Directional {
            direction,
            color,
            intensity,
        }
    }

    pub fn point(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self::Point {
            position,
            color,
            intensity,
        }
    }
}
