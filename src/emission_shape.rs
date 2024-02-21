use bevy::prelude::*;
use bevy_utilitarian::geometric::pitchyaw::PitchYaw;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[derive(Default, Clone, Copy, Debug, Reflect, Serialize, Deserialize)]
pub enum EmissionShape {
    #[default]
    Point,
    Sphere(f32),
    Circle {
        normal: Vec3,
        radius: f32,
    },
}

impl EmissionShape {
    pub fn generate_point(&self) -> Vec3 {
        match self {
            Self::Point => Vec3::ZERO,
            Self::Sphere(radius) => {
                let (u, v, r) = (
                    rand::random::<f32>() * 2. * PI,
                    rand::random::<f32>() * PI,
                    rand::random::<f32>(),
                );

                let spherical = PitchYaw::new(u, v);

                spherical.to_unit_vec() * r * (*radius)
            }
            Self::Circle { normal, radius } => {
                let (u, r) = (rand::random::<f32>() * 2. * PI, rand::random::<f32>());
                Quat::from_rotation_arc(Vec3::Y, *normal)
                    * Quat::from_rotation_y(u)
                    * Vec3::new(r * radius, 0., 0.)
            }
        }
    }
}
