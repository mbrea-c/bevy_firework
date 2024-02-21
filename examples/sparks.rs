use bevy::{core_pipeline::bloom::BloomSettings, prelude::*};
use bevy_firework::{
    core::{BlendMode, ParticleSpawnerBundle, ParticleSpawnerSettings},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;
use std::f32::consts::PI;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ParticleSystemPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, adjust_time_scale)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // spawn text
    commands.spawn(TextBundle {
        text: Text {
            sections: vec![TextSection {
                value: "Press Space to toggle slow motion".to_string(),
                style: TextStyle {
                    font_size: 40.0,
                    color: Color::WHITE,
                    ..default()
                },
            }],
            ..Default::default()
        },
        transform: Transform::from_xyz(-4.0, 4.0, 0.0),
        ..Default::default()
    });

    // circular base
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Circle::new(4.0).into()),
        material: materials.add(Color::WHITE.into()),
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ..default()
    });
    commands
        .spawn(ParticleSpawnerBundle::from_settings(
            ParticleSpawnerSettings {
                one_shot: false,
                rate: 1000.0,
                emission_shape: EmissionShape::Circle {
                    normal: Vec3::Y,
                    radius: 0.3,
                },
                lifetime: RandF32::constant(0.75),
                inherit_parent_velocity: true,
                initial_velocity: RandVec3 {
                    magnitude: RandF32 { min: 0., max: 10. },
                    direction: Vec3::Y,
                    spread: 30. / 180. * PI,
                },
                initial_scale: RandF32 {
                    min: 0.02,
                    max: 0.08,
                },
                scale_curve: ParamCurve::constant(1.),
                color: Gradient::linear(vec![
                    (0., Color::rgba(300., 100., 1., 1.).into()),
                    (0.7, Color::rgba(3., 1., 1., 1.).into()),
                    (0.8, Color::rgba(1., 0.3, 0.3, 1.).into()),
                    (0.9, Color::rgba(0.3, 0.3, 0.3, 1.).into()),
                    (1., Color::rgba(0.1, 0.1, 0.1, 0.).into()),
                ]),
                blend_mode: BlendMode::Blend,
                linear_drag: 0.1,
                pbr: false,
                ..default()
            },
        ))
        .insert(Transform::from_xyz(0., 0.1, 0.));

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                hdr: true,

                ..default()
            },
            ..default()
        },
        BloomSettings::default(),
    ));
}

fn adjust_time_scale(
    mut slowmo: Local<bool>,
    mut time: ResMut<Time<Virtual>>,
    input: Res<Input<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Space) {
        *slowmo = !*slowmo;
    }

    if *slowmo {
        time.set_relative_speed(0.05);
    } else {
        time.set_relative_speed(1.0);
    }
}
