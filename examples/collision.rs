use bevy::{core_pipeline::bloom::BloomSettings, prelude::*};
use bevy_firework::{
    core::{BlendMode, ParticleCollisionSettings, ParticleSpawnerBundle, ParticleSpawnerSettings},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;
use bevy_xpbd_3d::prelude::*;
use std::f32::consts::PI;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(PhysicsPlugins::default());

    // For now,Msaa must be disabled on the web due to this:
    // https://github.com/gfx-rs/wgpu/issues/5263
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(Msaa::Off);

    app.add_plugins(ParticleSystemPlugin)
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
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Cuboid::from_size(Vec3::new(8., 1., 8.))),
            material: materials.add(Color::WHITE),
            transform: Transform::from_translation(Vec3::new(0., -0.5, 0.)),
            ..default()
        })
        .insert(Collider::cuboid(8., 1., 8.));
    commands
        .spawn(ParticleSpawnerBundle::from_settings(
            ParticleSpawnerSettings {
                rate: 100.0,
                one_shot: false,
                emission_shape: EmissionShape::Circle {
                    normal: Vec3::Y,
                    radius: 0.3,
                },
                lifetime: RandF32::constant(6.75),
                initial_velocity: RandVec3 {
                    magnitude: RandF32 { min: 6., max: 8. },
                    direction: Vec3::Y,
                    spread: 30. / 180. * PI,
                },
                inherit_parent_velocity: true,
                initial_scale: RandF32 {
                    min: 0.02,
                    max: 0.08,
                },
                scale_curve: ParamCurve::constant(1.),
                linear_drag: 0.15,
                color: Gradient::linear(vec![
                    (0., Color::rgba(10., 7., 1., 1.).into()),
                    (0.7, Color::rgba(3., 1., 1., 1.).into()),
                    (0.8, Color::rgba(1., 0.3, 0.3, 1.).into()),
                    (0.9, Color::rgba(0.3, 0.3, 0.3, 1.).into()),
                    (1., Color::rgba(0.1, 0.1, 0.1, 0.).into()),
                ]),
                blend_mode: BlendMode::Blend,
                pbr: true,
                collision_settings: Some(ParticleCollisionSettings {
                    restitution: 0.6,
                    friction: 0.2,
                    filter: SpatialQueryFilter::default(),
                }),
                ..default()
            },
        ))
        .insert(Transform {
            translation: Vec3::new(5., 0.5, 0.),
            rotation: Quat::from_rotation_z(PI / 4.),
            ..default()
        });

    // angled cube
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Cuboid::from_size(Vec3::ONE)),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6)),
            transform: Transform {
                translation: Vec3::new(0., 0.5, 0.),
                rotation: Quat::from_rotation_x(PI / 4.) * Quat::from_rotation_y(PI / 4.),
                ..default()
            },
            ..default()
        })
        .insert(Collider::cuboid(1., 1., 1.));

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500000.0,
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
    input: Res<ButtonInput<KeyCode>>,
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
