use bevy::{core_pipeline::bloom::Bloom, prelude::*};
use bevy_firework::{
    core::{BlendMode, ParticleSpawner},
    curve::{FireworkCurve, FireworkGradient},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;
use std::f32::consts::PI;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);

    app.add_plugins(ParticleSystemPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, adjust_time_scale);

    #[cfg(feature = "physics_avian")]
    app.add_plugins(avian3d::prelude::PhysicsPlugins::default());

    app.run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // spawn text
    commands.spawn((
        Text("Press Space to toggle slow motion".to_string()),
        TextFont {
            font_size: 40.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Transform::from_xyz(-4.0, 4.0, 0.0),
    ));

    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        ParticleSpawner {
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
            scale_curve: FireworkCurve::constant(1.),
            color: FireworkGradient::uneven_samples(vec![
                (0., LinearRgba::new(150., 100., 15., 1.)),
                (0.7, LinearRgba::new(3., 1., 1., 1.)),
                (0.8, LinearRgba::new(1., 0.3, 0.3, 1.)),
                (0.9, LinearRgba::new(0.3, 0.3, 0.3, 1.)),
                (1., LinearRgba::new(0.1, 0.1, 0.1, 0.)),
            ]),
            blend_mode: BlendMode::Blend,
            linear_drag: 0.1,
            pbr: false,
            ..default()
        },
        Transform::from_xyz(0., 0.1, 0.),
    ));

    // light
    commands.spawn((
        PointLight {
            intensity: 1500000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        Bloom::default(),
        // For now,Msaa must be disabled on the web due to this:
        // https://github.com/gfx-rs/wgpu/issues/5263
        #[cfg(target_arch = "wasm32")]
        Msaa::Off,
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
