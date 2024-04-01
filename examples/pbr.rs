use bevy::{
    core_pipeline::{bloom::BloomSettings, prepass::DepthPrepass},
    prelude::*,
};
use bevy_firework::{
    core::{BlendMode, ParticleSpawnerBundle, ParticleSpawnerSettings},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);

    // For now,Msaa must be disabled on the web due to this:
    // https://github.com/gfx-rs/wgpu/issues/5263
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(Msaa::Off);

    // The particle system plugin must be added **after** any changes
    // to the MSAA setting.
    app.add_plugins(ParticleSystemPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, adjust_time_scale);
    #[cfg(feature = "physics_xpbd")]
    app.add_plugins(bevy_xpbd_3d::prelude::PhysicsPlugins::default());

    app.run();
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
        mesh: meshes.add(Circle::new(4.0)),
        material: materials.add(Color::WHITE),
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ..default()
    });
    commands
        .spawn(ParticleSpawnerBundle::from_settings(
            ParticleSpawnerSettings {
                one_shot: false,
                rate: 80.0,
                emission_shape: EmissionShape::Circle {
                    normal: Vec3::Y,
                    radius: 2.0,
                },
                lifetime: RandF32::constant(5.),
                inherit_parent_velocity: true,
                initial_velocity: RandVec3::constant(Vec3::ZERO),
                initial_scale: RandF32 { min: 0.5, max: 1.3 },
                scale_curve: ParamCurve::linear_uniform(vec![1., 2.]),
                color: Gradient::linear(vec![
                    (0., Color::rgba(0.6, 0.3, 0., 0.).into()),
                    (0.1, Color::rgba(0.6, 0.3, 0., 0.5).into()),
                    (1., Color::rgba(0.6, 0.3, 0., 0.0).into()),
                ]),
                blend_mode: BlendMode::Blend,
                linear_drag: 0.7,
                pbr: true,
                acceleration: Vec3::new(0., 0.3, 0.),
                fade_scene: 3.5,
                ..default()
            },
        ))
        .insert(Transform::from_xyz(0., 0.1, 0.));
    // cube
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::from_size(Vec3::ONE)),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6)),
        transform: Transform::from_xyz(1.0, 1.5, 0.0),
        ..default()
    });

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
        DepthPrepass::default(),
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
