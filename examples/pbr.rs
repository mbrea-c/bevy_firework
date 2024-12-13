use bevy::{
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    prelude::*,
};
use bevy_firework::{
    core::{BlendMode, ParticleSpawner},
    curve::{FireworkCurve, FireworkGradient},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);

    app.add_plugins(ParticleSystemPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (adjust_time_scale, rotate_point_light));
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
            rate: 150.0,
            emission_shape: EmissionShape::Circle {
                normal: Vec3::Y,
                radius: 3.5,
            },
            lifetime: RandF32::constant(5.),
            inherit_parent_velocity: true,
            initial_velocity: RandVec3::constant(Vec3::ZERO),
            initial_scale: RandF32 { min: 0.5, max: 1.3 },
            scale_curve: FireworkCurve::even_samples(vec![1., 2.]),
            color: FireworkGradient::uneven_samples(vec![
                (0., LinearRgba::new(0.6, 0.3, 0., 0.)),
                (0.1, LinearRgba::new(0.6, 0.3, 0., 0.35)),
                (1., LinearRgba::new(0.6, 0.3, 0., 0.0)),
            ]),
            blend_mode: BlendMode::Blend,
            linear_drag: 0.7,
            pbr: true,
            acceleration: Vec3::new(0., 0.3, 0.),
            fade_scene: 3.5,
            ..default()
        },
        Transform::from_xyz(0., 0.1, 0.),
    ));

    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(Vec3::ONE))),
        MeshMaterial3d(materials.add(Color::LinearRgba(LinearRgba::rgb(0.8, 0.7, 0.6)))),
        Transform::from_xyz(1.0, 1.5, 0.0),
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
        DepthPrepass::default(),
        // For now,Msaa must be disabled on the web due to this:
        // https://github.com/gfx-rs/wgpu/issues/5263
        #[cfg(target_arch = "wasm32")]
        Msaa::Off,
    ));
}

fn rotate_point_light(mut point_lights: Query<&mut Transform, With<PointLight>>, time: Res<Time>) {
    for mut transform in &mut point_lights {
        transform.translation = Vec3::new(
            4. * time.elapsed_secs().sin(),
            8. * ((time.elapsed_secs() * 0.78932).sin() + 1.) / 2.,
            4. * time.elapsed_secs().cos(),
        );
    }
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
