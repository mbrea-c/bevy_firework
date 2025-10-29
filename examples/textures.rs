use std::f32::consts::FRAC_PI_2;

use avian3d::prelude::{Collider, SpatialQueryFilter};
use bevy::{
    core_pipeline::prepass::DepthPrepass,
    image::{ImageLoaderSettings, ImageSamplerDescriptor},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
};
use bevy_firework::{
    core::{
        BlendMode, EmissionMode, EmissionPacing, EmissionSettings, ParticleCollisionSettings,
        ParticleSettings, ParticleSpawner, SpawnTransformMode,
    },
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
        .add_systems(Update, adjust_time_scale);
    app.add_plugins(avian3d::prelude::PhysicsPlugins::default());

    app.run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
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

    commands.spawn((
        ParticleSpawner {
            particle_settings: vec![
                ParticleSettings {
                    lifetime: RandF32::constant(5.),
                    scale_curve: FireworkCurve::constant(1.),
                    initial_scale: RandF32::constant(0.3),
                    linear_drag: 0.3,
                    angular_drag: 0.85,
                    base_color: FireworkGradient::uneven_samples(vec![
                        (0., LinearRgba::WHITE),
                        (0.9, LinearRgba::WHITE),
                        (1., LinearRgba::WHITE.with_alpha(0.)),
                    ]),
                    base_color_texture: Some(asset_server.load_with_settings(
                        "textures/bullet_case/diffuse.png",
                        |settings: &mut ImageLoaderSettings| {
                            settings.sampler = bevy::image::ImageSampler::Descriptor(
                                ImageSamplerDescriptor::nearest(),
                            );
                        },
                    )),
                    normal_map_texture: Some(asset_server.load_with_settings(
                        "textures/bullet_case/normal.png",
                        |settings: &mut ImageLoaderSettings| {
                            settings.is_srgb = false;
                        },
                    )),
                    orm_texture: Some(asset_server.load_with_settings(
                        "textures/bullet_case/orm.png",
                        |settings: &mut ImageLoaderSettings| {
                            settings.is_srgb = false;
                        },
                    )),
                    emissive_color: FireworkGradient::constant(LinearRgba::BLACK),
                    fade_scene: 0.,
                    fade_edge: 0.,
                    blend_mode: BlendMode::Blend,
                    pbr: true,
                    collision_settings: Some(ParticleCollisionSettings {
                        restitution: 0.4,
                        friction: 0.35,
                        filter: SpatialQueryFilter::default(),
                    }),
                    ..default()
                },
                ParticleSettings {
                    lifetime: RandF32::constant(2.),
                    scale_curve: FireworkCurve::even_samples(vec![1., 2.]),
                    initial_scale: RandF32 { min: 0.5, max: 0.8 },
                    acceleration: Vec3::new(0., 0.3, 0.),
                    linear_drag: 0.7,
                    base_color: FireworkGradient::uneven_samples(vec![
                        (0., LinearRgba::new(0.1, 0.1, 0.1, 0.)),
                        (0.1, LinearRgba::new(0.1, 0.1, 0.1, 0.15)),
                        (1., LinearRgba::new(0.1, 0.1, 0.1, 0.0)),
                    ]),
                    base_color_texture: None,
                    emissive_color: FireworkGradient::constant(LinearRgba::BLACK),
                    fade_scene: 3.5,
                    blend_mode: BlendMode::Blend,
                    pbr: true,
                    ..default()
                },
            ],
            emission_settings: vec![
                EmissionSettings {
                    particle_index: 0,
                    emission_mode: EmissionMode::Global,
                    emission_pacing: EmissionPacing::rate(12.),
                    emission_shape: EmissionShape::Point,
                    initial_velocity: RandVec3 {
                        magnitude: RandF32 { min: 2., max: 5. },
                        direction: Vec3::Y,
                        spread: 0.4,
                    },
                    initial_velocity_radial: RandF32::constant(0.),
                    inherit_parent_velocity: true,
                    initial_rotation: Quat::from_rotation_y(FRAC_PI_2),
                    initial_angular_velocity: RandVec3 {
                        magnitude: RandF32 { min: 5., max: 15. },
                        direction: -Vec3::Y,
                        spread: 0.,
                    },
                },
                EmissionSettings {
                    particle_index: 1,
                    emission_mode: EmissionMode::Nested {
                        target_particle_type: 0,
                    },
                    emission_pacing: EmissionPacing::CountOverDuration {
                        count: 6.,
                        offset_start: 0.,
                        offset_end: 0.1,
                        // Is ignored
                        duration: 0.,
                    },
                    emission_shape: EmissionShape::Point,
                    initial_velocity: RandVec3::constant(Vec3::ZERO),
                    initial_velocity_radial: RandF32::constant(0.),
                    inherit_parent_velocity: false,
                    initial_rotation: Quat::IDENTITY,
                    initial_angular_velocity: RandVec3::constant(Vec3::ZERO),
                },
            ],
            starts_enabled: true,
            spawn_transform_mode: SpawnTransformMode::Local,
            ..default()
        },
        Transform {
            translation: Vec3::new(-2., 2., 0.),
            rotation: Quat::from_rotation_arc(Vec3::Y, Vec3::X),
            ..default()
        },
    ));

    // cannon
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.25, 0.25, 1.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: LinearRgba::new(0.1, 0.1, 0.1, 1.).into(),
            metallic: 1.,
            perceptual_roughness: 0.2,
            ..default()
        })),
        Transform {
            translation: Vec3::new(-2., 2., -0.5),
            ..default()
        },
    ));

    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(4., 0.2).mesh().resolution(64))),
        MeshMaterial3d(materials.add(Color::from(LinearRgba::rgb(0.3, 0.1, 0.1)))),
        Transform::IDENTITY,
        Collider::cylinder(4., 0.2),
    ));

    // Pyramid
    commands.spawn((
        Mesh3d(meshes.add(Cone::new(0.5, 1.))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: LinearRgba::rgb(0.6, 0.3, 0.2).into(),
            metallic: 0.,
            perceptual_roughness: 0.05,
            ..default()
        })),
        Transform {
            translation: Vec3::new(0., 0.5, 0.),
            ..default()
        },
        Collider::cone(0.5, 1.),
    ));

    // light
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 4.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Transform::from_xyz(0., 8., 0.).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        Bloom::default(),
        DepthPrepass::default(),
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
