use avian3d::prelude::{
    Collider, CollisionEventsEnabled, Collisions, Friction, LinearVelocity, OnCollisionStart,
    Position, Restitution, RigidBody, Rotation,
};
use bevy::{
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    prelude::*,
};
use bevy_firework::{
    core::{BlendMode, EmissionMode, ParticleSpawner, ParticleSpawnerFinished, SpawnTransformMode},
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
    #[cfg(feature = "physics_avian")]
    app.add_plugins(avian3d::prelude::PhysicsPlugins::default());

    app.run();
}

const BALL_RADIUS: f32 = 0.5;

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

    // base
    commands.run_system_cached_with(spawn_wall, (Vec3::new(0., -3., 0.), Vec3::new(8., 1., 8.)));
    // walls
    commands.run_system_cached_with(spawn_wall, (Vec3::new(-4., 0., 0.), Vec3::new(1., 6., 8.)));
    commands.run_system_cached_with(spawn_wall, (Vec3::new(4., 0., 0.), Vec3::new(1., 6., 8.)));
    commands.run_system_cached_with(spawn_wall, (Vec3::new(0., 0., -4.), Vec3::new(8., 6., 1.)));
    commands.run_system_cached_with(spawn_wall, (Vec3::new(0., 0., 4.), Vec3::new(8., 6., 1.)));

    // bouncing ball
    commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(BALL_RADIUS))),
            MeshMaterial3d(materials.add(Color::LinearRgba(LinearRgba::rgb(0.85, 0.1, 0.2)))),
            Transform::from_translation(Vec3::new(0., 3., 0.)),
            Collider::sphere(BALL_RADIUS),
            RigidBody::Dynamic,
            LinearVelocity(Vec3::new(8., 0., 6.)),
            Friction::ZERO,
            Restitution {
                coefficient: 1.,
                ..default()
            },
            CollisionEventsEnabled,
        ))
        .observe(
            |trigger: Trigger<OnCollisionStart>,
             mut commands: Commands,
             collisions: Collisions,
             collider: Query<(&Position, &Rotation)>| {
                collisions
                    .collisions_with(trigger.collider)
                    .for_each(|pair| {
                        let (impulse, mut normal) = pair.max_normal_impulse();
                        if pair.collider1 != trigger.collider {
                            normal = -normal;
                        }

                        let Ok((position, rotation)) = collider.get(trigger.collider) else {
                            return;
                        };
                        let translation = pair.find_deepest_contact().map_or(Vec3::ZERO, |c| {
                            if pair.collider1 == trigger.collider {
                                c.global_point1(position, rotation)
                            } else {
                                c.global_point2(position, rotation)
                            }
                        });

                        commands
                            .spawn((
                                ParticleSpawner {
                                    emission_mode: EmissionMode::OneShot(20),
                                    emission_shape: EmissionShape::Circle {
                                        normal: Vec3::Y,
                                        radius: 0.4,
                                    },
                                    lifetime: RandF32::constant(2.5),
                                    inherit_parent_velocity: true,
                                    initial_velocity: RandVec3 {
                                        direction: Vec3::Y,
                                        magnitude: RandF32 { min: 0., max: 2. },
                                        spread: 0.,
                                    },
                                    initial_velocity_radial: RandF32 { min: 0., max: 2.5 },
                                    initial_scale: RandF32 {
                                        min: (impulse / 10. - 0.1).max(0.),
                                        max: (impulse / 10. + 0.1).min(1.),
                                    },
                                    scale_curve: FireworkCurve::even_samples(vec![1., 2.]),
                                    color: FireworkGradient::uneven_samples(vec![
                                        (0., LinearRgba::new(0.6, 0.3, 0., 0.)),
                                        (0.1, LinearRgba::new(0.6, 0.3, 0., 0.35)),
                                        (1., LinearRgba::new(0.6, 0.3, 0., 0.0)),
                                    ]),
                                    blend_mode: BlendMode::Blend,
                                    linear_drag: 0.7,
                                    pbr: true,
                                    acceleration: Vec3::new(0., -1.5, 0.),
                                    fade_scene: 3.5,
                                    spawn_transform_mode: SpawnTransformMode::Local,
                                    ..default()
                                },
                                Transform {
                                    translation,
                                    rotation: Quat::from_rotation_arc(Vec3::Y, normal),
                                    ..default()
                                },
                            ))
                            .observe(
                                |trigger: Trigger<ParticleSpawnerFinished>,
                                 mut commands: Commands| {
                                    commands.entity(trigger.target()).despawn();
                                },
                            );
                    });
            },
        );

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
        Transform::from_xyz(-2.5, 10., 4.0).looking_at(Vec3::new(0., -3., 0.), Vec3::Y),
        Bloom::default(),
        DepthPrepass::default(),
        // For now,Msaa must be disabled on the web due to this:
        // https://github.com/gfx-rs/wgpu/issues/5263
        #[cfg(target_arch = "wasm32")]
        Msaa::Off,
    ));
}

fn spawn_wall(
    In((center, size)): In<(Vec3, Vec3)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(size))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform {
            translation: center,
            ..default()
        },
        Collider::cuboid(size.x, size.y, size.z),
        RigidBody::Static,
        Friction::ZERO,
        Restitution {
            coefficient: 1.,
            ..default()
        },
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
