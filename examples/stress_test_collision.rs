use bevy::{
    core_pipeline::bloom::BloomSettings,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use bevy_firework::{
    core::{
        BlendMode, ParticleCollisionSettings, ParticleSpawnerBundle, ParticleSpawnerData,
        ParticleSpawnerSettings,
    },
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;
use bevy_xpbd_3d::plugins::{
    collision::Collider, spatial_query::SpatialQueryFilter, PhysicsPlugins,
};
use std::f32::consts::PI;

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, FrameTimeDiagnosticsPlugin))
        .add_plugins(PhysicsPlugins::default());

    // For now,Msaa must be disabled on the web due to this:
    // https://github.com/gfx-rs/wgpu/issues/5263
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(Msaa::Off);

    // The particle system plugin must be added **after** any changes
    // to the MSAA setting.
    app.add_plugins(ParticleSystemPlugin)
        .init_resource::<DebugInfo>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                adjust_time_scale,
                ((update_fps, update_particle_counts), update_debug_info_text).chain(),
            ),
        )
        .run();
}

#[derive(Resource, Default)]
struct DebugInfo {
    fps: f32,
    particle_count_collision: usize,
    particle_count_no_collision: usize,
    particle_system_count: usize,
}

#[derive(Component)]
struct DebugInfoText;

impl std::fmt::Debug for DebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FPS: {:.2}", self.fps)?;
        writeln!(
            f,
            "Particle count (with collision): {}",
            self.particle_count_collision
        )?;
        writeln!(
            f,
            "Particle count (without collision): {}",
            self.particle_count_no_collision
        )?;
        writeln!(f, "Particle system count: {}", self.particle_system_count)?;

        Ok(())
    }
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // spawn text
    // commands.spawn(TextBundle {
    //     text: Text {
    //         sections: vec![TextSection {
    //             value: "Press Space to toggle slow motion".to_string(),
    //             style: TextStyle {
    //                 font_size: 40.0,
    //                 color: Color::WHITE,
    //                 ..default()
    //             },
    //         }],
    //         ..Default::default()
    //     },
    //     transform: Transform::from_xyz(-4.0, 4.0, 0.0),
    //     ..Default::default()
    // });

    // Text with multiple sections
    commands.spawn((
        // Create a TextBundle that has a Text with a list of sections.
        TextBundle::from_sections([TextSection::new(
            "FPS: ",
            TextStyle {
                // This font is loaded and will be used instead of the default font.
                font_size: 20.0,
                ..default()
            },
        )]),
        DebugInfoText,
    ));

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
                rate: 80000.0,
                one_shot: false,
                emission_shape: EmissionShape::Circle {
                    normal: Vec3::Y,
                    radius: 0.3,
                },
                lifetime: RandF32::constant(2.0),
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
                    (0., LinearRgba::new(100., 70., 10., 1.)),
                    (0.7, LinearRgba::new(3., 1., 1., 1.)),
                    (0.8, LinearRgba::new(1., 0.3, 0.3, 1.)),
                    (0.9, LinearRgba::new(0.3, 0.3, 0.3, 1.)),
                    (1., LinearRgba::new(0.1, 0.1, 0.1, 0.)),
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
            material: materials.add(Color::from(LinearRgba::new(0.8, 0.7, 0.6, 1.))),
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

fn update_debug_info_text(
    debug_info: Res<DebugInfo>,
    mut debug_text_query: Query<&mut Text, With<DebugInfoText>>,
) {
    for mut text in &mut debug_text_query {
        text.sections[0].value = format!("{:?}", debug_info);
    }
}

fn update_fps(mut debug_info: ResMut<DebugInfo>, diagnostics: Res<DiagnosticsStore>) {
    if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps.smoothed() {
            debug_info.fps = value as f32;
        }
    }
}

fn update_particle_counts(
    mut debug_info: ResMut<DebugInfo>,
    particle_systems: Query<(&ParticleSpawnerSettings, &ParticleSpawnerData)>,
) {
    debug_info.particle_system_count = 0;
    debug_info.particle_count_collision = 0;
    debug_info.particle_count_no_collision = 0;

    for (settings, data) in &particle_systems {
        debug_info.particle_system_count += 1;
        if settings.collision_settings.is_some() {
            debug_info.particle_count_collision += data.particles.len();
        } else {
            debug_info.particle_count_no_collision += data.particles.len();
        }
    }
}
