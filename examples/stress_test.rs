use bevy::{
    core_pipeline::bloom::BloomSettings,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use bevy_firework::{
    core::{BlendMode, ParticleSpawnerBundle, ParticleSpawnerData, ParticleSpawnerSettings},
    emission_shape::EmissionShape,
    plugin::ParticleSystemPlugin,
};
use bevy_utilitarian::prelude::*;
use bevy_xpbd_3d::plugins::PhysicsPlugins;
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
                rate: 160000.0,
                emission_shape: EmissionShape::Circle {
                    normal: Vec3::Y,
                    radius: 0.3,
                },
                lifetime: RandF32::constant(1.),
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
                    (0., Color::rgba(10., 7., 1., 1.).into()),
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
