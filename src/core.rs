use crate::curve::{FireworkCurve, FireworkGradient};

use super::emission_shape::EmissionShape;
use bevy::{asset::weak_handle, prelude::*, render::batching::NoAutomaticBatching};
use bevy_utilitarian::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "physics_avian")]
use avian3d::prelude::*;

pub const DEFAULT_MESH: Handle<Mesh> = weak_handle!("ba671aee-04f4-485d-9d1e-ad7053dacfab");

#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub enum EmissionMode {
    /// Number of particles emitted at once
    OneShot(usize),
    /// Rate of particles per second
    Rate(f32),
    /// Particles get emitted on method calls
    OnDemand,
}

impl EmissionMode {
    pub fn is_one_shot(&self) -> bool {
        matches!(self, EmissionMode::OneShot(_))
    }
}

/// Mirrors AlphaMode, but implements serialize and deserialize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]
pub enum BlendMode {
    Opaque,
    Blend,
    Premultiplied,
    Add,
    Multiply,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]
pub enum SpawnTransformMode {
    /// Use `GlobalTransform` to determine the initial position of spawned particles
    #[default]
    Global,
    /// Use `Transform` to determine the initial position of spawned particles
    Local,
}

impl From<BlendMode> for AlphaMode {
    fn from(value: BlendMode) -> Self {
        match value {
            BlendMode::Opaque => AlphaMode::Opaque,
            BlendMode::Blend => AlphaMode::Blend,
            BlendMode::Premultiplied => AlphaMode::Premultiplied,
            BlendMode::Add => AlphaMode::Add,
            BlendMode::Multiply => AlphaMode::Multiply,
        }
    }
}

impl From<BlendMode> for u32 {
    fn from(value: BlendMode) -> Self {
        match value {
            BlendMode::Opaque => 0,
            BlendMode::Blend => 2,
            BlendMode::Premultiplied => 3,
            BlendMode::Add => 4,
            BlendMode::Multiply => 5,
        }
    }
}

#[derive(Reflect, Clone, Debug)]
pub struct ParticleSettings {
    /// Lifetime of a particle
    pub lifetime: RandF32,
    /// Evolution of scale over time, applied as a factor to initial scale. The curve should have
    /// domain [0,1]
    pub scale_curve: FireworkCurve<f32>,
    /// Initial scale of particles
    pub initial_scale: RandF32,
    /// Linear acceleration of particles
    pub acceleration: Vec3,
    /// Angular acceleration of particles
    pub angular_acceleration: Vec3,
    /// Drag applied as a linear coefficient of velocity
    pub linear_drag: f32,
    /// Angular drag applied as a linear coefficient of angular velocity
    pub angular_drag: f32,
    /// Color over lifetime. If a texture is specified the final color will be multiplied.
    pub base_color: FireworkGradient<LinearRgba>,
    pub base_color_texture: Option<Handle<Image>>,
    /// Emissive color over lifetime. If a texture is specified the final color will be multiplied.
    pub emissive_color: FireworkGradient<LinearRgba>,
    pub normal_map_texture: Option<Handle<Image>>,
    /// Occlusion/Roughness/Metallic texture; that is, a texture where the red, green and blue channels
    /// represent the material's occlusion, perceptual roughness and "metallic" setting,
    /// respectively. The occlusion value is currently ignored.
    pub orm_texture: Option<Handle<Image>>,
    /// Round out the particle with fading at the edges. If 0, no fading is applied. The value
    /// should be between 0 and 1.
    pub fade_edge: f32,
    /// Fade out at the intersections between particles and the scene to avoid sharp edges.
    /// The larger the value, the longer the fade range.
    pub fade_scene: f32,
    /// Alpha blend mode for the particles
    pub blend_mode: BlendMode,
    /// Whether to use the PBR pipeline for the particle
    pub pbr: bool,
    /// If Some, particles will collide with the scene according to the provided parameters
    /// If None, no particle collision will occur.
    #[cfg(feature = "physics_avian")]
    pub collision_settings: Option<ParticleCollisionSettings>,
}

#[derive(Reflect, Clone, Debug)]
pub struct EmissionSettings {
    /// Which particle settings to use, as an index to the particle settings vector
    pub particle_index: usize,
    /// Determines the way this spawner creates particles over time
    pub emission_mode: EmissionMode,
    /// Shape on which to spawn particles
    pub emission_shape: EmissionShape,
    /// Linear velocity applied to all particles at spawn
    pub initial_velocity: RandVec3,
    /// Velocity applied to all particles at spawn along radius from system origin
    pub initial_velocity_radial: RandF32,
    /// Whether to match the parent's velocity at spawn
    pub inherit_parent_velocity: bool,
    pub initial_rotation: Quat,
    pub initial_angular_velocity: RandVec3,
}

#[derive(Component, Reflect, Clone, Debug)]
#[require(
    ParticleSpawnerData,
    Visibility,
    Transform,
    Mesh3d,
    NoAutomaticBatching
)]
#[reflect(Component)]
pub struct ParticleSpawner {
    pub particle_settings: Vec<ParticleSettings>,
    pub emission_settings: Vec<EmissionSettings>,
    /// Whether to initialize the spawner in an enabled state
    pub starts_enabled: bool,
    /// Determines how to compute the initial position of the spawned particles
    pub spawn_transform_mode: SpawnTransformMode,
}

impl Default for ParticleSettings {
    fn default() -> Self {
        Self {
            lifetime: RandF32::constant(5.),
            scale_curve: FireworkCurve::constant(1.),
            initial_scale: RandF32::constant(1.),
            acceleration: Vec3::new(0., -9.81, 0.),
            angular_acceleration: Vec3::ZERO,
            linear_drag: 0.2,
            angular_drag: 0.2,
            base_color: FireworkGradient::constant(LinearRgba::WHITE),
            base_color_texture: None,
            emissive_color: FireworkGradient::constant(LinearRgba::BLACK),
            normal_map_texture: None,
            orm_texture: None,
            fade_edge: 0.7,
            fade_scene: 1.,
            blend_mode: BlendMode::Blend,
            pbr: false,
            #[cfg(feature = "physics_avian")]
            collision_settings: None,
        }
    }
}

impl Default for EmissionSettings {
    fn default() -> Self {
        Self {
            particle_index: 0,
            emission_mode: EmissionMode::Rate(5.),
            emission_shape: EmissionShape::Point,
            initial_velocity: RandVec3::constant(Vec3::ZERO),
            initial_velocity_radial: RandF32::constant(0.),
            inherit_parent_velocity: true,
            initial_rotation: Quat::IDENTITY,
            initial_angular_velocity: RandVec3::constant(Vec3::ZERO),
        }
    }
}

impl Default for ParticleSpawner {
    fn default() -> Self {
        Self {
            particle_settings: vec![ParticleSettings::default()],
            emission_settings: vec![EmissionSettings::default()],
            starts_enabled: true,
            spawn_transform_mode: default(),
        }
    }
}

#[cfg(feature = "physics_avian")]
#[derive(Reflect, Clone, Serialize, Deserialize)]
pub struct ParticleCollisionSettings {
    pub restitution: f32,
    pub friction: f32,
    #[reflect(ignore)]
    pub filter: SpatialQueryFilter,
}

#[cfg(feature = "physics_avian")]
impl std::fmt::Debug for ParticleCollisionSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ParticleCollisionSettings {{ elasticity: {:?} }}",
            self.restitution
        )
    }
}

#[derive(Component, Default)]
pub struct ParticleSpawnerData {
    /// Whether this particle system has already been initialized from the settings.
    // NOTE: This won't be needed once we have `Construct`
    pub initialized: bool,
    pub enabled: bool,
    pub cooldown: Vec<Timer>,
    pub particles: Vec<Vec<ParticleData>>,
    pub parent_velocity: Vec3,
    /// Whether we have already sent an event about the particle system having finished
    pub finished_notified: bool,
    /// Number of particles manually queued for creation this frame
    pub manual_queued_count: usize,
}

impl ParticleSpawnerData {
    pub fn queue_particles(&mut self, count: usize) {
        self.manual_queued_count += count;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleData {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
    // Needs to be stored for updating the scale via curves
    pub initial_scale: f32,
    pub scale: f32,
    pub age: f32,
    pub lifetime: f32,
    pub base_color: LinearRgba,
    pub emissive_color: LinearRgba,
    pub pbr: bool,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct EffectModifier {
    pub scale: f32,
    pub speed: f32,
}

impl Default for EffectModifier {
    fn default() -> Self {
        EffectModifier {
            scale: 1.,
            speed: 1.,
        }
    }
}

#[derive(Event)]
pub struct ParticleSpawnerFinished;

pub fn sync_spawner_data(
    mut spawners: Query<(&ParticleSpawner, &mut ParticleSpawnerData), Changed<ParticleSpawner>>,
) {
    for (settings, mut data) in &mut spawners {
        data.cooldown = settings
            .emission_settings
            .iter()
            .map(|emission_settings| {
                if let EmissionMode::Rate(rate) = &emission_settings.emission_mode {
                    Timer::new(Duration::from_secs_f32(1. / rate), TimerMode::Repeating)
                } else {
                    Timer::new(Duration::ZERO, TimerMode::Once)
                }
            })
            .collect();
        data.particles = vec![Vec::new(); settings.particle_settings.len()];
        if !data.initialized {
            data.enabled = settings.starts_enabled;
            data.initialized = true;
        }
    }
}

pub fn spawn_particles(
    mut particle_systems_query: Query<(
        &Transform,
        &GlobalTransform,
        &ParticleSpawner,
        &mut ParticleSpawnerData,
        Option<&EffectModifier>,
    )>,
    time: Res<Time>,
) {
    for (transform, global_transform, settings, mut data, opt_modifier) in
        &mut particle_systems_query
    {
        if data.enabled {
            for i in 0..settings.emission_settings.len() {
                let emission_settings = &settings.emission_settings[i];
                let particle_settings =
                    &settings.particle_settings[emission_settings.particle_index];
                let cooldown = &mut data.cooldown[i];
                cooldown.tick(time.delta());

                let particles_to_spawn = match &emission_settings.emission_mode {
                    EmissionMode::OneShot(count) => {
                        data.enabled = false;
                        *count
                    }
                    EmissionMode::Rate(_) => cooldown.times_finished_this_tick() as usize,
                    EmissionMode::OnDemand => {
                        let count = data.manual_queued_count;
                        data.manual_queued_count = 0;
                        count
                    }
                };

                let modifier = opt_modifier.cloned().unwrap_or_default();

                let origin = match settings.spawn_transform_mode {
                    SpawnTransformMode::Global => global_transform.compute_transform(),
                    SpawnTransformMode::Local => *transform,
                };

                for _ in 0..particles_to_spawn {
                    let spawn_offset = emission_settings.emission_shape.generate_point();

                    let velocity = modifier.speed
                        * (origin.rotation * emission_settings.initial_velocity.generate()
                            + spawn_offset.normalize_or_zero()
                                * emission_settings.initial_velocity_radial.generate())
                        + if emission_settings.inherit_parent_velocity {
                            data.parent_velocity
                        } else {
                            Vec3::ZERO
                        };

                    let initial_scale = particle_settings.initial_scale.generate() * modifier.scale;

                    data.particles[emission_settings.particle_index].push(ParticleData {
                        position: origin.translation + spawn_offset,
                        lifetime: particle_settings.lifetime.generate(),
                        initial_scale,
                        scale: initial_scale,
                        velocity,
                        age: 0.,
                        base_color: particle_settings.base_color.sample_clamped(0.),
                        emissive_color: particle_settings.emissive_color.sample_clamped(0.),
                        pbr: particle_settings.pbr,
                        rotation: emission_settings.initial_rotation,
                        angular_velocity: emission_settings.initial_angular_velocity.generate(),
                    })
                }
            }
        }
    }
}

pub fn update_particles(
    mut particle_systems_query: Query<(&ParticleSpawner, &mut ParticleSpawnerData)>,
    time: Res<Time>,
    #[cfg(feature = "physics_avian")] spatial_query: SpatialQuery,
) {
    particle_systems_query
        .par_iter_mut()
        .for_each(|(settings, mut data)| {
            for i in 0..settings.particle_settings.len() {
                let particle_settings = &settings.particle_settings[i];
                data.particles[i] = data.particles[i]
                    .iter()
                    .filter_map(|particle| {
                        let mut particle = *particle;

                        particle.age += time.delta_secs();
                        if particle.age >= particle.lifetime {
                            return None;
                        }

                        let age_percent = particle.age / particle.lifetime;
                        let scale_factor =
                            particle_settings.scale_curve.sample_clamped(age_percent);

                        particle.scale = particle.initial_scale * scale_factor;

                        #[cfg(feature = "physics_avian")]
                        let (new_pos, new_vel) = if let Some(collision_settigs) =
                            &particle_settings.collision_settings
                        {
                            particle_collision(
                                particle.position,
                                particle.velocity,
                                time.delta_secs(),
                                collision_settigs,
                                &spatial_query,
                            )
                        } else {
                            (
                                particle.position + particle.velocity * time.delta_secs(),
                                particle.velocity,
                            )
                        };
                        #[cfg(not(feature = "physics_avian"))]
                        let (new_pos, new_vel) = (
                            particle.position + particle.velocity * time.delta_secs(),
                            particle.velocity,
                        );

                        particle.position = new_pos;
                        particle.velocity = new_vel;

                        particle.velocity += (particle_settings.acceleration
                            - particle.velocity * particle_settings.linear_drag)
                            * time.delta_secs();

                        particle.rotation =
                            Quat::from_scaled_axis(particle.angular_velocity * time.delta_secs())
                                * particle.rotation;
                        particle.angular_velocity += (particle_settings.angular_acceleration
                            - particle_settings.angular_drag * particle.angular_velocity)
                            * time.delta_secs();

                        particle.base_color =
                            particle_settings.base_color.sample_clamped(age_percent);
                        particle.emissive_color =
                            particle_settings.emissive_color.sample_clamped(age_percent);

                        Some(particle)
                    })
                    .collect();
            }
        });
}

/// System that triggers observers on `ParticleSpawnerFinished` events once when a one-shot
/// particle system has "finished" (spawned all particles and they have all expired).
pub fn notify_finished_particle_spawners(
    mut commands: Commands,
    mut particle_systems_query: Query<(Entity, &ParticleSpawner, &mut ParticleSpawnerData)>,
) {
    for (entity, settings, mut data) in &mut particle_systems_query {
        if data.particles.iter().all(|i| i.is_empty())
            && settings
                .emission_settings
                .iter()
                .all(|e| e.emission_mode.is_one_shot())
            && !data.enabled
            && data.initialized
            && !data.finished_notified
        {
            commands.trigger_targets(ParticleSpawnerFinished, entity);
            data.finished_notified = true;
        }
    }
}

pub fn propagate_particle_spawner_modifier(
    mut commands: Commands,
    modifiers: Query<(Entity, &EffectModifier)>,
    children_query: Query<&Children>,
    particle_spawners: Query<&ParticleSpawner>,
) {
    for (entity, modifier) in &modifiers {
        for child in children_query.iter_descendants(entity) {
            if particle_spawners.contains(child) {
                commands.entity(child).insert(*modifier);
            }
        }
    }
}

pub fn setup_default_mesh(mut meshes: ResMut<Assets<Mesh>>) {
    meshes.insert(
        &mut DEFAULT_MESH.clone(),
        Rectangle::from_size(Vec2::new(1., 1.)).mesh().into(),
    );
}

#[cfg(feature = "physics_avian")]
pub fn sync_parent_velocity(
    velocity: Query<(
        Entity,
        &GlobalTransform,
        &LinearVelocity,
        &AngularVelocity,
        &CenterOfMass,
    )>,
    children_query: Query<&Children>,
    mut spawners: Query<(&GlobalTransform, &mut ParticleSpawnerData)>,
) {
    for (
        parent_entity,
        parent_transform,
        LinearVelocity(linvel),
        AngularVelocity(angvel),
        CenterOfMass(local_center_of_mass),
    ) in &velocity
    {
        for child_entity in children_query.iter_descendants(parent_entity) {
            if let Ok((spawner_transform, mut spawner_data)) = spawners.get_mut(child_entity) {
                spawner_data.parent_velocity = linear_velocity_at_point(
                    *linvel,
                    *angvel,
                    spawner_transform.translation(),
                    parent_transform.transform_point(*local_center_of_mass),
                );
            }
        }
    }
}

#[cfg(feature = "physics_avian")]
/// All quantities are in world-space
fn linear_velocity_at_point(linvel: Vec3, angvel: Vec3, point: Vec3, center_of_mass: Vec3) -> Vec3 {
    linvel + angvel.cross(point - center_of_mass)
}

#[cfg(feature = "physics_avian")]
fn particle_collision(
    mut pos: Vec3,
    mut vel: Vec3,
    mut delta: f32,
    collision_settings: &ParticleCollisionSettings,
    spatial_query: &SpatialQuery,
) -> (Vec3, Vec3) {
    let orig_delta = delta;
    let mut n_steps = 0;
    while delta > 0. && n_steps < 4 {
        if let Some(hit) = spatial_query.cast_ray(
            pos,
            match Dir3::try_from(vel) {
                Ok(dir) => dir,
                Err(_) => Dir3::Y,
            },
            vel.length() * delta,
            true,
            &collision_settings.filter,
        ) {
            if hit.distance == 0. {
                let mut normal = hit.normal;
                if normal == Vec3::ZERO {
                    if vel != Vec3::ZERO {
                        normal = vel.normalize();
                    } else {
                        normal = Vec3::Y;
                    }
                }
                pos += vel.length().max(1.) * normal * delta;
            } else {
                pos += vel.normalize_or_zero() * hit.distance;
                let vel_reject = vel.reject_from(hit.normal);
                let vel_project = vel.project_onto(hit.normal);
                let friction_dv =
                    vel_project.length().min(vel_reject.length()) * collision_settings.friction;
                vel = vel_reject
                    - (friction_dv * vel_reject.normalize_or_zero())
                    - collision_settings.restitution * vel_project;
                pos += hit.normal * 0.0001;
                delta = (delta - hit.distance).clamp(0., orig_delta);
            }
        } else {
            pos += vel * delta;
            delta = 0.;
        }
        n_steps += 1;
    }

    (pos, vel)
}
