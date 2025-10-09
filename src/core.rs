use crate::curve::{FireworkCurve, FireworkGradient};

use super::emission_shape::EmissionShape;
use bevy::{
    asset::uuid_handle, ecs::system::SystemId, prelude::*, render::batching::NoAutomaticBatching,
};
use bevy_utilitarian::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "physics_avian")]
use avian3d::prelude::*;

pub const DEFAULT_MESH: Handle<Mesh> = uuid_handle!("ba671aee-04f4-485d-9d1e-ad7053dacfab");

#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub enum EmissionPacing {
    /// Number of particles emitted at once
    OneShot(usize),
    /// Particles get emitted on method calls
    OnDemand,
    /// Spawn a given number of particles over the specified time period
    CountOverDuration {
        /// Number of particles that will be spawned in the given duration
        count: f32,
        /// Duration of spawn cycle. If emission mode is nested, this will be ignored, and duration
        /// will be the lifetime of the target particle
        duration: f32,
        /// Percentage of duration when particle spawning begins
        offset_start: f32,
        /// Percentage of duration when particle spawning ends
        offset_end: f32,
    },
}

impl EmissionPacing {
    pub fn is_one_shot(&self) -> bool {
        matches!(self, EmissionPacing::OneShot(_))
    }

    pub fn rate(rate: f32) -> Self {
        Self::CountOverDuration {
            count: rate,
            duration: 1.,
            offset_start: 0.,
            offset_end: 1.,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub enum EmissionMode {
    #[default]
    Global,
    Nested {
        /// The particle settings index of the particle we will be spawning under
        target_particle_type: usize,
    },
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
    #[reflect(ignore)]
    pub event_handlers: ParticleEventHandlers,
}

#[derive(Reflect, Clone, Debug)]
pub struct EmissionSettings {
    /// Which particle settings to use, as an index to the particle settings vector
    pub particle_index: usize,
    /// Determines the way this spawner creates particles over time
    pub emission_pacing: EmissionPacing,
    /// Determines where to spawn the particle and what velocity/rotation/position to inherit
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

#[derive(Clone, Debug, Default)]
pub struct ParticleEventHandlers {
    pub particles_destroyed: Option<SystemId<In<Vec<ParticleData>>, ()>>,
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
            event_handlers: default(),
        }
    }
}

impl Default for EmissionSettings {
    fn default() -> Self {
        Self {
            particle_index: 0,
            emission_mode: EmissionMode::default(),
            emission_pacing: EmissionPacing::rate(5.),
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

pub struct EmissionData {
    last_emission: f32,
    time_passed_in_cycle: f32,
    enabled: bool,
    /// True iff this emitter only emits on existing particles (e.g. [`EmissionMode::Nested`]).
    emits_on_other_particles: bool,
}

#[derive(Component, Default)]
pub struct ParticleSpawnerData {
    /// Whether this particle system has already been initialized from the settings.
    // NOTE: This won't be needed once we have `Construct`
    pub initialized: bool,
    pub particles: Vec<Vec<ParticleData>>,
    pub emission: Vec<EmissionData>,
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

    pub fn active(&self) -> bool {
        let mut enabled = false;

        for emission in &self.emission {
            if emission.emits_on_other_particles {
                let does_any_particle_type_have_particles =
                    self.particles.iter().any(|particles| !particles.is_empty());
                enabled |= emission.enabled && does_any_particle_type_have_particles;
            } else {
                enabled |= emission.enabled;
            }
        }

        enabled
    }
}

#[derive(Clone, Debug)]
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
    /// Age when the last subparticle was emitted
    pub last_emitted_age: Vec<f32>,
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

#[derive(EntityEvent)]
pub struct ParticleSpawnerFinished {
    pub entity: Entity,
}

pub fn sync_spawner_data(
    mut spawners: Query<(&ParticleSpawner, &mut ParticleSpawnerData), Changed<ParticleSpawner>>,
) {
    for (settings, mut data) in &mut spawners {
        data.emission = settings
            .emission_settings
            .iter()
            .map(|emission_settings| EmissionData {
                last_emission: 0.,
                time_passed_in_cycle: 0.,
                enabled: settings.starts_enabled,
                emits_on_other_particles: match emission_settings.emission_mode {
                    EmissionMode::Global => false,
                    EmissionMode::Nested { .. } => true,
                },
            })
            .collect();
        data.particles = vec![Vec::new(); settings.particle_settings.len()];
        if !data.initialized {
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
    for (transform, global_transform, settings, data, opt_modifier) in &mut particle_systems_query {
        if data.active() {
            let ParticleSpawnerData {
                particles,
                emission,
                parent_velocity,
                manual_queued_count,
                ..
            } = data.into_inner();
            for (i, emission_settings) in settings.emission_settings.iter().enumerate() {
                let emission_data = &mut emission[i];
                if !emission_data.enabled {
                    continue;
                }
                let particle_settings =
                    &settings.particle_settings[emission_settings.particle_index];

                match emission_settings.emission_mode {
                    EmissionMode::Global => {
                        let particles_to_spawn = match &emission_settings.emission_pacing {
                            EmissionPacing::OneShot(count) => {
                                emission_data.enabled = false;
                                *count
                            }
                            EmissionPacing::OnDemand => {
                                let count = *manual_queued_count;
                                *manual_queued_count = 0;
                                count
                            }
                            EmissionPacing::CountOverDuration {
                                count,
                                duration,
                                offset_start,
                                offset_end,
                            } => {
                                emission_data.time_passed_in_cycle =
                                    (emission_data.time_passed_in_cycle + time.delta_secs())
                                        .rem_euclid(*duration);
                                let (particles_to_spawn, next_last_emission) =
                                    compute_emission_count(
                                        emission_data.time_passed_in_cycle,
                                        emission_data.last_emission,
                                        *duration,
                                        *offset_start,
                                        *offset_end,
                                        *count,
                                    );
                                emission_data.last_emission = next_last_emission;

                                particles_to_spawn
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
                                    *parent_velocity
                                } else {
                                    Vec3::ZERO
                                };

                            let initial_scale =
                                particle_settings.initial_scale.generate() * modifier.scale;

                            particles[emission_settings.particle_index].push(ParticleData {
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
                                angular_velocity: emission_settings
                                    .initial_angular_velocity
                                    .generate(),
                                last_emitted_age: vec![f32::MIN; settings.emission_settings.len()],
                            })
                        }
                    }
                    EmissionMode::Nested {
                        target_particle_type,
                    } => {
                        let EmissionPacing::CountOverDuration {
                            count,
                            offset_start,
                            offset_end,
                            ..
                        } = &emission_settings.emission_pacing
                        else {
                            warn_once!(
                                "Only `CountOverDuration` emission pacing allowed in combination with `Nested` emission mode"
                            );
                            continue;
                        };
                        let modifier = opt_modifier.cloned().unwrap_or_default();

                        for p_i in 0..particles[target_particle_type].len() {
                            let other_particle = &mut particles[target_particle_type][p_i];
                            let (times_needed_to_emit_usize, next_last_emitted_age) =
                                compute_emission_count(
                                    other_particle.age,
                                    other_particle.last_emitted_age[i],
                                    other_particle.lifetime,
                                    *offset_start,
                                    *offset_end,
                                    *count,
                                );

                            other_particle.last_emitted_age[i] = next_last_emitted_age;

                            let origin_position = other_particle.position;
                            let origin_rotation = other_particle.rotation;
                            let origin_velocity = other_particle.velocity;

                            for _ in 0..times_needed_to_emit_usize {
                                let spawn_offset =
                                    emission_settings.emission_shape.generate_point();
                                let velocity = modifier.speed
                                    * (origin_rotation
                                        * emission_settings.initial_velocity.generate()
                                        + spawn_offset.normalize_or_zero()
                                            * emission_settings.initial_velocity_radial.generate())
                                    + if emission_settings.inherit_parent_velocity {
                                        origin_velocity
                                    } else {
                                        Vec3::ZERO
                                    };
                                let initial_scale =
                                    particle_settings.initial_scale.generate() * modifier.scale;
                                let initial_position = origin_position + spawn_offset;

                                particles[emission_settings.particle_index].push(ParticleData {
                                    position: initial_position,
                                    lifetime: particle_settings.lifetime.generate(),
                                    initial_scale,
                                    scale: initial_scale,
                                    velocity,
                                    age: 0.,
                                    base_color: particle_settings.base_color.sample_clamped(0.),
                                    emissive_color: particle_settings
                                        .emissive_color
                                        .sample_clamped(0.),
                                    pbr: particle_settings.pbr,
                                    rotation: emission_settings.initial_rotation,
                                    angular_velocity: emission_settings
                                        .initial_angular_velocity
                                        .generate(),
                                    last_emitted_age: vec![
                                        f32::MIN;
                                        settings.emission_settings.len()
                                    ],
                                })
                            }
                        }
                    }
                }
            }
        }
    }
}

fn compute_emission_count(
    time_passed_in_cycle: f32,
    last_emission: f32,
    cycle_duration: f32,
    // as a percentage
    emission_offset_start: f32,
    // as a percentage
    emission_offset_end: f32,
    particles_per_cycle: f32,
) -> (usize, f32) {
    let percent_passed = time_passed_in_cycle / cycle_duration;
    let last_emission_percent = last_emission / cycle_duration;
    let percent_passed_since_emission =
        percent_passed.min(emission_offset_end) - last_emission_percent.max(emission_offset_start);
    let percent_between_emissions =
        (emission_offset_end - emission_offset_start) / particles_per_cycle;
    let times_needed_to_emit = percent_passed_since_emission.div_euclid(percent_between_emissions);
    let times_needed_to_emit_usize = times_needed_to_emit as usize;
    let next_last_emission_percent = last_emission_percent.max(emission_offset_start)
        + times_needed_to_emit * percent_between_emissions;
    let next_last_emission = next_last_emission_percent * cycle_duration;
    (times_needed_to_emit_usize, next_last_emission)
}

pub fn update_particles(
    mut particle_systems_query: Query<(&ParticleSpawner, &mut ParticleSpawnerData)>,
    time: Res<Time>,
    commands: ParallelCommands,
    #[cfg(feature = "physics_avian")] spatial_query: SpatialQuery,
) {
    particle_systems_query
        .par_iter_mut()
        .for_each(|(settings, mut data)| {
            for i in 0..settings.particle_settings.len() {
                let particle_settings = &settings.particle_settings[i];
                let mut destroyed = vec![];
                data.particles[i] = data.particles[i]
                    .iter()
                    .filter_map(|particle| {
                        let mut particle = particle.clone();

                        particle.age += time.delta_secs();

                        if particle.age >= particle.lifetime {
                            destroyed.push(particle);
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
                if !destroyed.is_empty()
                    && let Some(destroyed_handler) =
                        &particle_settings.event_handlers.particles_destroyed
                {
                    commands.command_scope(|mut cmds| {
                        cmds.run_system_with(*destroyed_handler, destroyed);
                    })
                }
            }
        });
}

/// System that triggers observers on `ParticleSpawnerFinished` events once when a one-shot
/// particle system has "finished" (spawned all particles and they have all expired).
pub fn notify_finished_particle_spawners(
    mut commands: Commands,
    mut particle_systems_query: Query<(Entity, &mut ParticleSpawnerData)>,
) {
    for (entity, mut data) in &mut particle_systems_query {
        if data.particles.iter().all(|i| i.is_empty())
            && !data.active()
            && data.initialized
            && !data.finished_notified
        {
            commands.trigger(ParticleSpawnerFinished { entity });
            // commands.trigger_targets(ParticleSpawnerFinished, entity);
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
    let _ = meshes.insert(
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

#[cfg(test)]
mod tests {
    use crate::core::compute_emission_count;

    #[test]
    fn test_compute_emission_count() {
        let timestep = 0.016;
        let mut age = 0.;
        let mut last_emission = f32::MIN;
        let duration = 3.;
        let particles_per_duration = 23.;

        let mut particles_so_far = 0;

        while age <= duration {
            let (emit_particles, new_last_emitted) = compute_emission_count(
                age,
                last_emission,
                duration,
                0.,
                1.,
                particles_per_duration,
            );
            particles_so_far += emit_particles;
            last_emission = new_last_emitted;
            age += timestep;
        }

        assert!(
            particles_so_far == particles_per_duration as usize
                || particles_so_far == (particles_per_duration as usize - 1)
        );
    }
}
