use super::emission_shape::EmissionShape;
use bevy::{prelude::*, render::batching::NoAutomaticBatching};
use bevy_utilitarian::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "physics_xpbd")]
use bevy_xpbd_3d::prelude::*;

pub const DEFAULT_MESH: Handle<Mesh> =
    Handle::weak_from_u128(164408926256276437310893021157813788765);

/// Mirrors AlphaMode, but implements serialize and deserialize
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub enum BlendMode {
    Opaque,
    Blend,
    Premultiplied,
    Add,
    Multiply,
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

#[derive(Component, Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(Component)]
pub struct ParticleSpawnerSettings {
    /// Particles per second
    pub rate: f32,
    /// Whether to spawn `rate` particles at once and then stop
    pub one_shot: bool,
    /// Shape on which to spawn particles
    pub emission_shape: EmissionShape,
    /// Lifetime of a particle
    pub lifetime: RandF32,
    /// Linear velocity applied to all particles at spawn
    pub initial_velocity: RandVec3,
    /// Velocity applied to all particles at spawn along radius from system origin
    pub initial_velocity_radial: RandF32,
    /// Whether to match the parent's velocity at spawn
    pub inherit_parent_velocity: bool,
    /// Initial scale of particles
    pub initial_scale: RandF32,
    /// Evolution of scale over time, applied as a factor to initial scale
    pub scale_curve: ParamCurve<f32>,
    /// Linear acceleration of particles
    pub acceleration: Vec3,
    /// Drag applied as a linear coefficient of velocity
    pub linear_drag: f32,
    /// Color over lifetime
    pub color: Gradient,
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
    #[cfg(feature = "physics_xpbd")]
    /// If Some, particles will collide with the scene according to the provided parameters
    /// If None, no particle collision will occur.
    pub collision_settings: Option<ParticleCollisionSettings>,
}

impl Default for ParticleSpawnerSettings {
    fn default() -> Self {
        Self {
            rate: 5.,
            one_shot: false,
            emission_shape: EmissionShape::Point,
            lifetime: RandF32::constant(5.),
            initial_velocity: RandVec3::constant(Vec3::ZERO),
            initial_velocity_radial: RandF32::constant(0.),
            inherit_parent_velocity: true,
            initial_scale: RandF32::constant(1.),
            scale_curve: ParamCurve::linear_uniform(vec![1., 1.]),
            acceleration: Vec3::new(0., -9.81, 0.),
            color: Gradient::constant(Color::WHITE.into()),
            blend_mode: BlendMode::Blend,
            linear_drag: 0.,
            pbr: false,
            #[cfg(feature = "physics_xpbd")]
            collision_settings: None,
            fade_edge: 0.7,
            fade_scene: 1.,
        }
    }
}

#[cfg(feature = "physics_xpbd")]
#[derive(Reflect, Clone, Serialize, Deserialize)]
pub struct ParticleCollisionSettings {
    pub restitution: f32,
    pub friction: f32,
    #[reflect(ignore)]
    pub filter: SpatialQueryFilter,
}

#[cfg(feature = "physics_xpbd")]
impl std::fmt::Debug for ParticleCollisionSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ParticleCollisionSettings {{ elasticity: {:?} }}",
            self.restitution
        )
    }
}

#[derive(Component)]
pub struct ParticleSpawnerData {
    pub enabled: bool,
    pub cooldown: Timer,
    pub particles: Vec<ParticleData>,
    pub parent_velocity: Vec3,
}

impl Default for ParticleSpawnerData {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown: Timer::default(),
            particles: vec![],
            parent_velocity: Vec3::ZERO,
        }
    }
}

impl From<&ParticleSpawnerSettings> for ParticleSpawnerData {
    fn from(settings: &ParticleSpawnerSettings) -> Self {
        Self {
            enabled: true,
            cooldown: Timer::from_seconds(1. / settings.rate, TimerMode::Repeating),
            particles: vec![],
            parent_velocity: Vec3::ZERO,
        }
    }
}

#[derive(Bundle)]
pub struct ParticleSpawnerBundle {
    spatial: SpatialBundle,
    settings: ParticleSpawnerSettings,
    mesh: Handle<Mesh>,
    name: Name,
}

impl ParticleSpawnerBundle {
    pub fn from_settings(settings: ParticleSpawnerSettings) -> Self {
        Self {
            settings,
            spatial: SpatialBundle::default(),
            mesh: DEFAULT_MESH.clone(),
            name: Name::new("Particle System"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleData {
    pub position: Vec3,
    pub velocity: Vec3,
    // Needs to be stored for updating the scale via curves
    pub initial_scale: f32,
    pub scale: f32,
    pub age: f32,
    pub lifetime: f32,
    pub color: Color,
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

pub fn create_spawner_data(
    mut commands: Commands,
    mut spawners: Query<(Entity, &ParticleSpawnerSettings), Without<ParticleSpawnerData>>,
) {
    for (entity, settings) in &mut spawners {
        commands
            .entity(entity)
            .insert(ParticleSpawnerData::from(settings))
            .insert(NoAutomaticBatching);
    }
}

pub fn sync_spawner_data(
    mut spawners: Query<
        (&ParticleSpawnerSettings, &mut ParticleSpawnerData),
        Changed<ParticleSpawnerSettings>,
    >,
) {
    for (settings, mut data) in &mut spawners {
        data.cooldown
            .set_duration(Duration::from_secs_f32(1. / settings.rate));
        data.cooldown.set_mode(TimerMode::Repeating);
    }
}

pub fn spawn_particles(
    mut particle_systems_query: Query<(
        &GlobalTransform,
        &ParticleSpawnerSettings,
        &mut ParticleSpawnerData,
        Option<&EffectModifier>,
    )>,
    time: Res<Time>,
) {
    for (global_transform, settings, mut data, opt_modifier) in &mut particle_systems_query {
        if data.enabled {
            data.cooldown.tick(time.delta());

            let particles_to_spawn = if settings.one_shot {
                data.enabled = false;

                settings.rate as u32
            } else {
                data.cooldown.times_finished_this_tick()
            };

            let modifier = opt_modifier.cloned().unwrap_or_default();
            let origin = global_transform.compute_transform();

            for _ in 0..particles_to_spawn {
                let spawn_offset = settings.emission_shape.generate_point();

                let velocity = modifier.speed
                    * (origin.rotation * settings.initial_velocity.generate()
                        + spawn_offset.normalize_or_zero()
                            * settings.initial_velocity_radial.generate())
                    + if settings.inherit_parent_velocity {
                        data.parent_velocity
                    } else {
                        Vec3::ZERO
                    };

                let initial_scale = settings.initial_scale.generate() * modifier.scale;

                data.particles.push(ParticleData {
                    position: origin.translation + spawn_offset,
                    lifetime: settings.lifetime.generate(),
                    initial_scale,
                    scale: initial_scale,
                    velocity,
                    age: 0.,
                    color: *settings.color.get(0.),
                    pbr: settings.pbr,
                })
            }
        }
    }
}

pub fn update_particles(
    mut particle_systems_query: Query<(&ParticleSpawnerSettings, &mut ParticleSpawnerData)>,
    time: Res<Time>,
    #[cfg(feature = "physics_xpbd")] spatial_query: SpatialQuery,
) {
    particle_systems_query
        .par_iter_mut()
        .for_each(|(settings, mut data)| {
            data.particles = data
                .particles
                .iter()
                .filter_map(|particle| {
                    let mut particle = *particle;

                    particle.age += time.delta_seconds();
                    if particle.age >= particle.lifetime {
                        return None;
                    }

                    let age_percent = particle.age / particle.lifetime;
                    let scale_factor = settings.scale_curve.get(age_percent);

                    particle.scale = particle.initial_scale * scale_factor;

                    #[cfg(feature = "physics_xpbd")]
                    let (new_pos, new_vel) =
                        if let Some(collision_settigs) = &settings.collision_settings {
                            particle_collision(
                                particle.position,
                                particle.velocity,
                                time.delta_seconds(),
                                collision_settigs,
                                &spatial_query,
                            )
                        } else {
                            (
                                particle.position + particle.velocity * time.delta_seconds(),
                                particle.velocity,
                            )
                        };
                    #[cfg(not(feature = "physics_xpbd"))]
                    let (new_pos, new_vel) = (
                        particle.position + particle.velocity * time.delta_seconds(),
                        particle.velocity,
                    );

                    particle.position = new_pos;
                    particle.velocity = new_vel;
                    particle.velocity += (settings.acceleration
                        - particle.velocity * settings.linear_drag)
                        * time.delta_seconds();
                    particle.color = *settings.color.get(age_percent);

                    Some(particle)
                })
                .collect();
        });
}

pub fn propagate_particle_spawner_modifier(
    mut commands: Commands,
    modifiers: Query<(Entity, &EffectModifier)>,
    children_query: Query<&Children>,
    particle_spawners: Query<&ParticleSpawnerSettings>,
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
        DEFAULT_MESH.clone(),
        Mesh::from(shape::Quad::new(Vec2::new(1., 1.))),
    );
}

#[cfg(feature = "physics_xpbd")]
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

#[cfg(feature = "physics_xpbd")]
/// All quantities are in world-space
fn linear_velocity_at_point(linvel: Vec3, angvel: Vec3, point: Vec3, center_of_mass: Vec3) -> Vec3 {
    linvel + angvel.cross(point - center_of_mass)
}

#[cfg(feature = "physics_xpbd")]
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
            vel.try_normalize().unwrap_or(Vec3::Y),
            vel.length() * delta,
            true,
            collision_settings.filter.clone(),
        ) {
            if hit.time_of_impact == 0. {
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
                pos += vel.normalize_or_zero() * hit.time_of_impact;
                let vel_reject = vel.reject_from(hit.normal);
                let vel_project = vel.project_onto(hit.normal);
                let friction_dv =
                    vel_project.length().min(vel_reject.length()) * collision_settings.friction;
                vel = vel_reject
                    - (friction_dv * vel_reject.normalize_or_zero())
                    - collision_settings.restitution * vel_project;
                pos += hit.normal * 0.0001;
                delta = (delta - hit.time_of_impact).clamp(0., orig_delta);
            }
        } else {
            pos += vel * delta;
            delta = 0.;
        }
        n_steps += 1;
    }

    (pos, vel)
}
