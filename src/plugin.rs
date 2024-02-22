use crate::core::setup_default_mesh;

use super::{
    core::{
        create_spawner_data, propagate_particle_spawner_modifier, spawn_particles,
        sync_spawner_data, update_particles, ParticleSpawnerSettings,
    },
    render,
};
use bevy::{asset::load_internal_asset, prelude::*, transform::TransformSystem};

#[cfg(feature = "physics_xpbd")]
use super::core::sync_parent_velocity;

pub const PARTICLE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(272481238906797053434642785120685433641);

pub struct ParticleSystemPlugin;

impl Plugin for ParticleSystemPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PARTICLE_SHADER_HANDLE,
            "particles.wgsl",
            Shader::from_wgsl
        );

        app //
            .register_type::<ParticleSpawnerSettings>()
            .add_plugins(render::CustomMaterialPlugin)
            .add_systems(Startup, setup_default_mesh)
            .add_systems(
                Update,
                (
                    apply_deferred,
                    (create_spawner_data, propagate_particle_spawner_modifier),
                    apply_deferred,
                    sync_spawner_data,
                    #[cfg(feature = "physics_xpbd")]
                    sync_parent_velocity,
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                (spawn_particles, update_particles)
                    .chain()
                    .after(TransformSystem::TransformPropagate),
            );
    }
}
