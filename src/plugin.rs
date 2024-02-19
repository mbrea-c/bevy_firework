use super::{
    core::{
        create_spawner_data, propagate_particle_spawner_modifier, spawn_particles,
        sync_spawner_data, update_particles, ParticleSpawnerSettings,
    },
    render,
};
use bevy::prelude::*;

#[cfg(feature = "physics_xpbd")]
use super::core::sync_parent_velocity;

pub struct ParticleSystemPlugin;

impl Plugin for ParticleSystemPlugin {
    fn build(&self, app: &mut App) {
        app //
            .register_type::<ParticleSpawnerSettings>()
            .add_plugins(render::CustomMaterialPlugin)
            .add_systems(
                Update,
                (
                    apply_deferred,
                    (create_spawner_data, propagate_particle_spawner_modifier),
                    apply_deferred,
                    sync_spawner_data,
                    #[cfg(feature = "physics_xpbd")]
                    sync_parent_velocity,
                    spawn_particles,
                    update_particles,
                )
                    .chain(),
            );
    }
}
