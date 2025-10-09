use crate::core::{notify_finished_particle_spawners, setup_default_mesh};

use super::{
    core::{
        ParticleSpawner, propagate_particle_spawner_modifier, spawn_particles, sync_spawner_data,
        update_particles,
    },
    render,
};
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    ecs::{intern::Interned, schedule::ScheduleLabel},
    prelude::*,
};

#[cfg(feature = "physics_avian")]
use super::core::sync_parent_velocity;

pub const PARTICLE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("bdd76c4f-ce8a-45e4-a8d9-7f0470bed2f8");

pub struct ParticleSystemPlugin {
    pub update_schedule: Interned<dyn ScheduleLabel>,
}

impl Default for ParticleSystemPlugin {
    fn default() -> Self {
        Self {
            update_schedule: Update.intern(),
        }
    }
}

impl Plugin for ParticleSystemPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PARTICLE_SHADER_HANDLE,
            "particles.wgsl",
            Shader::from_wgsl
        );

        app //
            .register_type::<ParticleSpawner>()
            .add_plugins(render::CustomMaterialPlugin)
            .add_systems(Startup, setup_default_mesh)
            .add_systems(
                self.update_schedule,
                (
                    ApplyDeferred,
                    propagate_particle_spawner_modifier,
                    ApplyDeferred,
                    sync_spawner_data,
                    #[cfg(feature = "physics_avian")]
                    sync_parent_velocity,
                    spawn_particles,
                    update_particles,
                    notify_finished_particle_spawners,
                )
                    .chain(),
            );
    }
}
