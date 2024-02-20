use crate::plugin::PARTICLE_SHADER_HANDLE;

use super::core::{ParticleData, ParticleSpawnerData, ParticleSpawnerSettings};
use bevy::{
    core_pipeline::{
        core_3d::Transparent3d,
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::MeshVertexBufferLayout,
        primitives::Aabb,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::ExtractedView,
        Render, RenderApp, RenderSet,
    },
};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ParticleInstance {
    position: Vec3,
    scale: f32,
    color: [f32; 4],
    pbr: u32,
}

impl From<&ParticleData> for ParticleInstance {
    fn from(value: &ParticleData) -> Self {
        Self {
            position: value.position,
            scale: value.scale,
            color: value.color.into(),
            pbr: value.pbr as u32,
        }
    }
}

#[derive(Component, Deref)]
pub struct ParticleMaterialData {
    #[deref]
    particles: Vec<ParticleInstance>,
    alpha_mode: AlphaMode,
}

impl ExtractComponent for ParticleSpawnerData {
    type Query = (
        &'static ParticleSpawnerData,
        &'static ParticleSpawnerSettings,
    );
    type Filter = ();
    type Out = ParticleMaterialData;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self::Out> {
        let (data, settings) = item;
        Some(ParticleMaterialData {
            particles: data.particles.iter().map(|p| p.into()).collect(),
            alpha_mode: settings.blend_mode.into(),
        })
    }
}

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app //
            .add_plugins(ExtractComponentPlugin::<ParticleSpawnerData>::default())
            .add_systems(Last, update_aabbs);
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSet::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSet::PrepareResources),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).init_resource::<CustomPipeline>();
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<Mesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<(Entity, &ParticleMaterialData)>,
    mut views: Query<(
        &ExtractedView,
        &mut RenderPhase<Transparent3d>,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
    )>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (
        view,
        mut transparent_phase,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, particle_material_data) in &material_meshes {
            let Some(mesh_instance) = render_mesh_instances.get(&entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let mut key =
                view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
            match particle_material_data.alpha_mode {
                AlphaMode::Blend => {
                    key |= MeshPipelineKey::BLEND_ALPHA;
                }
                AlphaMode::Premultiplied | AlphaMode::Add => {
                    // Premultiplied and Add share the same pipeline key
                    // They're made distinct in the PBR shader, via `premultiply_alpha()`
                    key |= MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA;
                }
                AlphaMode::Multiply => {
                    key |= MeshPipelineKey::BLEND_MULTIPLY;
                }
                AlphaMode::Mask(_) => {
                    key |= MeshPipelineKey::MAY_DISCARD;
                }
                _ => (),
            };

            if normal_prepass {
                key |= MeshPipelineKey::NORMAL_PREPASS;
            }

            if depth_prepass {
                key |= MeshPipelineKey::DEPTH_PREPASS;
            }

            if motion_vector_prepass {
                key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
            }

            if deferred_prepass {
                key |= MeshPipelineKey::DEFERRED_PREPASS;
            }

            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
            transparent_phase.add(Transparent3d {
                entity,
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder
                    .distance_translation(&mesh_instance.transforms.transform.translation),
                batch_range: 0..1,
                dynamic_offset: None,
            });
        }
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &ParticleMaterialData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.particles.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

fn update_aabbs(mut query: Query<(&mut Aabb, &GlobalTransform, &ParticleSpawnerData)>) {
    for (mut aabb, global_transform, spawner_data) in &mut query {
        if spawner_data.particles.is_empty() {
            continue;
        }
        let min = spawner_data
            .particles
            .iter()
            .map(|p| p.position - Vec3::splat(p.scale))
            .fold(Vec3::MAX, |acc, v| acc.min(v));
        let max = spawner_data
            .particles
            .iter()
            .map(|p| p.position + Vec3::splat(p.scale))
            .fold(Vec3::MIN, |acc, v| acc.max(v));
        let center = (min + max) / 2.;
        let half_extents = (max - min) / 2.;
        aabb.center = global_transform
            .compute_matrix()
            .inverse()
            .transform_point3(center)
            .into();
        aabb.half_extents = half_extents.into();
    }
}

#[derive(Resource)]
pub struct CustomPipeline {
    vertex_shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for CustomPipeline {
    fn from_world(world: &mut World) -> Self {
        let vertex_shader = PARTICLE_SHADER_HANDLE;
        let mesh_pipeline = world.resource::<MeshPipeline>();

        CustomPipeline {
            vertex_shader,
            mesh_pipeline: mesh_pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // meshes typically live in bind group 2. because we are using bindgroup 1
        // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
        // linked in the shader
        descriptor
            .vertex
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor
            .fragment
            .as_mut()
            .unwrap()
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor.depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: false,
            // Bevy uses reverse-Z, so Greater really means closer
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });

        descriptor.vertex.shader = self.vertex_shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
                VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: VertexFormat::Float32x4.size() * 2,
                    shader_location: 5,
                },
            ],
        });
        let fragment = descriptor.fragment.as_mut().unwrap();
        fragment.shader = self.vertex_shader.clone();

        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMeshInstanced,
);

pub struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (SRes<RenderAssets<Mesh>>, SRes<RenderMeshInstances>);
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: &'w InstanceBuffer,
        (meshes, render_mesh_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(mesh_instance) = render_mesh_instances.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Failure;
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        pass.draw(0..6, 0..instance_buffer.length as u32);
        RenderCommandResult::Success
    }
}
