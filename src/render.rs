use crate::plugin::PARTICLE_SHADER_HANDLE;

use super::core::{ParticleData, ParticleSpawner, ParticleSpawnerData};
use bevy::{
    core_pipeline::{
        core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d},
        prepass::{
            DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass, ViewPrepassTextures,
        },
    },
    ecs::system::{SystemParamItem, lifetimeless::*},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshViewBindGroup,
        ShadowFilteringMethod,
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSet,
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        primitives::Aabb,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::MainEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};
use bytemuck::{Pod, Zeroable};

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app //
            .add_plugins(UniformComponentPlugin::<FireworkUniform>::default())
            .add_systems(Last, update_aabbs);

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedRenderPipelines<FireworkPipeline>>()
            .add_systems(
                ExtractSchedule,
                (cleanup_firework_components, extract_firework_components).chain(),
            )
            .add_systems(
                Render,
                (
                    ensure_dummy_textures_exist,
                    (
                        queue_custom.in_set(RenderSet::QueueMeshes),
                        prepare_instance_buffers.in_set(RenderSet::PrepareResources),
                        prepare_firework_bindgroup.in_set(RenderSet::PrepareBindGroups),
                    ),
                )
                    .chain(),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app.insert_resource(FireworkUniformBindgroupLayouts::default());
        render_app.insert_resource(FireworkPipelines::default());
    }
}

fn ensure_dummy_textures_exist(
    dummy_textures: Option<Res<DummyTextures>>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    if dummy_textures.is_none() {
        commands.insert_resource(DummyTextures::new(&render_device));
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ParticleInstance {
    position: Vec3,
    scale: f32,
    rotation: Quat,
    base_color: [f32; 4],
    emissive_color: [f32; 4],
}

impl From<&ParticleData> for ParticleInstance {
    fn from(value: &ParticleData) -> Self {
        Self {
            position: value.position,
            scale: value.scale,
            rotation: value.rotation,
            base_color: value.base_color.to_f32_array(),
            emissive_color: value.emissive_color.to_f32_array(),
        }
    }
}

#[derive(Component, Deref)]
pub struct ParticleMaterialData {
    #[deref]
    particles: Vec<ParticleInstance>,
    alpha_mode: AlphaMode,
}

#[derive(Component)]
pub struct FireworkImages {
    base_color_texture: Option<Handle<Image>>,
    normal_map_texture: Option<Handle<Image>>,
    orm_texture: Option<Handle<Image>>,
}

#[derive(Component)]
pub struct FireworkRenderEntityMarker;

#[derive(Resource)]
pub struct DummyTextures {
    /// A dummy texture per sample count setting
    pub depth_textures: HashMap<u32, TextureView>,
    pub base_color_texture_sampler: Sampler,
    pub base_color_texture: TextureView,
    pub normal_map_texture_sampler: Sampler,
    pub normal_map_texture: TextureView,
    pub orm_texture_sampler: Sampler,
    pub orm_texture: TextureView,
}

impl DummyTextures {
    pub fn new(render_device: &RenderDevice) -> Self {
        let make_color_texture_and_sampler = |label_texture, label_sampler| {
            let base_color_texture = render_device
                .create_texture(&TextureDescriptor {
                    label: Some(label_texture),
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba32Float,
                    usage: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    usage: None,
                });

            let base_color_texture_sampler = render_device.create_sampler(&SamplerDescriptor {
                label: Some(label_sampler),
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                ..Default::default()
            });

            (base_color_texture, base_color_texture_sampler)
        };
        let (base_color_texture, base_color_texture_sampler) =
            make_color_texture_and_sampler("Dummy Base Color Texture", "Base Color Sampler");
        let (normal_map_texture, normal_map_texture_sampler) =
            make_color_texture_and_sampler("Dummy Normal Map Texture", "Normal Map Sampler");
        let (orm_texture, orm_texture_sampler) =
            make_color_texture_and_sampler("Dummy Normal Map Texture", "Normal Map Sampler");

        Self {
            depth_textures: HashMap::new(),
            base_color_texture_sampler,
            base_color_texture,
            normal_map_texture_sampler,
            normal_map_texture,
            orm_texture_sampler,
            orm_texture,
        }
    }

    pub fn ensure_has_samples(&mut self, sample_count: u32, render_device: &RenderDevice) {
        if !self.depth_textures.contains_key(&sample_count) {
            let texture = render_device
                .create_texture(&TextureDescriptor {
                    label: Some("Dummy Depth Texture"),
                    size: Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Depth32Float,
                    usage: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    usage: None,
                });
            self.depth_textures.insert(sample_count, texture.clone());
        }
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

#[derive(Resource, Default)]
pub struct FireworkUniformBindgroupLayouts {
    pub layouts: HashMap<u32, BindGroupLayout>,
}

impl FireworkUniformBindgroupLayouts {
    pub fn ensure_created(&mut self, render_device: &RenderDevice, msaa_samples: u32) {
        let layout = render_device.create_bind_group_layout(
            Some("Firework Uniform Layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(FireworkUniform::min_size()),
                    },
                    count: None,
                },
                // The depth texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: msaa_samples > 1,
                    },
                    count: None,
                },
                // base color texture sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // base color texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // normal map texture sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // normal map texture
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // orm texture sampler
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // orm texture
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        );

        self.layouts.insert(msaa_samples, layout);
    }

    pub fn get(&self, msaa_samples: u32) -> &BindGroupLayout {
        self.layouts.get(&msaa_samples).unwrap()
    }
}

#[derive(Component)]
pub struct FireworkUniformBindgroups {
    bindgroups: HashMap<Entity, BindGroup>,
}

#[allow(dead_code)]
#[derive(Component, ShaderType, Clone, Debug)]
pub struct FireworkUniform {
    alpha_mode: u32,
    pbr: u32,
    fade_edge: f32,
    fade_scene: f32,
    flags: u32,
    __padding: Vec3,
}

pub const FIREWORK_BASE_COLOR_TEXTURE_BIT: u32 = 1;
pub const FIREWORK_NORMAL_MAP_TEXTURE_BIT: u32 = 1 << 1;
pub const FIREWORK_ORM_TEXTURE_BIT: u32 = 1 << 2;

fn extract_component(
    item: (&ParticleSpawnerData, &ParticleSpawner),
) -> Vec<(
    FireworkRenderEntityMarker,
    ParticleMaterialData,
    FireworkUniform,
    FireworkImages,
)> {
    let (data, settings) = item;
    data.particles
        .iter()
        .enumerate()
        .filter(|(_, particles)| !particles.is_empty())
        .map(|(index, particles)| {
            let particle_settings = &settings.particle_settings[index];

            let mut flags = 0;
            if particle_settings.base_color_texture.is_some() {
                flags |= FIREWORK_BASE_COLOR_TEXTURE_BIT;
            }
            if particle_settings.normal_map_texture.is_some() {
                flags |= FIREWORK_NORMAL_MAP_TEXTURE_BIT;
            }
            if particle_settings.orm_texture.is_some() {
                flags |= FIREWORK_ORM_TEXTURE_BIT;
            }

            (
                FireworkRenderEntityMarker,
                ParticleMaterialData {
                    particles: particles.iter().map(|p| p.into()).collect(),
                    alpha_mode: particle_settings.blend_mode.into(),
                },
                FireworkUniform {
                    alpha_mode: particle_settings.blend_mode.into(),
                    pbr: particle_settings.pbr.into(),
                    fade_edge: particle_settings.fade_edge,
                    fade_scene: particle_settings.fade_scene,
                    flags,
                    __padding: Vec3::ZERO,
                },
                FireworkImages {
                    base_color_texture: particle_settings.base_color_texture.clone(),
                    normal_map_texture: particle_settings.normal_map_texture.clone(),
                    orm_texture: particle_settings.orm_texture.clone(),
                },
            )
        })
        .collect()
}

/// Due to persistence of the render world, we clean up every frame
// TODO: Maybe add some logic to only despawn orphaned render entities if we see performance issues
fn cleanup_firework_components(
    mut commands: Commands,
    query: Query<Entity, With<FireworkRenderEntityMarker>>,
) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

/// We need to do some custom extraction since each spawner entity can produce several render
/// entities
fn extract_firework_components(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    query: Extract<Query<(Entity, (&ParticleSpawnerData, &ParticleSpawner))>>,
) {
    let mut values = Vec::with_capacity(*previous_len);
    for (entity, query_item) in &query {
        for bundle in extract_component(query_item) {
            values.push((MainEntity::from(entity), bundle));
        }
    }
    *previous_len = values.len();
    commands.spawn_batch(values);
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    mut firework_uniform_layouts: ResMut<FireworkUniformBindgroupLayouts>,
    mut firework_pipelines: ResMut<FireworkPipelines>,
    render_device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FireworkPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_mesh_instances: Res<RenderMeshInstances>,
    particle_materials: Query<(Entity, &MainEntity, &ParticleMaterialData)>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    mut views: Query<(
        &ExtractedView,
        Option<&ShadowFilteringMethod>,
        &Msaa,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
    )>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    for (
        view,
        maybe_shadow_filtering_method,
        msaa,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        firework_uniform_layouts.ensure_created(&render_device, msaa.samples());
        let bindgroup_layout = firework_uniform_layouts.get(msaa.samples());
        firework_pipelines.ensure_created(
            msaa.samples(),
            mesh_pipeline.clone(),
            bindgroup_layout.clone(),
        );
        let custom_pipeline = firework_pipelines.get(msaa.samples());
        let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };
        let mut view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);

        match maybe_shadow_filtering_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        let rangefinder = view.rangefinder3d();
        for (entity, main_entity, particle_material_data) in &particle_materials {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let mut key = view_key
                | MeshPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
            //key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
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

            let pipeline = pipelines.specialize(&pipeline_cache, custom_pipeline, key);
            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder.distance_translation(&mesh_instance.translation),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
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

#[allow(clippy::too_many_arguments)]
pub fn prepare_firework_bindgroup(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    firework_uniform_layouts: Res<FireworkUniformBindgroupLayouts>,
    firework_uniforms: Res<ComponentUniforms<FireworkUniform>>,
    mut dummy_depth_textures: ResMut<DummyTextures>,
    view_query: Query<(Entity, &Msaa, Option<&ViewPrepassTextures>), With<ViewTarget>>,
    item_query: Query<(Entity, &FireworkImages), With<FireworkRenderEntityMarker>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    if let Some(binding) = firework_uniforms.uniforms().binding() {
        for (item_entity, firework_images) in &item_query {
            let mut bindgroups = HashMap::new();
            for (view_entity, msaa, view_prepass_textures_opt) in &view_query {
                let bindgroup_layout = firework_uniform_layouts.get(msaa.samples());

                dummy_depth_textures.ensure_has_samples(msaa.samples(), &render_device);

                let DummyTextures {
                    depth_textures,
                    base_color_texture_sampler,
                    base_color_texture,
                    normal_map_texture_sampler,
                    normal_map_texture,
                    orm_texture_sampler,
                    orm_texture,
                } = dummy_depth_textures.as_ref();

                let base_color_texture_view = firework_images
                    .base_color_texture
                    .as_ref()
                    .and_then(|img| gpu_images.get(img.id()))
                    .map(|img| &img.texture_view)
                    .unwrap_or(base_color_texture);

                let normal_map_texture_view = firework_images
                    .normal_map_texture
                    .as_ref()
                    .and_then(|img| gpu_images.get(img.id()))
                    .map(|img| &img.texture_view)
                    .unwrap_or(normal_map_texture);

                let orm_texture_view = firework_images
                    .orm_texture
                    .as_ref()
                    .and_then(|img| gpu_images.get(img.id()))
                    .map(|img| &img.texture_view)
                    .unwrap_or(orm_texture);

                let prepass_depth_texture_view = if let Some(depth) =
                    view_prepass_textures_opt.and_then(|vpt| vpt.depth.as_ref())
                {
                    &depth.texture.default_view
                } else {
                    depth_textures.get(&msaa.samples()).unwrap()
                };

                let entries = BindGroupEntries::sequential((
                    binding.clone(),
                    prepass_depth_texture_view,
                    base_color_texture_sampler,
                    base_color_texture_view,
                    normal_map_texture_sampler,
                    normal_map_texture_view,
                    orm_texture_sampler,
                    orm_texture_view,
                ));

                bindgroups.insert(
                    view_entity,
                    render_device.create_bind_group(
                        "Firework Uniform Bindgroup",
                        bindgroup_layout,
                        &entries,
                    ),
                );
            }

            commands
                .entity(item_entity)
                .insert(FireworkUniformBindgroups { bindgroups });
        }
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
            .flat_map(|i| i.iter())
            .map(|p| p.position - Vec3::splat(p.scale))
            .fold(Vec3::MAX, |acc, v| acc.min(v));
        let max = spawner_data
            .particles
            .iter()
            .flat_map(|i| i.iter())
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

#[derive(Resource, Default)]
pub struct FireworkPipelines {
    /// One pipeline per MSAA sample count
    /// will be initialize during "prepare" stage
    pipelines: HashMap<u32, FireworkPipeline>,
}

impl FireworkPipelines {
    pub fn ensure_created(
        &mut self,
        msaa_samples: u32,
        mesh_pipeline: MeshPipeline,
        uniform_layout: BindGroupLayout,
    ) {
        let vertex_shader = PARTICLE_SHADER_HANDLE;

        self.pipelines.insert(
            msaa_samples,
            FireworkPipeline {
                vertex_shader,
                mesh_pipeline,
                uniform_layout,
            },
        );
    }

    pub fn get(&self, msaa_samples: u32) -> &FireworkPipeline {
        self.pipelines.get(&msaa_samples).unwrap()
    }
}

pub struct FireworkPipeline {
    vertex_shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
    uniform_layout: BindGroupLayout,
}

// Added a basic Debug impl for simpler checking of the contents of the sample->pipeline map.
impl std::fmt::Debug for FireworkPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FireworkPipeline {{...}}")
    }
}

impl SpecializedRenderPipeline for FireworkPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let view_layout = self.mesh_pipeline.get_view_layout(key.into()).clone();
        let layout = vec![view_layout, self.uniform_layout.clone()];
        let format = if key.contains(MeshPipelineKey::HDR) {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let mut shader_defs = vec!["MESH_BINDGROUP_1".into(), "VERTEX_UVS".into()];

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }

        if key.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        RenderPipelineDescriptor {
            label: Some("Firework Pipeline".into()),
            layout,
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                // meshes typically live in bind group 2. because we are using bindgroup 1
                // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
                // linked in the shader
                shader_defs: shader_defs.clone(),
                entry_point: "vertex".into(),
                buffers: vec![VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                    step_mode: VertexStepMode::Instance,
                    attributes: vec![
                        // position and scale
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                        },
                        // rotation
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: VertexFormat::Float32x4.size(),
                            shader_location: 4,
                        },
                        // base color
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 2 * VertexFormat::Float32x4.size(),
                            shader_location: 5,
                        },
                        // emissive color
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 3 * VertexFormat::Float32x4.size(),
                            shader_location: 6,
                        },
                    ],
                }],
            },
            fragment: Some(FragmentState {
                shader: self.vertex_shader.clone(),
                // meshes typically live in bind group 2. because we are using bindgroup 1
                // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
                // linked in the shader
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: false,
                // Bevy uses reverse-Z, so Greater really means closer
                depth_compare: CompareFunction::Greater,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: true,
        }
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetFireworkBindGroup<1>,
    DrawFirework,
);

pub struct SetFireworkBindGroup<const I: usize>;
impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetFireworkBindGroup<I> {
    type Param = ();
    type ViewQuery = Entity;
    type ItemQuery = (
        Read<DynamicUniformIndex<FireworkUniform>>,
        &'static FireworkUniformBindgroups,
    );

    fn render<'w>(
        _item: &P,
        view_entity: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        item_query: bevy::ecs::query::ROQueryItem<'w, Option<Self::ItemQuery>>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (uniform_index, firework_uniform_bindgroups) = item_query.unwrap();

        let bindgroup = firework_uniform_bindgroups
            .bindgroups
            .get(&view_entity)
            .unwrap();
        pass.set_bind_group(I, bindgroup, &[uniform_index.index()]);
        RenderCommandResult::Success
    }
}

pub struct DrawFirework;

impl<P: PhaseItem> RenderCommand<P> for DrawFirework {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffer_slice = instance_buffer.unwrap().buffer.slice(..);
        let buffer_length = instance_buffer.unwrap().length as u32;

        pass.set_vertex_buffer(0, buffer_slice);
        pass.draw(0..6, 0..buffer_length);

        RenderCommandResult::Success
    }
}
