#import bevy_pbr::view_transformations::{
    position_world_to_clip,
    position_world_to_view,
    position_view_to_clip,
    direction_view_to_world,
}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::pbr_types::{
    STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, 
    STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND, 
    STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MULTIPLY, 
    STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT,
    STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT,
    STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
    pbr_input_new
}
#import bevy_pbr::mesh_types::{
    MESH_FLAGS_SHADOW_RECEIVER_BIT,
    MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT
};
#import bevy_pbr::pbr_functions as fns
#import bevy_pbr::pbr_bindings,

struct Vertex {
    @location(3) i_pos_scale: vec4<f32>,
    @location(4) i_color: vec4<f32>,
    @builtin(vertex_index) index: u32,
};

struct FireworkUniform {
    alpha_mode: u32,
    pbr: u32,
    fade_edge: f32,
    fade_scene: f32,
}

@group(1) @binding(0) var<uniform> firework_uniform: FireworkUniform;

#ifdef DEPTH_PREPASS
#ifdef MULTISAMPLED
@group(1) @binding(1) var depth_prepass_texture: texture_depth_multisampled_2d;
#else // MULTISAMPLED
@group(1) @binding(1) var depth_prepass_texture: texture_depth_2d;
#endif // MULTISAMPLED
#endif // DEPTH_PREPASS


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) world_tangent: vec3<f32>,
};

fn extract_rot(bigmat: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3(bigmat[0].xyz, bigmat[1].xyz, bigmat[2].xyz) * (1. / bigmat[3][3]);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var positions = array<vec3<f32>, 6>(
        vec3(0.5, 0.5, 0.),
        vec3(-0.5, 0.5, 0.),
        vec3(-0.5, -0.5, 0.),
        vec3(0.5, -0.5, 0.),
        vec3(0.5, 0.5, 0.),
        vec3(-0.5, -0.5, 0.),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2(1., 1.),
        vec2(0., 1.),
        vec2(0., 0.),
        vec2(1., 0.),
        vec2(1., 1.),
        vec2(0., 0.),
    );

    var position_world = direction_view_to_world(positions[vertex.index] * vertex.i_pos_scale.w) + vertex.i_pos_scale.xyz;

    var out: VertexOutput;
    out.world_position = vec4(position_world, 1.);
    out.position = position_world_to_clip(position_world);
    out.color = vertex.i_color;
    out.world_normal = direction_view_to_world(vec3(0., 0., 1.));
    out.world_tangent = direction_view_to_world(vec3(1., 0., 0.));
    out.uv = uvs[vertex.index];

    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var color = in.color;

    if firework_uniform.fade_edge > 0. {
        let vec_to_center = in.uv - vec2(0.5, 0.5);
        let dist_to_center = length(vec_to_center) * 2.;
        let dist_from_edge = clamp(1. - dist_to_center, 0., 1.);
        let edge_blend = smoothstep(0., firework_uniform.fade_edge, dist_from_edge);
        color.a *= edge_blend;
    }

#ifdef DEPTH_PREPASS
    let depth_scene = prepass_depth(in.position, 0u);
    let diff = abs(1. / in.position.z - 1. / depth_scene);
    let scene_blend = smoothstep(0., firework_uniform.fade_scene, diff);
    color.a *= scene_blend;
#endif


    if color.a == 0. {
        discard;
    }

    if firework_uniform.pbr == 0u {
        return color;
    } else {
        return pbr_stuff(color, in.position, in.world_position, in.world_normal, in.uv, in.world_tangent, is_front);
    }
}

fn pbr_stuff(
    base_color: vec4<f32>, 
    position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    uv: vec2<f32>,
    world_tangent: vec3<f32>,
    is_front: bool,
) -> vec4<f32> {
    var pbr_input = pbr_input_new();
    pbr_input.material.flags = 
        (STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND 
        | STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT
        | STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT);
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr_input.material.base_color = base_color;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.frag_coord = position;
    pbr_input.world_position = world_position;
    pbr_input.flags |= MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT;

    let double_sided = (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

    pbr_input.world_normal = normalize(fns::prepare_world_normal(
        world_normal,
        double_sided,
        is_front,
    ));

    pbr_input.N = pbr_input.world_normal;

    pbr_input.V = fns::calculate_view(pbr_input.world_position, pbr_input.is_orthographic);

    var out: vec4<f32>;

    pbr_input.material.base_color = fns::alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef PREPASS_PIPELINE
    out = vec4(1., 0., 0., 1.);
#else
    out = fns::apply_pbr_lighting(pbr_input);
    out = fns::main_pass_post_lighting_processing(pbr_input, out);
#endif

    return out;
}

#ifdef DEPTH_PREPASS
fn prepass_depth(frag_coord: vec4<f32>, sample_index: u32) -> f32 {
#ifdef MULTISAMPLED
    return textureLoad(depth_prepass_texture, vec2<i32>(frag_coord.xy), i32(sample_index));
#else // MULTISAMPLED
    return textureLoad(depth_prepass_texture, vec2<i32>(frag_coord.xy), 0);
#endif // MULTISAMPLED
}
#endif // DEPTH_PREPASS
