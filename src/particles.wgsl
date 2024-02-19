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

struct Vertex {
    @location(3) i_pos_scale: vec4<f32>,
    @location(4) i_color: vec4<f32>,
    @location(5) pbr: u32,
    @builtin(vertex_index) index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) pbr: u32,
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
    out.pbr = vertex.pbr;
    out.uv = uvs[vertex.index];

    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let vec_to_center = in.uv - vec2(0.5, 0.5);
    let dist_to_center = length(vec_to_center) * 2.;
    var color = in.color;
    color.a *= sqrt(clamp(1. - dist_to_center, 0., 1.));

    if color.a == 0. {
        discard;
    }

    if in.pbr == 0u {
        return color;
    } else {
        return pbr_stuff(color, in.position, in.world_position, in.world_normal, in.uv, is_front);
    }
}

fn pbr_stuff(
    base_color: vec4<f32>, 
    position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    uv: vec2<f32>,
    is_front: bool,
) -> vec4<f32> {
    var pbr_input = pbr_input_new();
    //if alpha_mode == 0u {
    pbr_input.material.flags = 
        (STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND 
        | STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT
        | STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT);
        //& ~STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
    // } else if alpha_mode == 1u {
    //     pbr_input.material.flags = STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MULTIPLY;
    // }
    pbr_input.is_orthographic = view.projection[3].w == 1.0;
    pbr_input.material.base_color = base_color;
    pbr_input.material.metallic = 0.0;
    pbr_input.material.perceptual_roughness = 1.0;
    pbr_input.frag_coord = position;
    pbr_input.world_position = world_position;
    pbr_input.flags |= MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT;

    let double_sided = (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

    pbr_input.world_normal = fns::prepare_world_normal(
        world_normal,
        double_sided,
        is_front,
    );

    pbr_input.N = fns::apply_normal_mapping(
        pbr_input.material.flags,
        world_normal,
        double_sided,
        is_front,
#ifdef VERTEX_TANGENTS
#ifdef STANDARDMATERIAL_NORMAL_MAP
    // TODO: Sort out the tangents stuff
        vec3(0.,0.,1.),
#endif
#endif
        uv,
        view.mip_bias,
    );
    pbr_input.V = fns::calculate_view(world_position, pbr_input.is_orthographic);

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

