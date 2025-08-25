#import bevy_pbr::view_transformations::{
    position_world_to_clip,
    position_world_to_view,
    position_view_to_clip,
    direction_view_to_world,
}
#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::pbr_types
#import bevy_pbr::lighting::LAYER_BASE
#import bevy_pbr::lighting::LAYER_CLEARCOAT
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
#import bevy_pbr::pbr_bindings

const FIREWORK_BASE_COLOR_TEXTURE_BIT: u32 = 1u;
const FIREWORK_NORMAL_MAP_TEXTURE_BIT: u32 = 1u << 1u;
const FIREWORK_ORM_TEXTURE_BIT: u32 = 1u << 2u;

struct Vertex {
    @location(3) i_pos_scale: vec4<f32>,
    @location(4) i_rotation: vec4<f32>,
    @location(5) i_base_color: vec4<f32>,
    @location(6) i_emissive_color: vec4<f32>,
    @builtin(vertex_index) index: u32,
};

struct FireworkUniform {
    alpha_mode: u32,
    pbr: u32,
    fade_edge: f32,
    fade_scene: f32,
    flags: u32,
    padding: vec3<f32>,
}

@group(1) @binding(0) var<uniform> firework_uniform: FireworkUniform;

#ifdef DEPTH_PREPASS
#ifdef MULTISAMPLED
@group(1) @binding(1) var depth_prepass_texture: texture_depth_multisampled_2d;
#else // MULTISAMPLED
@group(1) @binding(1) var depth_prepass_texture: texture_depth_2d;
#endif // MULTISAMPLED
#endif // DEPTH_PREPASS

@group(1) @binding(2)
var base_color_texture_sampler: sampler;
@group(1) @binding(3)
var base_color_texture: texture_2d<f32>;
@group(1) @binding(4)
var normal_map_texture_sampler: sampler;
@group(1) @binding(5)
var normal_map_texture: texture_2d<f32>;
@group(1) @binding(6)
var orm_texture_sampler: sampler;
@group(1) @binding(7)
var orm_texture: texture_2d<f32>;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive_color: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) world_tangent: vec3<f32>,
};

fn extract_rot(bigmat: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3(bigmat[0].xyz, bigmat[1].xyz, bigmat[2].xyz) * (1. / bigmat[3][3]);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let positions = array<vec3<f32>, 6>(
        vec3(0.5, 0.5, 0.),
        vec3(-0.5, 0.5, 0.),
        vec3(-0.5, -0.5, 0.),
        vec3(0.5, -0.5, 0.),
        vec3(0.5, 0.5, 0.),
        vec3(-0.5, -0.5, 0.),
    );
    let uvs = array<vec2<f32>, 6>(
        vec2(1., 1.),
        vec2(0., 1.),
        vec2(0., 0.),
        vec2(1., 0.),
        vec2(1., 1.),
        vec2(0., 0.),
    );

    let rot = vertex.i_rotation;
    let quad_to_camera_world = normalize(direction_view_to_world(vec3<f32>(0.,0.,1.)));
    let swing_twist = swing_twist_from_quat(rot, quad_to_camera_world);

    let position_world = quat_rotate_vec3(swing_twist.twist, direction_view_to_world(positions[vertex.index] * vertex.i_pos_scale.w)) + vertex.i_pos_scale.xyz;

    var out: VertexOutput;
    out.world_position = vec4(position_world, 1.);
    out.position = position_world_to_clip(position_world);
    out.color = vertex.i_base_color;
    out.emissive_color = vertex.i_emissive_color;
    out.world_normal = direction_view_to_world(vec3(0., 0., 1.));
    out.world_tangent = quat_rotate_vec3(swing_twist.twist, direction_view_to_world(vec3(1., 0., 0.)));
    out.uv = uvs[vertex.index];

    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var color = in.color;
    var emissive_color = in.emissive_color;
    var metallic = 0.;
    var perceptual_roughness = 1.0;

    if (firework_uniform.flags & FIREWORK_BASE_COLOR_TEXTURE_BIT) != 0u {
        let sample = textureSample(base_color_texture, base_color_texture_sampler, in.uv);
        color *= sample;
    }
    if (firework_uniform.flags & FIREWORK_ORM_TEXTURE_BIT) != 0u {
        let sample = textureSample(orm_texture, orm_texture_sampler, in.uv);
        perceptual_roughness = sample.g;
        metallic = sample.b;
    }

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
        return pbr_stuff(
            color,
            emissive_color,
            perceptual_roughness,
            metallic,
            in.position,
            in.world_position,
            in.world_normal,
            in.uv,
            in.world_tangent,
            is_front
        );
    }
}

fn pbr_stuff(
    base_color: vec4<f32>,
    emissive_color: vec4<f32>,
    perceptual_roughness: f32,
    metallic: f32,
    position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    uv: vec2<f32>,
    world_tangent: vec3<f32>,
    is_front: bool,
) -> vec4<f32> {
    var pbr_input = pbr_input_new();

    pbr_input.material.flags = (STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND | STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT | STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT);
    pbr_input.is_orthographic = view.clip_from_view[3].w == 1.0;

    pbr_input.material.base_color = base_color;
    pbr_input.material.emissive = vec4<f32>(emissive_color.rgb, 0.);
    pbr_input.material.metallic = metallic;
    pbr_input.material.perceptual_roughness = perceptual_roughness;
    pbr_input.frag_coord = position;
    pbr_input.world_position = world_position;
    pbr_input.flags |= MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT;

    let double_sided = (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;

    pbr_input.world_normal = normalize(fns::prepare_world_normal(
        world_normal,
        double_sided,
        is_front,
    ));

    if (firework_uniform.flags & FIREWORK_NORMAL_MAP_TEXTURE_BIT) != 0u { 
        let TBN = fns::calculate_tbn_mikktspace(pbr_input.world_normal, vec4(world_tangent, 1.));
        let Nt = textureSampleBias(
           normal_map_texture,
           normal_map_texture_sampler,
           uv,
           0.0,
        ).rgb;
        pbr_input.N = fns::apply_normal_mapping(pbr_input.material.flags, TBN, double_sided, is_front, Nt);
    } else {
        pbr_input.N = pbr_input.world_normal;
    }

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


// ---------------------------------------------
// Swing–Twist decomposition in WGSL
// Conventions:
// - Quaternions are (x, y, z, w) with Hamilton product.
// - Composition: quat_mul(a, b) == a ⊗ b (apply b, then a).
// - Result: q = swing ⊗ twist  (apply twist first, then swing).
// - `axis` must be a normalized 3D vector.
// ---------------------------------------------

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
  let n = sqrt(dot(q, q));
  return select(q, q / n, n != 0.0);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-q.xyz, q.w);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  // Hamilton product: a ⊗ b
  let aw = a.w; let bw = b.w;
  let av = a.xyz; let bv = b.xyz;
  let xyz = aw * bv + bw * av + cross(av, bv);
  let w   = aw * bw - dot(av, bv);
  return vec4<f32>(xyz, w);
}

fn quat_rotate_vec3(q_in: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  // Ensure unit quaternion
  let q = normalize(q_in);

  // Split into vector (imag) and scalar parts
  let u = q.xyz;
  let s = q.w;

  // Optimized form of q * v * conj(q)
  return 2.0 * dot(u, v) * u
       + (s*s - dot(u, u)) * v
       + 2.0 * s * cross(u, v);
}

// --- Swing–Twist ---

struct SwingTwist {
  swing: vec4<f32>, // unit quaternion
  twist: vec4<f32>, // unit quaternion (pure twist about `axis`)
};

// Decompose a unit quaternion q into swing * twist about `axis` (unit).
fn swing_twist_from_quat(q_in: vec4<f32>, axis_unit: vec3<f32>) -> SwingTwist {
  // Ensure q is unit to keep numerics stable
  let q = quat_normalize(q_in);

  // Project the quaternion's vector part onto the axis
  let v = q.xyz;
  let p = dot(v, axis_unit) * axis_unit;

  // Twist keeps the projected vector part + original scalar (w)
  var twist = vec4<f32>(p, q.w);
  twist = quat_normalize(twist);

  // swing = q * conj(twist)  (since q = swing ⊗ twist)
  var swing = quat_mul(q, quat_conj(twist));
  swing = quat_normalize(swing);

  // Optional: make twist represent the "shortest" rotation around axis
  // (so w >= 0), which can help continuity.
  if (twist.w < 0.0) {
    twist = -twist;
    swing = -swing; // keep product equal
  }

  return SwingTwist(swing, twist);
}
