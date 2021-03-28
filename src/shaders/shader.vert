#version 450
//glslangValidator shader.vert -V -o vert.spv

layout(push_constant) uniform UniformBuffer {
    vec2 u_screen_size;
    bool is_egui_system_texture;
};
layout(location = 0) in vec2 a_pos;//R32G32SFloat
layout(location = 1) in vec2 a_tex_coord;//R32G32SFloat
layout(location = 2) in uint a_color;//R32Uint
layout(location = 0) out vec2 v_tex_coord;//R32G32SFloat
layout(location = 1) out vec4 v_color;//R32G32B32A32SFloat

vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    v_tex_coord = a_tex_coord;
    // [u8; 4] SRGB as u32 -> [r, g, b, a]
    vec4 color = vec4(a_color & 0xFFu, (a_color >> 8) & 0xFFu, (a_color >> 16) & 0xFFu, (a_color >> 24) & 0xFFu);
    v_color = vec4(linear_from_srgb(color.rgb), color.a / 255.0);
    gl_Position=vec4(2.0*((a_pos.x/u_screen_size.x)-0.5),2.0*((a_pos.y/u_screen_size.y)-0.5),0.0,1.0);

}
