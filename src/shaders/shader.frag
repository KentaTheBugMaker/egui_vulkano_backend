#version 450
//glslangValidator shader.frag -V -o frag.spv
layout(location = 0) in vec2 v_tex_coord;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform texture2D t_texture;
layout(set = 0, binding = 0) uniform sampler s_texture;
layout(push_constant) uniform UniformBuffer {
    vec2 u_screen_size;
    bool is_egui_system_texture;
};
void main() {
    vec4 texture_color=texture(sampler2D(t_texture,s_texture),v_tex_coord);
    if (is_egui_system_texture){
        texture_color= texture_color.rrrr;
    }
    f_color = v_color * texture_color;
}