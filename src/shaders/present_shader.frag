#version 450
layout (location=0) in vec2 tex_coord;
layout (location=0) out vec4 color;
layout(set = 1, binding = 0) uniform texture2D t_texture;
layout(set = 0, binding = 0) uniform sampler s_texture;

//glslangValidator present_shader.frag -V -o present_frag.spv
void main(){
    color= texture(sampler2D(t_texture, s_texture), tex_coord);
}