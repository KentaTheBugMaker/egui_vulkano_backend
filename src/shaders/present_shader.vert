#version 450

//glslangValidator present_shader.vert -V -o present_vert.spv

layout(location = 0) out vec2 v_tex_coord;//R32G32SFloat

void main() {
    switch (gl_VertexIndex){
        case 0:
        gl_Position=vec4(-1.0, -1.0, 0.0, 1.0);
        v_tex_coord=vec2(0.0, 0.0);
        break;
        case 1:
        gl_Position=vec4(-1.0, 1.0, 0.0, 1.0);
        v_tex_coord=vec2(0.0, 1.0);
        break;
        case 2:
        gl_Position=vec4(1.0, -1.0, 0.0, 1.0);
        v_tex_coord=vec2(1.0, 0.0);
        break;
        case 3:
        gl_Position=vec4(1.0, 1.0, 0.0, 1.0);
        v_tex_coord=vec2(1.0, 1.0);
        break;
        default :
        break;
    }
}