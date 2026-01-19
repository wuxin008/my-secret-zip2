#version 450
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - renderUbo.Nx / 2.0f) / factor, (pos.y - renderUbo.Ny / 2.0f) / factor, (pos.z - renderUbo.Nz / 2.0f) / factor);
}

void main() {
    gl_PointSize = 2.0;
    float factor = max(renderUbo.Nx, max(renderUbo.Ny, renderUbo.Nz));
    gl_Position = renderUbo.proj * renderUbo.view * renderUbo.model * vec4(particle_norm(inPosition, 1), 1.0);
    fragColor = inColor.rgb;
}