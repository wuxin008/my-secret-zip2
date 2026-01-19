#version 450
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location=0) flat out float outviewdepth;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - renderUbo.Nx / 2.0f) / factor, (pos.y - renderUbo.Ny / 2.0f) / factor, (pos.z - renderUbo.Nz / 2.0f) / factor);
}

void main(){
    float factor = max(renderUbo.Nx, max(renderUbo.Ny, renderUbo.Nz));

    vec4 viewlocation = renderUbo.view*renderUbo.model*vec4(particle_norm(inPosition, 1),1);
    
    outviewdepth = viewlocation.z;

    gl_Position = renderUbo.proj*viewlocation;

    float nearHeight = 2*renderUbo.zNear*tan(renderUbo.fovy/2);

    float scale = 900/nearHeight;

    float nearSize = renderUbo.particleRadius*renderUbo.zNear/(-outviewdepth);

    gl_PointSize = 2*scale*nearSize;
} 