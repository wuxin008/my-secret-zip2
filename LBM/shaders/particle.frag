#version 450
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout(location=0) flat in float inviewdepth;

layout(location=0) out float outdepth;
layout(location=1) out float outthickness;

void main(){
    float radius = renderUbo.particleRadius;
    vec2 pos = gl_PointCoord - vec2(0.5);
    if(length(pos) > 0.5){
        discard;
        return;
    }
    float l = radius*2*length(pos);
    float viewdepth = inviewdepth + sqrt(radius*radius - l*l);
    
    vec4 temp = vec4(0,0,viewdepth,1);
    
    temp = renderUbo.proj*temp;

    outdepth = temp.z/temp.w;

    outthickness = 4*sqrt(radius*radius - l*l) / 1000.0f;
    
}