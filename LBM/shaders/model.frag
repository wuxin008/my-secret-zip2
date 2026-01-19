#version 450
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout(binding = 3) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord);
}