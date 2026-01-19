#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    //if (fragColor.x < 0 || fragColor.y < 0 || fragColor.z < 0) discard;
    outColor = vec4(fragColor, 1.0f);
}