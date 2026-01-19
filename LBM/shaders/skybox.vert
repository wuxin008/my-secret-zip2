#version 450
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout (location = 0) in vec3 inPos;

layout (location = 0) out vec3 outUVW;

void main() 
{
	outUVW = vec3(inPos.x, inPos.z, -inPos.y);
	mat4 viewMat = mat4(mat3(renderUbo.view * renderUbo.model));
	gl_Position = renderUbo.proj * viewMat * vec4(inPos.xyz, 1.0);
	gl_Position.z = (1-1e-4) * gl_Position.w;
}
