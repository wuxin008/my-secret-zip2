#version 450
#extension GL_ARB_shading_language_include : require
#include "render_header.glsl"

layout(std430, push_constant) uniform ModelPushConstants{
    uint nTriangles;
    uint nFrames;
} pcs;

layout(std430, binding = 1) readonly buffer PointsBuffer {
    float points[];
};

layout(std430, binding = 2) readonly buffer TMBuffer {
    mat4 TMs[];
};

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec2 fragTexCoord;

vec3 particle_norm(vec3 pos, float factor) {
    return vec3((pos.x - renderUbo.Nx / 2.0f) / factor, (pos.y - renderUbo.Ny / 2.0f) / factor, (pos.z - renderUbo.Nz / 2.0f) / factor);
}

void main() {
    // 获取当前顶点在整个渲染调用中的全局索引
    uint globalVertexIndex = gl_VertexIndex;

    // 计算当前正在处理的三角形的索引
    // 每个三角形有3个顶点，所以通过除以3可以得到三角形的索引
    uint triangleIndex = globalVertexIndex / 3;

    // 计算当前顶点是该三角形的第几个点 (0, 1, 或 2)
    uint vertexInTriangleIndex = globalVertexIndex % 3;

    // --- 计算位置数据的偏移量 (以 float 为单位) ---
    // 每个 glm::vec3 占用 3 个 float
    uint p0_start_float_idx = triangleIndex * 3; // p0 的 x 分量起始索引
    uint p1_start_float_idx = p0_start_float_idx + pcs.nTriangles * 3; // p1 的 x 分量起始索引
    uint p2_start_float_idx = p0_start_float_idx + pcs.nTriangles * 6; // p2 的 x 分量起始索引

    vec3 position;
    if (vertexInTriangleIndex == 0) {
        position = vec3(points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p0_start_float_idx], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p0_start_float_idx + 1], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p0_start_float_idx + 2]);
    } else if (vertexInTriangleIndex == 1) {
        position = vec3(points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p1_start_float_idx], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p1_start_float_idx + 1], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p1_start_float_idx + 2]);
    } else { // vertexInTriangleIndex == 2
        position = vec3(points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p2_start_float_idx], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p2_start_float_idx + 1], points[(renderUbo.t % pcs.nFrames) * pcs.nTriangles * 9 + p2_start_float_idx + 2]);
    }

    // --- 计算纹理坐标数据的偏移量 (以 float 为单位) ---
    // 纹理坐标数据紧跟在所有位置数据之后
    // 纹理坐标部分起始的 float 索引
    uint texCoord_buffer_start_float_idx = uint(pcs.nFrames) * pcs.nTriangles * 9; // p0 + p1 + p2 的总 float 数量

    // 每个 float2 占用 2 个 float
    uint tc0_start_float_idx = texCoord_buffer_start_float_idx + triangleIndex * 2; // tc0 的 u 分量起始索引
    uint tc1_start_float_idx = tc0_start_float_idx + pcs.nTriangles * 2; // tc1 的 u 分量起始索引
    uint tc2_start_float_idx = tc0_start_float_idx + pcs.nTriangles * 4; // tc2 的 u 分量起始索引

    vec2 texCoord;
    if (vertexInTriangleIndex == 0) {
        texCoord = vec2(points[tc0_start_float_idx], points[tc0_start_float_idx + 1]);
    } else if (vertexInTriangleIndex == 1) {
        texCoord = vec2(points[tc1_start_float_idx], points[tc1_start_float_idx + 1]);
    } else { // vertexInTriangleIndex == 2
        texCoord = vec2(points[tc2_start_float_idx], points[tc2_start_float_idx + 1]);
    }
    
    float factor = max(renderUbo.Nx, max(renderUbo.Ny, renderUbo.Nz));
    gl_Position = renderUbo.proj * renderUbo.view * renderUbo.model * vec4(particle_norm(position, 1.0), 1.0);
    //gl_Position = vec4(position, 1.0); // 设置最终的裁剪空间位置
    fragTexCoord = texCoord;           // 传递纹理坐标到片段着色器
    fragPosition = position;           // 传递位置到片段着色器，如果你需要的话
}