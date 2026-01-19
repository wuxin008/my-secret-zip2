#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_debug_printf : enable

layout(std430, binding = 0) uniform RenderingUBO {
    uint Nx;
    uint Ny;
    uint Nz;
    uint Nxyz;
    uint t;
    float zNear;
    float zFar;
    float fovy;
    float aspectRatio;
    float particleRadius;
    float cameraDistance;
    uint renderType;
    uint fluidType;
    float transmittance;
    vec4 fluidColor;
    vec3 cameraPos;
    vec3 cameraView;
    vec3 cameraUp;
    mat4 model;
    mat4 view;
    mat4 invView;
    mat4 proj;
    mat4 invProj;
} renderUbo;