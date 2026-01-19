#pragma once

//#define RAYTRACE
//#define NODEBUG
#define DEBUGTOOLS

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <ktx.h>
#include <ktxvulkan.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/packing.hpp>

#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_obj_loader.h>
#include <tiny_gltf.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <random>
#include <utilities.hpp>

#define TYPE_S  0x01 // 0b00000001 // (stationary or moving) solid boundary
#define TYPE_E  0x02 // 0b00000010 // equilibrium boundary (inflow/outflow)
#define TYPE_T  0x04 // 0b00000100 // temperature boundary
#define TYPE_F  0x08 // 0b00001000 // fluid
#define TYPE_I  0x10 // 0b00010000 // interface
#define TYPE_G  0x20 // 0b00100000 // gas
#define TYPE_X  0x40 // 0b01000000 // reserved type X
#define TYPE_Y  0x80 // 0b10000000 // reserved type Y

#define TYPE_MS 0x03 // 0b00000011 // cell next to moving solid boundary
#define TYPE_BO 0x03 // 0b00000011 // any flag bit used for boundaries (temperature excluded)
#define TYPE_IF 0x18 // 0b00011000 // change from interface to fluid
#define TYPE_IG 0x30 // 0b00110000 // change from interface to gas
#define TYPE_GI 0x38 // 0b00111000 // change from gas to interface
#define TYPE_SU 0x38 // 0b00111000 // any flag bit used for SURFACE

#define D 3 // dimension of space
#define Q 19

#ifdef NODEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsAndComputeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities = {};
    std::vector<VkSurfaceFormatKHR> formats{};
    std::vector<VkPresentModeKHR> presentModes{};
};

struct vecQ {
    float f[Q];
};

struct SimulateUBO {
    alignas(4) uint32_t Nx = 0;
    alignas(4) uint32_t Ny = 0;
    alignas(4) uint32_t Nz = 0;
    alignas(4) uint32_t Nxyz = 0;
    //alignas(4) uint32_t particleCount = 0;
    alignas(4) float particleRho = 1.0f;
    alignas(4) float niu = 0.0f;
    alignas(4) float smoothness = 0.5f;
    alignas(4) float tau = 0.0f;
    alignas(4) float inv_tau = 0.0f;
    alignas(4) float isoVal = 0.0f;
    alignas(4) float fx = 0.0f;
    alignas(4) float fy = 0.0f;
    alignas(4) float fz = 0.0f;
    alignas(4) float sigmas = 0.0f;
    alignas(4) float rx = 0.0f;
    alignas(4) float ry = 0.0f;
    alignas(4) float distance = 0.0f;
    alignas(4) uint32_t t = 0;
};

struct RenderingUBO {
    alignas(4)  uint32_t Nx = 0;
    alignas(4)  uint32_t Ny = 0;
    alignas(4)  uint32_t Nz = 0;
    alignas(4)  uint32_t Nxyz = 0;
    alignas(4)  uint32_t t = 0;
    alignas(4)  float zNear = 0.0f;
    alignas(4)  float zFar = 0.0f;
    alignas(4)  float fovy = 0.0f;
    alignas(4)  float aspectRatio = 0.0f;
    alignas(4)  float particleRadius = 0.0f;
    alignas(4)  float cameraDistance = 0.0f;
    alignas(4)  uint renderType = 0u;
    alignas(4)  uint fluidType = 0u;
    alignas(4)  float transmittance = 0.0f;
    alignas(16) glm::vec4 fluidColor = glm::vec4(0.0f);
    alignas(16) glm::vec3 cameraPos = glm::vec3(0.0f);
    alignas(16) glm::vec3 cameraView = glm::vec3(0.0f);
    alignas(16) glm::vec3 cameraUp = glm::vec3(0.0f);
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 invView;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 invProj;
};

struct VoxelizationPushConstants {
    float particle_number;
    uint flag_type;
};

struct ModelPushConstants {
    uint32_t nTriangles;
    uint32_t nFrames;
};

struct Particle {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);

        return attributeDescriptions;
    }
};

struct Vertex {
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

void ImGui_ImplVulkanH_CreateWindowCommandBuffers(VkPhysicalDevice physical_device, VkDevice device, ImGui_ImplVulkanH_Window* wd, uint32_t queue_family, const VkAllocationCallbacks* allocator);
void ImGui_ImplVulkanH_DestroyFrame(VkDevice device, ImGui_ImplVulkanH_Frame* fd, const VkAllocationCallbacks* allocator);
void ImGui_ImplVulkanH_DestroyFrameSemaphores(VkDevice device, ImGui_ImplVulkanH_FrameSemaphores* fsd, const VkAllocationCallbacks* allocator);

class LBM {
public:
    //particles
    std::vector<float> vels; // velocity
    std::vector<float> rhos; // rho
    std::vector<uint> flags; // flag
    std::vector<float> cfs; // cellForce
    void init(uint Nx = 128, uint Ny = 128, uint Nz = 128, float niu = 0.5f, float sigmas = 1e-6);
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    uint get_Nxyz() const {
        return this->Nxyz;
    }

private:
    uint Nx = 256;
    uint Ny = 256;
    uint Nz = 256;
    uint Nxyz = Nx * Ny * Nz;

    const int MAX_FRAMES_IN_FLIGHT = 1;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation",
        "VK_LAYER_LUNARG_monitor",
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        //VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        "VK_KHR_storage_buffer_storage_class",
    };

    GLFWwindow* window;
    bool firstMouse = true;
    bool mouseFree = false;
    double lastX = 0, lastY = 0;
    float rx = 0, ry = 90;
    float distanceFactor = 1.0f;
    float distance = distanceFactor * max(Nx, max(Ny, Nz));
    bool framebufferResized = false;
    bool isInit = false;
    bool isRun = true;
    uint32_t currentTime = 0;
    uint32_t currentFrame = 0;
    uint32_t upscaleTimes = 0;
    std::vector<glm::ivec2> imageSize;

    float lastFrameTime = 0.0f;
    float totalTime = 0.0f;
    double lastTime = 0.0f;

    //uint32_t particle_count = 0;
    //uint32_t wireframe_count = 0;
    //uint32_t surface_count = 0;

    RenderingUBO renderUbo{};
    SimulateUBO simulateUbo{};

    std::vector<glm::vec3> skyboxVertices = {
        { -1,  1, -1,},
        { -1, -1, -1,},
        {  1, -1, -1,},
        {  1,  1, -1,},
        { -1,  1,  1,},
        {  1,  1,  1,},
        {  1, -1,  1,},
        { -1, -1,  1,},
        { -1,  1, -1,},
        {  1,  1, -1,},
        {  1,  1,  1,},
        { -1,  1,  1,},
        {  1,  1, -1,},
        {  1, -1, -1,},
        {  1, -1,  1,},
        {  1,  1,  1,},
        {  1, -1, -1,},
        { -1, -1, -1,},
        { -1, -1,  1,},
        {  1, -1,  1,},
        { -1, -1, -1,},
        { -1,  1, -1,},
        { -1,  1,  1,},
        { -1, -1,  1,},
    };

    std::vector<uint32_t> skyboxIndices = {
         0,
         1,
         2,
         2,
         3,
         0,
         4,
         5,
         6,
         6,
         7,
         4,
         8,
         9,
        10,
        10,
        11,
         8,
        12,
        13,
        14,
        14,
        15,
        12,
        16,
        17,
        18,
        18,
        19,
        16,
        20,
        21,
        22,
        22,
        23,
        20,
    };

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;  
    VkQueue computeQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSampler depthImageSampler;
    VkImage fluidDepthImage;
    VkDeviceMemory fluidDepthImageMemory;
    VkImageView fluidDepthImageView;
    VkSampler fluidDepthImageSampler;
    VkImage filteredFluidDepthImage;
    VkDeviceMemory filteredFluidDepthImageMemory;
    VkImageView filteredFluidDepthImageView;
    VkSampler filteredFluidDepthImageSampler;
    VkImage thickImage;
    VkDeviceMemory thickImageMemory;
    VkImageView thickImageView;
    VkSampler thickImageSampler;
    VkImage backgroundImage;
    VkDeviceMemory backgroundImageMemory;
    VkImageView backgroundImageView;
    VkSampler backgroundImageSampler;

    VkImage skyboxImage;
    VkDeviceMemory skyboxImageMemory;
    VkImageView skyboxImageView;
    VkSampler skyboxSampler;

    VkRenderPass renderPass;
    VkRenderPass fluidGraphicRenderPass;
    std::vector<VkFramebuffer> fluidsFramebuffers;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkDescriptorSetLayout graphicsDescriptorSetLayout;
    VkDescriptorSetLayout raytraceDescriptorSetLayout;
    VkDescriptorSetLayout upscaleDescriptorSetLayout;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkDescriptorSetLayout postprocessDescriptorSetLayout;
    VkDescriptorSetLayout filteredDescriptorSetLayout;

    VkPipelineLayout graphicsPipelineLayout;
    VkPipelineLayout raytracePipelineLayout;
    VkPipelineLayout upscalePipelineLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipelineLayout postprocessPipelineLayout;
    VkPipelineLayout filteredPipelineLayout;

    VkPipeline particlePipeline;
    VkPipeline wireframePipeline;
    VkPipeline surfacePipeline;
    VkPipeline skyboxPipeline;
    VkPipeline raytracePipeline;
    VkPipeline upscalePipeline;
    VkPipeline computePipeline;
    VkPipeline initPipeline;
    VkPipeline surface0Pipeline;
    VkPipeline surface1Pipeline;
    VkPipeline surface2Pipeline;
    VkPipeline surface3Pipeline;
    VkPipeline collideAndStreamPipeline;
    VkPipeline filteredPipeline;
    VkPipeline postprocessPipeline;

    VkCommandPool commandPool;

    std::vector<VkBuffer> totalMassBuffers;
    std::vector<VkDeviceMemory> totalMassBuffersMemory;

    std::vector<VkBuffer> stateBuffers;
    std::vector<VkDeviceMemory> stateBuffersMemory;
    std::vector<VkBuffer> velocityBuffers;
    std::vector<VkDeviceMemory> velocityBuffersMemory;
    std::vector<VkBuffer> rhoBuffers;
    std::vector<VkDeviceMemory> rhoBuffersMemory;
    std::vector<VkBuffer> flagBuffers;
    std::vector<VkDeviceMemory> flagBuffersMemory;
    std::vector<VkBuffer> DDF1Buffers;
    std::vector<VkDeviceMemory> DDF1BuffersMemory;
    std::vector<VkBuffer> DDF2Buffers;
    std::vector<VkDeviceMemory> DDF2BuffersMemory;
    std::vector<VkBuffer> cellForceBuffers;
    std::vector<VkDeviceMemory> cellForceBuffersMemoryBuffers;
    std::vector<VkBuffer> PhiBuffers;
    std::vector<VkDeviceMemory> PhiBuffersMemory;
    std::vector<VkBuffer> MassBuffers;
    std::vector<VkDeviceMemory> MassBuffersMemory;
    std::vector<VkBuffer> MassExBuffers;
    std::vector<VkDeviceMemory> MassExBuffersMemory;

    std::vector<VkBuffer> particleBuffers;
    std::vector<VkDeviceMemory> particleBuffersMemory;
    std::vector<VkBuffer> particleCountBuffers;
    std::vector<VkDeviceMemory> particleCountBuffersMemory;
    std::vector<VkBuffer> wireframeBuffers;
    std::vector<VkDeviceMemory> wireframeBuffersMemory;
    std::vector<VkBuffer> wireframeIndexBuffers;
    std::vector<VkDeviceMemory> wireframeIndexBuffersMemory;
    std::vector<VkBuffer> wireframeCountBuffers;
    std::vector<VkDeviceMemory> wireframeCountBuffersMemory;
    std::vector<VkBuffer> surfaceBuffers;
    std::vector<VkDeviceMemory> surfaceBuffersMemory;
    std::vector<VkBuffer> surfaceCountBuffers;
    std::vector<VkDeviceMemory> surfaceCountBuffersMemory;
    std::vector<VkBuffer> skyboxBuffers;
    std::vector<VkDeviceMemory> skyboxBuffersMemory;
    std::vector<VkBuffer> skyboxIndexBuffers;
    std::vector<VkDeviceMemory> skyboxIndexBuffersMemory;

    std::vector<std::vector<VkImage>> upscaleImages;
    std::vector<std::vector<VkDeviceMemory>> upscaleImagesMemory;
    std::vector<std::vector<VkImageView>> upscaleImagesView;
    std::vector<VkSampler> upscaleImagesSampler;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::vector<VkBuffer> renderingUBOBuffers;
    std::vector<VkDeviceMemory> renderingUBOBuffersMemory;
    std::vector<void*> renderingUBOBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> graphicsDescriptorSets;
    std::vector<VkDescriptorSet> raytraceDescriptorSets;
    std::vector<std::vector<VkDescriptorSet>> upscaleDescriptorSets;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    std::vector<VkDescriptorSet> filteredDescriptorSets;
    std::vector<VkDescriptorSet> postprocessDescriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> computeInFlightFences;

#ifdef DEBUGTOOLS
    // debug particles
    bool debugParticleFlag = false;
    std::vector<Vertex> debugParticles;
    std::vector<VkBuffer> debugParticleBuffers;
    std::vector<VkDeviceMemory> debugParticleBuffersMemory;
    VkPipelineLayout debugParticlePipelineLayout;
    VkPipeline debugParticlePipeline;
    VkDescriptorSetLayout debugParticleDescriptorSetLayout;
    std::vector<VkDescriptorSet> debugParticleDescriptorSets;
#endif

    // model
    std::vector<VkImage> modelImages;
    std::vector<VkDeviceMemory> modelImageMemorys;
    std::vector<VkImageView> modelImageViews;
    std::vector<VkSampler> modelSamplers;
    std::vector<std::vector<VkBuffer>> modelVertexBuffers;
    std::vector<std::vector<VkDeviceMemory>> modelVertexBuffersMemory;
    std::vector<std::vector<VkBuffer>> modelTMBuffers;
    std::vector<std::vector<VkDeviceMemory>> modelTMBuffersMemory;
    std::vector<std::vector<VkBuffer>> modelUniformBuffers;
    std::vector<std::vector<VkDeviceMemory>> modelUniformBuffersMemory;
    std::vector<std::vector<void*>> modelUniformBuffersMapped;
    std::vector<std::vector<ModelPushConstants>> modelPushConstants;
    VkDescriptorSetLayout modelDescriptorSetLayout;
    std::vector<std::vector<VkDescriptorSet>> modelDescriptorSets;
    VkPipelineLayout modelPipelineLayout;
    VkPipeline modelPipeline;

    // imgui
    VkRenderPass imGuiRenderPass;
    std::vector<VkFramebuffer> imGuiFrameBuffers;
    VkDescriptorPool imGuiDescriptorPool;
    VkCommandPool imGuiCommandPool;
    std::vector<VkCommandBuffer> imGuiCommandBuffers;

    void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    void initVulkan();
    void mainLoop();
    void cleanupSwapChain();
    void cleanup();
    void recreateSwapChain();
    void createInstance();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsDescriptorSetLayout();
    void createComputeDescriptorSetLayout();
    void updateDescriptorSets();
    void createGraphicsPipelineLayout();
    void createComputePipelineLayout();
    void createGraphicsPipeline();
    void createSkyBoxPipeline();
    void createComputePipeline();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
    void createSkybox();
    void createVertexBuffers();
    void createIndexBuffers();
    void createShaderStorageBuffers();
    void createUniformBuffers();
    void createDescriptorPool();
    void loadModel();
    void createGraphicsDescriptorSets();
    void createComputeDescriptorSets();
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createCommandBuffers();
    void createComputeCommandBuffers();
    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBuffer ImGuiCommandBuffer, uint32_t imageIndex);
    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer);
    // initialize ImGui
    void createGuiDescriptorPool();
    void createGuiRenderPass();
    void createGuiFrameBuffers();
    void createGuiCommandBuffers();
    void initImGui();
    void cleanupImGui();
    void createSyncObjects();
    void updateUniformBuffer(uint32_t currentImage);
    void drawFrame();
    void voxelize_mesh(const Mesh* mesh, const std::vector<glm::vec3>& particles, const int flag);
    void render_mesh(const Mesh* mesh);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange);
    bool hasStencilComponent(VkFormat format);
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    std::vector<const char*> getRequiredExtensions();
    bool checkValidationLayerSupport();
    static std::vector<char> readFile(const std::string& filename);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
#ifdef DEBUGTOOLS
    void debugParticle(Mesh* mesh);
    void cleanupDEBUGTOOLS();
#endif
};