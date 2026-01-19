#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include "LBM.h"

uint32_t WIDTH = 2000;
uint32_t HEIGHT = 1000;
static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

void LBM::init(uint Nx, uint Ny, uint Nz, float niu, float sigmas) {
    this->Nx = Nx;
    this->Ny = Ny;
    this->Nz = Nz;
    this->Nxyz = Nx * Ny * Nz;

    distance = distanceFactor * max(Nx, max(Ny, Nz));

    vels.resize(Nxyz * 3, 0.0f);
    rhos.resize(Nxyz, 1.0f);
    flags.resize(Nxyz, 0);
    cfs.resize(Nxyz * 3, 0.0f);

    renderUbo.transmittance = 0.25f;
    simulateUbo.niu = niu;
    simulateUbo.fx = 0.0f;
    simulateUbo.fy = 0.0f;
    simulateUbo.fz = -0.001f;
    simulateUbo.sigmas = sigmas;

    simulateUbo.smoothness = 1.0f;
}

void LBM::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    lastTime = glfwGetTime();
}

void LBM::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<LBM*>(glfwGetWindowUserPointer(window));
    WIDTH = width;
    HEIGHT = height;
    app->framebufferResized = true;
}

void LBM::mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
    auto app = reinterpret_cast<LBM*>(glfwGetWindowUserPointer(window));

    if (!app->mouseFree) return;

    if (app->firstMouse) {
        app->lastX = xpos;
        app->lastY = ypos;
        app->firstMouse = false;
    }

    float xoffset = xpos - app->lastX;
    float yoffset = app->lastY - ypos;
    app->lastX = xpos;
    app->lastY = ypos;

    float sensitivity = 0.5f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    app->rx -= xoffset;
    app->ry += yoffset;

    if (app->ry > 179.0f)
        app->ry = 179.0f;
    if (app->ry < 1.0f)
        app->ry = 1.0f;
}

void LBM::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto app = reinterpret_cast<LBM*>(glfwGetWindowUserPointer(window));

    if (!app->mouseFree) return;
    // 当鼠标向上滚轮时，yoffset=1，产生场景放大效果；
    // 当鼠标向下滚轮时，yoffset=-1，产生场景缩小效果。
    if (yoffset > 0) {
        app->distanceFactor += 0.1f;
    }
    else if (yoffset < 0) {
        app->distanceFactor -= 0.1f;
    }
    if (app->distanceFactor < 0.1f) {
        app->distanceFactor = 0.1f;
    }
    app->distance = app->distanceFactor * max(app->Nx, max(app->Ny, app->Nz));
}

void LBM::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    auto app = reinterpret_cast<LBM*>(glfwGetWindowUserPointer(window));

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        app->mouseFree = 1 - app->mouseFree;
        if (!app->mouseFree) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwSetCursorPos(window, WIDTH / 2.0, HEIGHT / 2.0);
            app->firstMouse = true;
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
}

void LBM::initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsDescriptorSetLayout();
    createComputeDescriptorSetLayout();
    createGraphicsPipelineLayout();
    createComputePipelineLayout();
    createGraphicsPipeline();
    createSkyBoxPipeline();
    createComputePipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createSkybox();
    createVertexBuffers();
    createIndexBuffers();
    createShaderStorageBuffers();
    createUniformBuffers();
    createDescriptorPool();
    loadModel();
    createGraphicsDescriptorSets();
    createComputeDescriptorSets();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();

    //imgui
    createGuiDescriptorPool();
    createGuiRenderPass();
    createGuiFrameBuffers();
    createGuiCommandBuffers();
    initImGui();
}

void LBM::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        static double start_time = glfwGetTime();
        glfwPollEvents();
        drawFrame();
        // We want to animate the particle system using the last frames time to get smooth, frame-rate independent animation
        double time = glfwGetTime();
        lastFrameTime = (time - lastTime) * 1000.0;
        totalTime = glfwGetTime() - start_time;
        lastTime = time;
    }

    vkDeviceWaitIdle(device);
}

void LBM::cleanupSwapChain() {
    vkDestroyImage(device, depthImage, nullptr);
    vkDestroyImageView(device, depthImageView, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
    vkDestroySampler(device, depthImageSampler, nullptr);
    vkDestroyImage(device, fluidDepthImage, nullptr);
    vkDestroyImageView(device, fluidDepthImageView, nullptr);
    vkFreeMemory(device, fluidDepthImageMemory, nullptr);
    vkDestroySampler(device, fluidDepthImageSampler, nullptr);
    vkDestroyImage(device, thickImage, nullptr);
    vkDestroyImageView(device, thickImageView, nullptr);
    vkFreeMemory(device, thickImageMemory, nullptr);
    vkDestroySampler(device, thickImageSampler, nullptr);
    vkDestroyImage(device, filteredFluidDepthImage, nullptr);
    vkDestroyImageView(device, filteredFluidDepthImageView, nullptr);
    vkFreeMemory(device, filteredFluidDepthImageMemory, nullptr);
    vkDestroySampler(device, filteredFluidDepthImageSampler, nullptr);
    vkDestroyImage(device, backgroundImage, nullptr);
    vkDestroyImageView(device, backgroundImageView, nullptr);
    vkFreeMemory(device, backgroundImageMemory, nullptr);
    vkDestroySampler(device, backgroundImageSampler, nullptr);

    for (int i = 0; i < swapChainImages.size(); ++i) {
        for (int j = 0; j < upscaleTimes; ++j) {
            vkDestroyImage(device, upscaleImages[i][j], nullptr);
            vkDestroyImageView(device, upscaleImagesView[i][j], nullptr);
            vkFreeMemory(device, upscaleImagesMemory[i][j], nullptr);
        }
        vkDestroySampler(device, upscaleImagesSampler[i], nullptr);
        vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        vkDestroyFramebuffer(device, fluidsFramebuffers[i], nullptr);
        vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }

    //for (auto framebuffer : swapChainFramebuffers) {
    //    vkDestroyFramebuffer(device, framebuffer, nullptr);
    //}
    //for (auto framebuffer : fluidsFramebuffers) {
    //    vkDestroyFramebuffer(device, framebuffer, nullptr);
    //}
    //
    //for (auto imageView : swapChainImageViews) {
    //    vkDestroyImageView(device, imageView, nullptr);
    //}

    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

void LBM::cleanup() {
    cleanupSwapChain();

    cleanupImGui();
    cleanupDEBUGTOOLS();

    vkDestroyPipeline(device, particlePipeline, nullptr);
    vkDestroyPipeline(device, wireframePipeline, nullptr);
    vkDestroyPipeline(device, surfacePipeline, nullptr);
    vkDestroyPipeline(device, skyboxPipeline, nullptr);
    vkDestroyPipeline(device, modelPipeline, nullptr);
    vkDestroyPipeline(device, raytracePipeline, nullptr);
    vkDestroyPipeline(device, upscalePipeline, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipeline(device, initPipeline, nullptr);
    vkDestroyPipeline(device, surface0Pipeline, nullptr);
    vkDestroyPipeline(device, surface1Pipeline, nullptr);
    vkDestroyPipeline(device, surface2Pipeline, nullptr);
    vkDestroyPipeline(device, surface3Pipeline, nullptr);
    vkDestroyPipeline(device, collideAndStreamPipeline, nullptr);
    vkDestroyPipeline(device, filteredPipeline, nullptr);
    vkDestroyPipeline(device, postprocessPipeline, nullptr);
    vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, modelPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, raytracePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, upscalePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, filteredPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, postprocessPipelineLayout, nullptr);

    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyRenderPass(device, fluidGraphicRenderPass, nullptr);

    vkDestroyImage(device, skyboxImage, nullptr);
    vkFreeMemory(device, skyboxImageMemory, nullptr);
    vkDestroyImageView(device, skyboxImageView, nullptr);
    vkDestroySampler(device, skyboxSampler, nullptr);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, particleBuffers[i], nullptr);
        vkFreeMemory(device, particleBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, renderingUBOBuffers[i], nullptr);
        vkFreeMemory(device, renderingUBOBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, stateBuffers[i], nullptr);
        vkFreeMemory(device, stateBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, velocityBuffers[i], nullptr);
        vkFreeMemory(device, velocityBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, rhoBuffers[i], nullptr);
        vkFreeMemory(device, rhoBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, flagBuffers[i], nullptr);
        vkFreeMemory(device, flagBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, DDF1Buffers[i], nullptr);
        vkFreeMemory(device, DDF1BuffersMemory[i], nullptr);
        vkDestroyBuffer(device, DDF2Buffers[i], nullptr);
        vkFreeMemory(device, DDF2BuffersMemory[i], nullptr);
        vkDestroyBuffer(device, cellForceBuffers[i], nullptr);
        vkFreeMemory(device, cellForceBuffersMemoryBuffers[i], nullptr);
        vkDestroyBuffer(device, PhiBuffers[i], nullptr);
        vkFreeMemory(device, PhiBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, wireframeBuffers[i], nullptr);
        vkFreeMemory(device, wireframeBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, wireframeIndexBuffers[i], nullptr);
        vkFreeMemory(device, wireframeIndexBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, surfaceBuffers[i], nullptr);
        vkFreeMemory(device, surfaceBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, skyboxBuffers[i], nullptr);
        vkFreeMemory(device, skyboxBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, skyboxIndexBuffers[i], nullptr);
        vkFreeMemory(device, skyboxIndexBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, MassBuffers[i], nullptr);
        vkFreeMemory(device, MassBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, MassExBuffers[i], nullptr);
        vkFreeMemory(device, MassExBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, particleCountBuffers[i], nullptr);
        vkFreeMemory(device, particleCountBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, wireframeCountBuffers[i], nullptr);
        vkFreeMemory(device, wireframeCountBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, surfaceCountBuffers[i], nullptr);
        vkFreeMemory(device, surfaceCountBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorSetLayout(device, graphicsDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, modelDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, raytraceDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, upscaleDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, filteredDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, postprocessDescriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
        vkDestroyFence(device, computeInFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    // clear model
    for (uint i = 0; i < modelImages.size(); ++i) {
        vkDestroyImage(device, modelImages[i], nullptr);
        vkFreeMemory(device, modelImageMemorys[i], nullptr);
        vkDestroyImageView(device, modelImageViews[i], nullptr);
        vkDestroySampler(device, modelSamplers[i], nullptr);
        for (uint j = 0; j < MAX_FRAMES_IN_FLIGHT; j++) {
            vkDestroyBuffer(device, modelVertexBuffers[i][j], nullptr);
            vkFreeMemory(device, modelVertexBuffersMemory[i][j], nullptr);
            vkDestroyBuffer(device, modelTMBuffers[i][j], nullptr);
            vkFreeMemory(device, modelTMBuffersMemory[i][j], nullptr);
            vkDestroyBuffer(device, modelUniformBuffers[i][j], nullptr);
            vkFreeMemory(device, modelUniformBuffersMemory[i][j], nullptr);
        }
    }

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
}

void LBM::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    // cleanup imgui
    for (auto framebuffer : imGuiFrameBuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    // cleanup imageSize
    imageSize.clear();
    cleanupSwapChain();
    for (uint i = 0; i < swapChainImages.size(); ++i) {
        upscaleImages[i].clear();
        upscaleImagesMemory[i].clear();
        upscaleImagesView[i].clear();
    }

    createSwapChain();
    createImageViews();
    createDepthResources();
    createFramebuffers();
    updateDescriptorSets();

    createGuiFrameBuffers();
}

void LBM::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);

        //createInfo.pNext = &debugCreateInfo;
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void LBM::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    //createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void LBM::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void LBM::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void LBM::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void LBM::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    //VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures;
    //atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    //atomicFloatFeatures.pNext = nullptr;
    //atomicFloatFeatures.shaderBufferFloat32Atomics = true; // this allows to perform atomic operations on storage buffers
    //atomicFloatFeatures.shaderBufferFloat32AtomicAdd = true; // this allows to perform atomic operations on storage buffers
    //atomicFloatFeatures.shaderBufferFloat64Atomics = false;
    //atomicFloatFeatures.shaderBufferFloat64AtomicAdd = false;
    //atomicFloatFeatures.shaderSharedFloat32Atomics = false;
    //atomicFloatFeatures.shaderSharedFloat32AtomicAdd = false;
    //atomicFloatFeatures.shaderSharedFloat64Atomics = false;
    //atomicFloatFeatures.shaderSharedFloat64AtomicAdd = false;
    //atomicFloatFeatures.shaderImageFloat32Atomics = false;
    //atomicFloatFeatures.shaderImageFloat32AtomicAdd = false;
    //atomicFloatFeatures.sparseImageFloat32Atomics = false;
    //atomicFloatFeatures.sparseImageFloat32AtomicAdd = false;

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.shaderInt64 = VK_TRUE;
    deviceFeatures.independentBlend = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    // ================== 特性链表设置 (FP16 & Storage) ==================

    // 1. FP16 运算特性 (VK_KHR_shader_float16_int8)
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR float16Features{};
    float16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
    float16Features.shaderFloat16 = VK_TRUE; // 开启 FP16 算术运算

    // 2. 16-bit 存储特性 (VK_KHR_16bit_storage)
    VkPhysicalDevice16BitStorageFeatures storage16Features{};
    storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

    // 允许 SSBO (Storage Buffer) 使用 float16
    storage16Features.storageBuffer16BitAccess = VK_TRUE;

    // 【关键修复】允许 UBO (Uniform Buffer) 使用 float16
    // 解决报错: SPIR-V contains an 16-bit OpVariable with Uniform Storage Class...
    storage16Features.uniformAndStorageBuffer16BitAccess = VK_TRUE;

    // 链接链表: storage16 -> float16
    storage16Features.pNext = &float16Features;

    // 3. 使用 Features2 包装器传递所有特性
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features = deviceFeatures; // 复制原有的基础特性

    // 链接链表: features2 -> storage16 -> float16
    deviceFeatures2.pNext = &storage16Features;

    // ===================================================================

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    // 使用 pNext 链传递扩展特性
    createInfo.pNext = &deviceFeatures2;

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    // 当使用 pNext (VkPhysicalDeviceFeatures2) 时，pEnabledFeatures 必须为 nullptr
    // createInfo.pEnabledFeatures = &deviceFeatures; 
    createInfo.pEnabledFeatures = nullptr;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

void LBM::createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsAndComputeFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void LBM::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void LBM::createRenderPass() {
    // universal
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = VK_FORMAT_R8G8B8A8_SRGB;
        //colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        //depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // particle
    {
        VkAttachmentDescription thickattachment{};
        thickattachment.format = VK_FORMAT_R32_SFLOAT;
        thickattachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        thickattachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        thickattachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        thickattachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        thickattachment.samples = VK_SAMPLE_COUNT_1_BIT;
        thickattachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        thickattachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        VkAttachmentDescription customdepthattachement{};
        customdepthattachement.format = VK_FORMAT_R32_SFLOAT;
        customdepthattachement.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        customdepthattachement.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        customdepthattachement.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        customdepthattachement.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        customdepthattachement.samples = VK_SAMPLE_COUNT_1_BIT;
        customdepthattachement.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        customdepthattachement.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        std::array<VkAttachmentDescription, 2> attachments = { thickattachment,customdepthattachement };
        VkAttachmentReference thickattachment_ref{};
        thickattachment_ref.attachment = 0;
        thickattachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference customdepthattachment_ref{};
        customdepthattachment_ref.attachment = 1;
        customdepthattachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        std::array<VkAttachmentReference, 2> colorattachment_ref = { customdepthattachment_ref,thickattachment_ref };
        std::array<VkSubpassDescription, 1> subpasses{};
        subpasses[0].colorAttachmentCount = static_cast<uint32_t>(colorattachment_ref.size());
        subpasses[0].pColorAttachments = colorattachment_ref.data();
        subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

        VkRenderPassCreateInfo createinfo{};
        createinfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createinfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        createinfo.pAttachments = attachments.data();
        createinfo.subpassCount = static_cast<uint32_t>(subpasses.size());
        createinfo.pSubpasses = subpasses.data();

        if (vkCreateRenderPass(device, &createinfo, nullptr, &fluidGraphicRenderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fluid graphic renderpass!");
        }
    }
}

void LBM::createGraphicsDescriptorSetLayout() {
    // universal
    {
        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // sampler
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &graphicsDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // model
    {
        std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // points
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // tms
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        // texture
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &modelDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create model descriptor set layout!");
        }
    }
}

void LBM::createComputeDescriptorSetLayout() {
    // universal
    {
        std::array<VkDescriptorSetLayoutBinding, 19> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

        // particles
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // velocity
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // rho
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // flags
        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // DDF
        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[5].pImmutableSamplers = nullptr;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // cellForce
        layoutBindings[6].binding = 6;
        layoutBindings[6].descriptorCount = 1;
        layoutBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[6].pImmutableSamplers = nullptr;
        layoutBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // Phi
        layoutBindings[7].binding = 7;
        layoutBindings[7].descriptorCount = 1;
        layoutBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[7].pImmutableSamplers = nullptr;
        layoutBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // wireframeVertex
        layoutBindings[8].binding = 8;
        layoutBindings[8].descriptorCount = 1;
        layoutBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[8].pImmutableSamplers = nullptr;
        layoutBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // wireframeIndex
        layoutBindings[9].binding = 9;
        layoutBindings[9].descriptorCount = 1;
        layoutBindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[9].pImmutableSamplers = nullptr;
        layoutBindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // surfaceVertex
        layoutBindings[10].binding = 10;
        layoutBindings[10].descriptorCount = 1;
        layoutBindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[10].pImmutableSamplers = nullptr;
        layoutBindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // Mass
        layoutBindings[11].binding = 11;
        layoutBindings[11].descriptorCount = 1;
        layoutBindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[11].pImmutableSamplers = nullptr;
        layoutBindings[11].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // MassEx
        layoutBindings[12].binding = 12;
        layoutBindings[12].descriptorCount = 1;
        layoutBindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[12].pImmutableSamplers = nullptr;
        layoutBindings[12].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // particleCount
        layoutBindings[13].binding = 13;
        layoutBindings[13].descriptorCount = 1;
        layoutBindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[13].pImmutableSamplers = nullptr;
        layoutBindings[13].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // wireframeCount
        layoutBindings[14].binding = 14;
        layoutBindings[14].descriptorCount = 1;
        layoutBindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[14].pImmutableSamplers = nullptr;
        layoutBindings[14].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // surfaceCount
        layoutBindings[15].binding = 15;
        layoutBindings[15].descriptorCount = 1;
        layoutBindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[15].pImmutableSamplers = nullptr;
        layoutBindings[15].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // DDF
        layoutBindings[16].binding = 16;
        layoutBindings[16].descriptorCount = 1;
        layoutBindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[16].pImmutableSamplers = nullptr;
        layoutBindings[16].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // State
        layoutBindings[17].binding = 17;
        layoutBindings[17].descriptorCount = 1;
        layoutBindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[17].pImmutableSamplers = nullptr;
        layoutBindings[17].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // totalMass
        layoutBindings[18].binding = 18;
        layoutBindings[18].descriptorCount = 1;
        layoutBindings[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[18].pImmutableSamplers = nullptr;
        layoutBindings[18].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    // raytrace
    {
        std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings{};

        // uniform buffer
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // framebuffer
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // skybox
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // phi
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // flags
        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &raytraceDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create raytrace descriptor set layout!");
        }
    }

    // upscale
    {
        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &upscaleDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create upscale descriptor set layout!");
        }
    }

    // filtered
    {
        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &filteredDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create filtered descriptor set layout!");
        }
    }

    // postprocess
    {
        std::array<VkDescriptorSetLayoutBinding, 6> layoutBindings{};

        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[5].pImmutableSamplers = nullptr;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &postprocessDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create postprocess descriptor set layout!");
        }
    }
}

void LBM::createGraphicsPipelineLayout() {
    // universal
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &graphicsDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // model
    {
        VkPushConstantRange modelPushConstantRange{};
        modelPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        modelPushConstantRange.offset = 0;
        modelPushConstantRange.size = sizeof(ModelPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &modelDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &modelPushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &modelPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }
}

void LBM::createComputePipelineLayout() {
    // universal pipelinelayout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
    }

    // raytrace pipelinelayout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &raytraceDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &raytracePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create raytrace pipeline layout!");
        }
    }

    // upscale pipelinelayout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &upscaleDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &upscalePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create upscale pipeline layout!");
        }
    }

    // filtered pipelinelayout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &filteredDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &filteredPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create filtered pipeline layout!");
        }
    }

    // postprocess pipelinelayout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &postprocessDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &postprocessPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create postprocess pipeline layout!");
        }
    }
}

void LBM::createGraphicsPipeline() {
    auto particleVertShaderCode = readFile("shaders/particle_vert.spv");
    auto particleFragShaderCode = readFile("shaders/particle_frag.spv");
    VkShaderModule particleVertShaderModule = createShaderModule(particleVertShaderCode);
    VkShaderModule particleFragShaderModule = createShaderModule(particleFragShaderCode);
    VkPipelineShaderStageCreateInfo particleVertShaderStageInfo{};
    particleVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    particleVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    particleVertShaderStageInfo.module = particleVertShaderModule;
    particleVertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo particleFragShaderStageInfo{};
    particleFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    particleFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    particleFragShaderStageInfo.module = particleFragShaderModule;
    particleFragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo particleShaderStages[] = { particleVertShaderStageInfo, particleFragShaderStageInfo };

    auto wireframeVertShaderCode = readFile("shaders/wireframe_vert.spv");
    auto wireframeFragShaderCode = readFile("shaders/wireframe_frag.spv");
    VkShaderModule wireframeVertShaderModule = createShaderModule(wireframeVertShaderCode);
    VkShaderModule wireframeFragShaderModule = createShaderModule(wireframeFragShaderCode);
    VkPipelineShaderStageCreateInfo wireframeVertShaderStageInfo{};
    wireframeVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    wireframeVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    wireframeVertShaderStageInfo.module = wireframeVertShaderModule;
    wireframeVertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo wireframeFragShaderStageInfo{};
    wireframeFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    wireframeFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    wireframeFragShaderStageInfo.module = wireframeFragShaderModule;
    wireframeFragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo wireframeShaderStages[] = { wireframeVertShaderStageInfo, wireframeFragShaderStageInfo };

    auto surfaceVertShaderCode = readFile("shaders/surface_vert.spv");
    auto surfaceFragShaderCode = readFile("shaders/surface_frag.spv");
    VkShaderModule surfaceVertShaderModule = createShaderModule(surfaceVertShaderCode);
    VkShaderModule surfaceFragShaderModule = createShaderModule(surfaceFragShaderCode);
    VkPipelineShaderStageCreateInfo surfaceVertShaderStageInfo{};
    surfaceVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    surfaceVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    surfaceVertShaderStageInfo.module = surfaceVertShaderModule;
    surfaceVertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo surfaceFragShaderStageInfo{};
    surfaceFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    surfaceFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    surfaceFragShaderStageInfo.module = surfaceFragShaderModule;
    surfaceFragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo surfaceShaderStages[] = { surfaceVertShaderStageInfo, surfaceFragShaderStageInfo };

    auto modelVertShaderCode = readFile("shaders/model_vert.spv");
    auto modelFragShaderCode = readFile("shaders/model_frag.spv");
    VkShaderModule modelVertShaderModule = createShaderModule(modelVertShaderCode);
    VkShaderModule modelFragShaderModule = createShaderModule(modelFragShaderCode);
    VkPipelineShaderStageCreateInfo modelVertShaderStageInfo{};
    modelVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    modelVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    modelVertShaderStageInfo.module = modelVertShaderModule;
    modelVertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo modelFragShaderStageInfo{};
    modelFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    modelFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    modelFragShaderStageInfo.module = modelFragShaderModule;
    modelFragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo modelShaderStages[] = { modelVertShaderStageInfo, modelFragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // model
    {
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = modelShaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = modelPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &modelPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    // particle
    {
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        VkPipelineColorBlendAttachmentState depthblendattachment{};
        depthblendattachment.blendEnable = VK_TRUE;
        depthblendattachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        depthblendattachment.colorBlendOp = VK_BLEND_OP_MIN;
        depthblendattachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        depthblendattachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        VkPipelineColorBlendAttachmentState thickblendattachment{};
        thickblendattachment.blendEnable = VK_TRUE;
        thickblendattachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        thickblendattachment.colorBlendOp = VK_BLEND_OP_ADD;
        thickblendattachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        thickblendattachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        thickblendattachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        thickblendattachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        std::vector<VkPipelineColorBlendAttachmentState> fluidcolorblendattachments = { depthblendattachment, thickblendattachment };
        VkPipelineColorBlendStateCreateInfo colorblending{};
        colorblending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorblending.attachmentCount = static_cast<uint32_t>(fluidcolorblendattachments.size());
        colorblending.logicOpEnable = VK_FALSE;
        colorblending.pAttachments = fluidcolorblendattachments.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.flags = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = particleShaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorblending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = fluidGraphicRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &particlePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create particle pipeline!");
        }
    }

    // surface
    {
        bindingDescription = Vertex::getBindingDescription();
        attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = surfaceShaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surfacePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    // wireframe
    {
        bindingDescription = Vertex::getBindingDescription();
        attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = wireframeShaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &wireframePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    vkDestroyShaderModule(device, particleFragShaderModule, nullptr);
    vkDestroyShaderModule(device, particleVertShaderModule, nullptr);
    vkDestroyShaderModule(device, wireframeFragShaderModule, nullptr);
    vkDestroyShaderModule(device, wireframeVertShaderModule, nullptr);
    vkDestroyShaderModule(device, surfaceFragShaderModule, nullptr);
    vkDestroyShaderModule(device, surfaceVertShaderModule, nullptr);
    vkDestroyShaderModule(device, modelFragShaderModule, nullptr);
    vkDestroyShaderModule(device, modelVertShaderModule, nullptr);
}

void LBM::createSkyBoxPipeline() {
    auto vertShaderCode = readFile("shaders/skybox_vert.spv");
    auto fragShaderCode = readFile("shaders/skybox_frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(glm::vec3);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = 0;

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = graphicsPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyboxPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void LBM::createComputePipeline() {
    // init
    {
        auto computeShaderCode = readFile("shaders/init_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &initPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // collide_and_stream
    {
        auto computeShaderCode = readFile("shaders/collide_and_stream_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &collideAndStreamPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // calc
    {
        auto computeShaderCode = readFile("shaders/calc_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // surface0
    {
        auto computeShaderCode = readFile("shaders/surface0_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surface0Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // surface1
    {
        auto computeShaderCode = readFile("shaders/surface1_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surface1Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // surface2
    {
        auto computeShaderCode = readFile("shaders/surface2_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surface2Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // surface3
    {
        auto computeShaderCode = readFile("shaders/surface3_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &surface3Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // filtered
    {
        auto computeShaderCode = readFile("shaders/filtered_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = filteredPipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &filteredPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create filtered pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // postprocess
    {
        auto computeShaderCode = readFile("shaders/postprocess_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = postprocessPipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &postprocessPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create postprocess pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }
    
    // raytrace
    {
        auto computeShaderCode = readFile("shaders/raytrace_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = raytracePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &raytracePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create raytrace pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    // upscale
    {
        auto computeShaderCode = readFile("shaders/upscale_comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = upscalePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &upscalePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create upscale pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }
}

void LBM::createFramebuffers() {
    // universal
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                //swapChainImageViews[i],
                backgroundImageView,
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // particle
    {
        fluidsFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> fluidsattachments = {
                thickImageView,
                fluidDepthImageView
            };

            VkFramebufferCreateInfo fluidsframebufferinfo{};
            fluidsframebufferinfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fluidsframebufferinfo.attachmentCount = static_cast<uint32_t>(fluidsattachments.size());
            fluidsframebufferinfo.pAttachments = fluidsattachments.data();
            fluidsframebufferinfo.width = swapChainExtent.width;
            fluidsframebufferinfo.height = swapChainExtent.height;
            fluidsframebufferinfo.layers = 1;
            fluidsframebufferinfo.renderPass = fluidGraphicRenderPass;

            if (vkCreateFramebuffer(device, &fluidsframebufferinfo, nullptr, &fluidsFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create fluids framebuffer!");
            }
        }
    }
}

void LBM::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void LBM::createDepthResources() {
    // depth image
    {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;
        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, subresourceRange);

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.anisotropyEnable = VK_FALSE;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        vkCreateSampler(device, &samplerinfo, nullptr, &depthImageSampler);
    }

    // custom depth iamge
    {
        createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fluidDepthImage, fluidDepthImageMemory);
        fluidDepthImageView = createImageView(fluidDepthImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;
        transitionImageLayout(fluidDepthImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, subresourceRange);

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.anisotropyEnable = VK_FALSE;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        vkCreateSampler(device, &samplerinfo, nullptr, &fluidDepthImageSampler);
    }

    // filtered depth image
    {
        createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, filteredFluidDepthImage, filteredFluidDepthImageMemory);
        filteredFluidDepthImageView = createImageView(filteredFluidDepthImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.anisotropyEnable = VK_FALSE;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        vkCreateSampler(device, &samplerinfo, nullptr, &filteredFluidDepthImageSampler);
    }

    // thick image
    {
        createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, thickImage, thickImageMemory);
        thickImageView = createImageView(thickImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;
        transitionImageLayout(thickImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, subresourceRange);

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.anisotropyEnable = VK_FALSE;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        vkCreateSampler(device, &samplerinfo, nullptr, &thickImageSampler);
    }

    // background image
    {
        createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, backgroundImage, backgroundImageMemory);
        backgroundImageView = createImageView(backgroundImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;
        transitionImageLayout(backgroundImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, subresourceRange);

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.anisotropyEnable = VK_FALSE;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        vkCreateSampler(device, &samplerinfo, nullptr, &backgroundImageSampler);
    }

    // upscale images
    {
        upscaleImages.resize(swapChainImages.size());
        upscaleImagesMemory.resize(swapChainImages.size());
        upscaleImagesView.resize(swapChainImages.size());
        upscaleImagesSampler.resize(swapChainImages.size());

        for (uint i = 0; i < swapChainImages.size(); ++i) {
            uint imageWidth = swapChainExtent.width;
            uint imageHeight = swapChainExtent.height;
            for (uint j = 0; j < upscaleTimes; ++j) {
                if (i == 0) imageSize.push_back(glm::ivec2(imageWidth, imageHeight));
                imageWidth /= 2;
                imageHeight /= 2;

                VkImage upscaleImage;
                VkDeviceMemory upscaleImageMemory;
                VkImageView upscaleImageView;
                createImage(imageWidth, imageHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, upscaleImage, upscaleImageMemory);
                upscaleImageView = createImageView(upscaleImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

                VkImageSubresourceRange subresourceRange = {};
                subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                subresourceRange.baseMipLevel = 0;
                subresourceRange.levelCount = 1;
                subresourceRange.baseArrayLayer = 0;
                subresourceRange.layerCount = 1;
                transitionImageLayout(upscaleImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);

                upscaleImages[i].push_back(upscaleImage);
                upscaleImagesMemory[i].push_back(upscaleImageMemory);
                upscaleImagesView[i].push_back(upscaleImageView);
            }

            VkSamplerCreateInfo samplerinfo{};
            samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerinfo.anisotropyEnable = VK_FALSE;
            samplerinfo.minFilter = VK_FILTER_LINEAR;
            samplerinfo.magFilter = VK_FILTER_LINEAR;
            samplerinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerinfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            vkCreateSampler(device, &samplerinfo, nullptr, &upscaleImagesSampler[i]);
        }
    }
}

void LBM::createSkybox() {
    ktxResult result;
    ktxTexture* ktxTexture;

    result = ktxTexture_CreateFromNamedFile("textures/output.ktx", KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS, "cannot load skybox texture");

    const uint32_t width = ktxTexture->baseWidth;
    const uint32_t height = ktxTexture->baseHeight;
    const uint32_t mipLevels = ktxTexture->numLevels;

    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetDataSize(ktxTexture);

    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReqs;

    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = ktxTextureSize;
    // This buffer is used as a transfer source for the buffer copy
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create staging buffer!");
    }

    // Get memory requirements for the staging buffer (alignment, memory type bits)
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    // Get memory type index for a host visible buffer
    memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate staging memory!");
    }
    if (vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS) {
        throw std::runtime_error("failed to bind staging buffer memory!");
    }

    // Copy texture data into staging buffer
    uint8_t* data = nullptr;
    vkMapMemory(device, stagingMemory, 0, memReqs.size, 0, (void**)&data);
    memcpy(data, ktxTextureData, ktxTextureSize);
    vkUnmapMemory(device, stagingMemory);

    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    // Create optimal tiled target image
    {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = mipLevels;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.extent = { width, height, 1 };
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        // Cube faces count as array layers in Vulkan
        imageCreateInfo.arrayLayers = 6;
        // This flag is required for cube map images
        imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        if (vkCreateImage(device, &imageCreateInfo, nullptr, &skyboxImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox image!");
        }

        vkGetImageMemoryRequirements(device, skyboxImage, &memReqs);
        memAllocInfo.allocationSize = memReqs.size;
        memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &memAllocInfo, nullptr, &skyboxImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }
        if (vkBindImageMemory(device, skyboxImage, skyboxImageMemory, 0) != VK_SUCCESS) {
            throw std::runtime_error("failed to bind image memory!");
        }
    }

    std::vector<VkBufferImageCopy> bufferCopyRegions;

    uint32_t offset = 0;
    for (uint32_t face = 0; face < 6; face++)
    {
        for (uint32_t level = 0; level < mipLevels; level++)
        {
            // Calculate offset into staging buffer for the current mip level and face
            ktx_size_t offset;
            KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
            assert(ret == KTX_SUCCESS);
            VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = level;
            bufferCopyRegion.imageSubresource.baseArrayLayer = face;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
            bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = offset;
            bufferCopyRegions.push_back(bufferCopyRegion);
        }
    }

    // Image barrier for optimal image (target)
    // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 6;

    transitionImageLayout(skyboxImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

    VkCommandBuffer copyCmd = beginSingleTimeCommands();// Setup buffer copy regions for each face including all of its miplevels
    // Copy the cube map faces from the staging buffer to the optimal tiled image
    vkCmdCopyBufferToImage(
        copyCmd,
        stagingBuffer,
        skyboxImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(bufferCopyRegions.size()),
        bufferCopyRegions.data()
    );
    endSingleTimeCommands(copyCmd);

    VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transitionImageLayout(skyboxImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, imageLayout, subresourceRange);

    // Create sampler
    {
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.compareOp = VK_COMPARE_OP_NEVER;
        sampler.minLod = 0.0f;
        sampler.maxLod = static_cast<float>(mipLevels);
        sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler.maxAnisotropy = 1.0f;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        sampler.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        sampler.anisotropyEnable = VK_FALSE;
        if (vkCreateSampler(device, &sampler, nullptr, &skyboxSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox sampler!");
        }
    }

    // Create image view
    {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        // Cube map view type
        view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        view.format = format;
        view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        // 6 array layers (faces)
        view.subresourceRange.layerCount = 6;
        // Set number of mip levels
        view.subresourceRange.levelCount = mipLevels;
        view.image = skyboxImage;
        if (vkCreateImageView(device, &view, nullptr, &skyboxImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create skybox image view!");
        }
    }

    // Clean up staging resources
    vkFreeMemory(device, stagingMemory, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    ktxTexture_Destroy(ktxTexture);
}

void LBM::createVertexBuffers() {
    // wireframes
    {
        //VkDeviceSize bufferSize = sizeof(Vertex) * wireframeVertices.size();
        VkDeviceSize bufferSize = sizeof(Vertex) * Nxyz * 2;

        wireframeBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        wireframeBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        //memcpy(data, wireframeVertices.data(), (size_t)bufferSize);
        memset(data, 0, (uint32_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wireframeBuffers[i], wireframeBuffersMemory[i]);
            copyBuffer(stagingBuffer, wireframeBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // surfaces
    {
        //VkDeviceSize bufferSize = sizeof(Vertex) * surfaceVertices.size();
        VkDeviceSize bufferSize = sizeof(Vertex) * Nxyz * 3 * 5;

        surfaceBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        surfaceBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        //memcpy(data, surfaceVertices.data(), (size_t)bufferSize);
        memset(data, 0, (uint32_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, surfaceBuffers[i], surfaceBuffersMemory[i]);
            copyBuffer(stagingBuffer, surfaceBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // skybox
    {
        VkDeviceSize bufferSize = sizeof(skyboxVertices[0]) * skyboxVertices.size();
        skyboxBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        skyboxBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, skyboxVertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxBuffers[i], skyboxBuffersMemory[i]);
            copyBuffer(stagingBuffer, skyboxBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
}

void LBM::createIndexBuffers() {
    // wireframes
    {
        //VkDeviceSize bufferSize = sizeof(uint32_t) * wireframeIndices.size();
        VkDeviceSize bufferSize = sizeof(uint32_t) * Nxyz * 2;

        wireframeIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        wireframeIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        //memcpy(data, wireframeIndices.data(), (size_t)bufferSize);
        memset(data, 0, (uint32_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wireframeIndexBuffers[i], wireframeIndexBuffersMemory[i]);
            copyBuffer(stagingBuffer, wireframeIndexBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // skybox
    {
        VkDeviceSize bufferSize = sizeof(skyboxIndices[0]) * skyboxIndices.size();
        skyboxIndexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        skyboxIndexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, skyboxIndices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skyboxIndexBuffers[i], skyboxIndexBuffersMemory[i]);
            copyBuffer(stagingBuffer, skyboxIndexBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
}

void LBM::createShaderStorageBuffers() {
    // particles
    {
        VkDeviceSize bufferSize = sizeof(Particle) * Nxyz;

        // Create a staging buffer used to upload data to the gpu
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        particleBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        particleBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        // Copy initial particle data to all storage buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, particleBuffers[i], particleBuffersMemory[i]);
            copyBuffer(stagingBuffer, particleBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // States
    {
        // create state buffer, size = width * height * (vec2)
        VkDeviceSize bufferSize = Nxyz * 4 * sizeof(uint16_t);

        stateBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        stateBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        
        std::vector<uint16_t>states_half(Nxyz * 4);
        parallel_for(Nxyz, [&](uint32_t i) {
            // Velocity (x, y, z)
            states_half[i * 4 + 0] = glm::packHalf1x16(vels[0u * Nxyz + i]);
            states_half[i * 4 + 1] = glm::packHalf1x16(vels[1u * Nxyz + i]);
            states_half[i * 4 + 2] = glm::packHalf1x16(vels[2u * Nxyz + i]);
            // Density (w)
            states_half[i * 4 + 3] = glm::packHalf1x16(rhos[i]);
        });
        
        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, states_half.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        // create velocity buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, stateBuffers[i], stateBuffersMemory[i]);
            copyBuffer(stagingBuffer, stateBuffers[i], bufferSize);
        }
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // velocity
    {
        // create velocity buffer, size = width * height * (vec2)
        VkDeviceSize bufferSize = Nxyz * sizeof(uint16_t) * 3;
        velocityBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        velocityBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        std::vector<uint16_t>vels_half(vels.size());
        parallel_for(vels.size(), [&](uint32_t i) {
            // glm::packHalf1x16 将 float 转为 uint16_t (符合 IEEE 754 FP16 标准)
            vels_half[i] = glm::packHalf1x16(vels[i]);
            });

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vels_half.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create velocity buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, velocityBuffers[i], velocityBuffersMemory[i]);
            copyBuffer(stagingBuffer, velocityBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // rho
    {
        // create rho buffer, size = width * height * (float)
        VkDeviceSize bufferSize = Nxyz * sizeof(uint16_t);
        rhoBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        rhoBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        std::vector<float> rhos_half(rhos.size());
        parallel_for(rhos.size(), [&](uint32_t i) {
            // glm::packHalf1x16 将 float 转为 uint16_t (符合 IEEE 754 FP16 标准)
            rhos_half[i] = glm::packHalf1x16(rhos[i]);
            });

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, rhos_half.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create rho buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rhoBuffers[i], rhoBuffersMemory[i]);
            copyBuffer(stagingBuffer, rhoBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // flags
    {
        auto cube = [](const uint x, const uint y, const uint z, const glm::vec3& p, const float l) {
            const glm::vec3 t = glm::vec3(x, y, z) - p;
            return t.x >= -0.5f * l && t.x <= 0.5f * l && t.y >= -0.5f * l && t.y <= 0.5f * l && t.z >= -0.5f * l && t.z <= 0.5f * l;
            };

        auto cuboid = [](const uint x, const uint y, const uint z, const glm::vec3& p, const glm::vec3& l) {
            const glm::vec3 t = glm::vec3(x, y, z) - p;
            return t.x >= -0.5f * l.x && t.x <= 0.5f * l.x && t.y >= -0.5f * l.y && t.y <= 0.5f * l.y && t.z >= -0.5f * l.z && t.z <= 0.5f * l.z;
            };

        auto cylinder = [](const uint x, const uint y, const uint z, const glm::vec3& p, const glm::vec3& n, const float r) {
            const glm::vec3 t = glm::vec3(x, y, z) - p;
            const float sqnt = sq(glm::dot(glm::normalize(n), t));
            const float dist = sq(t.x) + sq(t.y) + sq(t.z) - sqnt;
            return dist <= sq(r) && sqnt <= sq(0.5f * glm::length(n));
            };

        // create flags buffer, size = width * height * (float)
        VkDeviceSize bufferSize = Nxyz * sizeof(uint32_t);
        flagBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        flagBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, flags.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create flags buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, flagBuffers[i], flagBuffersMemory[i]);
            copyBuffer(stagingBuffer, flagBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // DDF1
    {
        // create FOld buffer, size = width * height * (vecQ)
        VkDeviceSize bufferSize = Nxyz * 5 * 4 * sizeof(uint16_t);

        DDF1Buffers.resize(MAX_FRAMES_IN_FLIGHT);
        DDF1BuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create DDF buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DDF1Buffers[i], DDF1BuffersMemory[i]);
            copyBuffer(stagingBuffer, DDF1Buffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // DDF2
    {
        // create FOld buffer, size = width * height * (vecQ)
        VkDeviceSize bufferSize = Nxyz * 5 * 4 * sizeof(uint16_t);

        DDF2Buffers.resize(MAX_FRAMES_IN_FLIGHT);
        DDF2BuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create DDF buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DDF2Buffers[i], DDF2BuffersMemory[i]);
            copyBuffer(stagingBuffer, DDF2Buffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // cellForce
    {
        VkDeviceSize bufferSize = Nxyz * sizeof(float) * 3;

        cellForceBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        cellForceBuffersMemoryBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        //memset(data, 0, (size_t)bufferSize);
        memcpy(data, cfs.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create cellForce buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cellForceBuffers[i], cellForceBuffersMemoryBuffers[i]);
            copyBuffer(stagingBuffer, cellForceBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // Phi
    {
        // create phis buffer, size = width * height * (float)
        VkDeviceSize bufferSize = Nxyz * sizeof(float);
        PhiBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        PhiBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        std::vector<float> phis(Nxyz, 0);
        parallel_for(Nxyz, [&](uint32_t index) {
            uint x = index % Nx, y = (index - x) / Nx % Ny, z = index / Nx / Ny;
            //phis[index] = sqrt(sq(x - (Nx - 1.0f) / 2.0f) + sq(y - (Ny - 1.0f) / 2.0f) + sq(z - (Nz - 1.0f) / 2.0f));
            phis[index] = 0.0f;
            });

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, phis.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create phis buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, PhiBuffers[i], PhiBuffersMemory[i]);
            copyBuffer(stagingBuffer, PhiBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // Mass
    {
        // create Mass buffer, size = width * height * (vecQ)
        VkDeviceSize bufferSize = Nxyz * sizeof(float);

        MassBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        MassBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create Mass buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MassBuffers[i], MassBuffersMemory[i]);
            copyBuffer(stagingBuffer, MassBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // MassEx
    {
        // create Mass buffer, size = width * height * (vecQ)
        VkDeviceSize bufferSize = Nxyz * sizeof(float);

        MassExBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        MassExBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create MassEx buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, MassExBuffers[i], MassExBuffersMemory[i]);
            copyBuffer(stagingBuffer, MassExBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // particleCount
    {
        VkDeviceSize bufferSize = sizeof(VkDrawIndirectCommand);

        particleCountBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        particleCountBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkDrawIndirectCommand initCmd = {};
        initCmd.vertexCount = 0;
        initCmd.instanceCount = 1;
        initCmd.firstVertex = 0;
        initCmd.firstInstance = 0;

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, &initCmd, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        // create particleCount buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, particleCountBuffers[i], particleCountBuffersMemory[i]);
            copyBuffer(stagingBuffer, particleCountBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // wireframeCount
    {
        VkDeviceSize bufferSize = sizeof(VkDrawIndirectCommand);

        wireframeCountBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        wireframeCountBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkDrawIndirectCommand initCmd = {};
        initCmd.vertexCount = 0;
        initCmd.instanceCount = 1;
        initCmd.firstVertex = 0;
        initCmd.firstInstance = 0;

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, &initCmd, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create wireframeCount buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wireframeCountBuffers[i], wireframeCountBuffersMemory[i]);
            copyBuffer(stagingBuffer, wireframeCountBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // surfaceCount
    {
        VkDeviceSize bufferSize = sizeof(VkDrawIndirectCommand);

        surfaceCountBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        surfaceCountBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        VkDrawIndirectCommand initCmd = {};
        initCmd.vertexCount = 0;
        initCmd.instanceCount = 1;
        initCmd.firstVertex = 0;
        initCmd.firstInstance = 0;

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, &initCmd, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create surfaceCount buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, surfaceCountBuffers[i], surfaceCountBuffersMemory[i]);
            copyBuffer(stagingBuffer, surfaceCountBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // total mass
    {
        // create Mass buffer, size = width * height * (vecQ)
        VkDeviceSize bufferSize = 2 * sizeof(float);

        totalMassBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        totalMassBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memset(data, 0, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create totalMass buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, totalMassBuffers[i], totalMassBuffersMemory[i]);
            copyBuffer(stagingBuffer, totalMassBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
}

void LBM::createUniformBuffers() {
    VkDeviceSize bufferSize1 = sizeof(SimulateUBO);
    VkDeviceSize bufferSize2 = sizeof(RenderingUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    renderingUBOBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    renderingUBOBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    renderingUBOBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(bufferSize1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize1, 0, &uniformBuffersMapped[i]);
        createBuffer(bufferSize2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, renderingUBOBuffers[i], renderingUBOBuffersMemory[i]);
        vkMapMemory(device, renderingUBOBuffersMemory[i], 0, bufferSize2, 0, &renderingUBOBuffersMapped[i]);
    }
}

void LBM::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void LBM::loadModel() {
    //Mesh* stator = nullptr;
    //Mesh* rotor = nullptr;
    //const glm::vec3 center = glm::vec3(Nx / 2.0f - 0.5f, Ny / 2.0f - 0.5f, Nz / 2.0f - 0.5f);
    //const float3x3 rotation = float3x3(glm::vec3(0, 0, 1), radians(180.0f));
    //stator = read_stl("stl/edf_v39.stl", 1.0f, rotation); // https://www.thingiverse.com/thing:3014759/files
    //rotor = read_stl("stl/edf_v391.stl", 1.0f, rotation); // https://www.thingiverse.com/thing:3014759/files
    //plank = read_stl("stl/muban.stl", 1.0f);
    //plank = read_obj("models/muban.obj", "textures/test_muban_texture_001.png", 1.0f);
    //stator->model_name = "stator";
    //rotor->model_name = "rotor";
    //plank->model_name = "plank";
    //const float scale = 0.5f * stator->get_scale_for_box_fit(glm::vec3(Nx, Ny, Nz)); // scale stator and rotor to simulation box size
    //stator->scale(scale);
    //rotor->scale(scale);
    //plank->scale(10.0f);
    //stator->translate(center - stator->get_bounding_box_center() - glm::vec3(0.0f, 0.2f * stator->get_max_size(), 0.0f)); // move stator and rotor to simulation box center
    //rotor->translate(center - rotor->get_bounding_box_center() - glm::vec3(0.0f, 0.41f * stator->get_max_size(), 0.0f));
    //plank->translate(center - plank->get_bounding_box_center() - glm::vec3(0.0f, 0.0f, 0.0f));
    //stator->translate(glm::vec3(0.0f, 3.0f * Ny / 32.0f, 7.0f * Nz / -32.0f));
    //rotor->translate(glm::vec3(0.0f, 3.0f * Ny / 32.0f, 7.0f * Nz / -32.0f));
    //plank->translate(glm::vec3(0.0f, 9.0f * Ny / 32.0f, 5.0f * Nz / -16.0f));
    //stator->set_center(stator->get_bounding_box_center()); // set center of meshes to their bounding box center
    //rotor->set_center(rotor->get_bounding_box_center());
    //plank->set_center(plank->get_bounding_box_center());
    //voxelize_mesh(stator, TYPE_S, center);
    //voxelize_mesh(rotor, TYPE_S, center);
    //voxelize_mesh(plank, TYPE_S, center);

    //Mesh* plank = nullptr;
    //const glm::vec3 center = glm::vec3(Nx / 2.0f - 0.5f, Ny / 2.0f - 0.5f, Nz / 2.0f - 0.5f);
    // //plank = read_obj("models/viking_room.obj", "textures/viking_room.png", 5.0f);
    //plank = read_abc("models/plank_480_60_2.abc", "textures/Textura_tabla_3.jpg", 1.0f);
    //plank->model_name = "plank";
    //plank->scale(15.0f);
    //plank->translate(center - plank->get_bounding_box_center() - glm::vec3(0.0f, 0.0f, 0.0f));
    //plank->set_center(plank->get_bounding_box_center());
    //auto particles = plank->generate_points(1.0f);

    //float total_max_move = 0.0f;
    //for (uint i = 1u; i < plank->frame_number; i++) {
    //    float max_move = 0.0f;
    //    for (auto particle : particles) {
    //        glm::vec3 p0 = plank->tm[i - 1] * glm::vec4(particle, 1.0f);
    //        glm::vec3 p1 = plank->tm[i] * glm::vec4(particle, 1.0f);
    //        max_move = glm::max(max_move, glm::distance(p0, p1));
    //    }
    //    //std::cout << "index: " << i << " per frame max_move: " << max_move << std::endl;
    //    total_max_move = glm::max(total_max_move, max_move);
    //}
    ////std::cout << "total_max_move: " << total_max_move << std::endl;
    //if (total_max_move > 0.5f) {
    //    std::cout << "WARNING: plank mesh has a maximum movement of " << total_max_move << " which is greater than 0.5!" << std::endl;
    //}

    //for (auto particle : particles) {
    //    debugParticles.push_back(Vertex({particle, glm::vec4(1.0f)}));
    //}
    ////voxelize_mesh(plank, TYPE_S, center);
    //render_mesh(plank);
    //debugParticle(plank);

    Mesh* pool = nullptr;
    const glm::vec3 center = glm::vec3(Nx / 2.0f - 0.5f, Ny / 2.0f - 0.5f, Nz / 2.0f - 0.5f);
    glm::mat3 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    pool = read_glb("models/pool.glb", "textures/pool_bottom.jpg", 1.0f, rotation);
    pool->model_name = "pool";
    pool->scale(40.0f);
    pool->translate(center - pool->get_bounding_box_center() - glm::vec3(0.0f, 0.0f, Nz / 2.0f - 10.0f));
    pool->set_center(pool->get_bounding_box_center());
    render_mesh(pool);

    Mesh* pool_ground = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    pool_ground = read_glb("models/pool_ground.glb", "textures/pool_ground.jpg", 1.0f, rotation);
    pool_ground->model_name = "pool_ground";
    pool_ground->scale(40.0f);
    pool_ground->translate(center - pool_ground->get_bounding_box_center() - glm::vec3(0.0f, 0.0f, Nz / 2.0f - 10.0f));
    pool_ground->set_center(pool_ground->get_bounding_box_center());
    render_mesh(pool_ground);

    Mesh* ladder = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    rotation = glm::rotate(glm::mat4(rotation), glm::radians(90.0f), glm::vec3(0, 1, 0));
    ladder = read_glb("models/ladder.glb", "textures/ladder.jpg", 1.0f, rotation);
    ladder->model_name = "ladder";
    ladder->scale(20.0f);
    ladder->translate(center - ladder->get_bounding_box_center() - glm::vec3(0.0f, Ny / -2.0f, Nz / 2.0f - 20.0f));
    ladder->set_center(ladder->get_bounding_box_center());
    render_mesh(ladder);

    Mesh* ladder2 = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    rotation = glm::rotate(glm::mat4(rotation), glm::radians(-90.0f), glm::vec3(0, 1, 0));
    ladder2 = read_glb("models/ladder.glb", "textures/ladder.jpg", 1.0f, rotation);
    ladder2->model_name = "ladder2";
    ladder2->scale(20.0f);
    ladder2->translate(center - ladder2->get_bounding_box_center() - glm::vec3(0.0f, Ny / 2.0f, Nz / 2.0f - 20.0f));
    ladder2->set_center(ladder2->get_bounding_box_center());
    render_mesh(ladder2);

    Mesh* chair = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    rotation = glm::rotate(glm::mat4(rotation), glm::radians(-30.0f), glm::vec3(0, 1, 0));
    chair = read_obj("models/chair.obj", "textures/chair.png", 1.0f, rotation);
    chair->model_name = "chair";
    chair->scale(1.0f);
    chair->translate(center - chair->get_bounding_box_center() - glm::vec3(Nx / -4.0f + Nx * 3 / -16.0f, Ny * 7 / -8.0f, Nz / 2.0f - 40.0f));
    chair->set_center(chair->get_bounding_box_center());
    render_mesh(chair);

    Mesh* chair2 = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    rotation = glm::rotate(glm::mat4(rotation), glm::radians(30.0f), glm::vec3(0, 1, 0));
    chair2 = read_obj("models/chair.obj", "textures/chair.png", 1.0f, rotation);
    chair2->model_name = "chair2";
    chair2->scale(1.0f);
    chair2->translate(center - chair2->get_bounding_box_center() - glm::vec3(Nx / -4.0f + Nx * 3 / 16.0f, Ny * 7 / -8.0f, Nz / 2.0f - 40.0f));
    chair2->set_center(chair2->get_bounding_box_center());
    render_mesh(chair2);

    Mesh* umbrella = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    umbrella = read_obj("models/umbrella.obj", "textures/umbrella.png", 1.0f, rotation);
    umbrella->model_name = "umbrella";
    umbrella->scale(0.5f);
    umbrella->translate(center - umbrella->get_bounding_box_center() - glm::vec3(Nx / -4.0f, Ny * 7 / -8.0f, Nz / 2.0f - 80.0f));
    umbrella->set_center(umbrella->get_bounding_box_center());
    render_mesh(umbrella);

    Mesh* duck_float = nullptr;
    rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    rotation = glm::rotate(glm::mat4(rotation), glm::radians(-90.0f), glm::vec3(0, 1, 0));
    duck_float = read_glb("models/duck_float.glb", "textures/duck_float.jpg", 1.0f, rotation);
    duck_float->model_name = "duck_float";
    duck_float->scale(40.0f);
    duck_float->translate(center - duck_float->get_bounding_box_center() - glm::vec3(0.0f, 0.0f, Nz / 2.0f - 10.0f));
    duck_float->set_center(duck_float->get_bounding_box_center());
    std::vector<glm::vec3> particles = duck_float->generate_points(1.0f);
    for (auto particle : particles) {
        debugParticles.push_back(Vertex({ particle, glm::vec4(1.0f) }));
    }
    voxelize_mesh(duck_float, particles, TYPE_S);
    render_mesh(duck_float);
}

void LBM::createGraphicsDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphicsDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    graphicsDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &allocInfo, graphicsDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        VkDescriptorBufferInfo renderingUBOBufferInfo{};
        renderingUBOBufferInfo.buffer = renderingUBOBuffers[i];
        renderingUBOBufferInfo.offset = 0;
        renderingUBOBufferInfo.range = sizeof(RenderingUBO);
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = graphicsDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &renderingUBOBufferInfo;

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = skyboxImageView;
        imageInfo.sampler = skyboxSampler;
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = graphicsDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
}

void LBM::createComputeDescriptorSets() {
    // universal
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    }
    
    // raytrace
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), raytraceDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        raytraceDescriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, raytraceDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    }

    // upscale
    if (upscaleTimes > 0) {
        upscaleDescriptorSets.resize(swapChainImages.size());
        for (uint i = 0; i < swapChainImages.size(); ++i) {
            std::vector<VkDescriptorSetLayout> layouts(upscaleTimes, upscaleDescriptorSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(upscaleTimes);
            allocInfo.pSetLayouts = layouts.data();

            upscaleDescriptorSets[i].resize(upscaleTimes);
            if (vkAllocateDescriptorSets(device, &allocInfo, upscaleDescriptorSets[i].data()) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate descriptor sets!");
            }
        }
    }

    // filtered
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, filteredDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        filteredDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, filteredDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    }

    // postprocess
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), postprocessDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        postprocessDescriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, postprocessDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    }

    updateDescriptorSets();
}

void LBM::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void LBM::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void LBM::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

VkImageView LBM::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

uint32_t LBM::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void LBM::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void LBM::createComputeCommandBuffers() {
    computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)computeCommandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate compute command buffers!");
    }
}

void LBM::recordCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBuffer ImGuiCommandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

#ifndef RAYTRACE
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    std::array<VkClearValue, 2> clearColors{};
    clearColors[0].color = { {0.5f, 0.5f, 0.5f, 1.0f} };
    clearColors[1].depthStencil = { 1.0f, 0 };
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearColors.size());
    renderPassInfo.pClearValues = clearColors.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // get surfaceCount data from surfaceCountBuffers
    //{
    //    VkDeviceSize bufferSize = sizeof(uint32_t);
    //
    //    VkBuffer stagingBuffer;
    //    VkDeviceMemory stagingBufferMemory;
    //    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    //    copyBuffer(surfaceCountBuffers[currentFrame], stagingBuffer, bufferSize);
    //
    //    uint32_t* data = nullptr;
    //    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, (void**)&data);
    //    surface_count = *data;
    //    vkUnmapMemory(device, stagingBufferMemory);
    //
    //    vkDestroyBuffer(device, stagingBuffer, nullptr);
    //    vkFreeMemory(device, stagingBufferMemory, nullptr);
    //}

    // surfaces
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, surfacePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &surfaceBuffers[currentFrame], offsets);

        vkCmdDrawIndirect(commandBuffer, surfaceCountBuffers[currentFrame], 0, 1, sizeof(VkDrawIndirectCommand));
    }

    // get wireframeCount data from wireframeCountBuffers
    //{
    //    VkDeviceSize bufferSize = sizeof(uint32_t);
    //
    //    VkBuffer stagingBuffer;
    //    VkDeviceMemory stagingBufferMemory;
    //    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    //    copyBuffer(wireframeCountBuffers[currentFrame], stagingBuffer, bufferSize);
    //
    //    uint32_t* data = nullptr;
    //    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, (void**)&data);
    //    wireframe_count = *data;
    //    vkUnmapMemory(device, stagingBufferMemory);
    //
    //    vkDestroyBuffer(device, stagingBuffer, nullptr);
    //    vkFreeMemory(device, stagingBufferMemory, nullptr);
    //}

    // wireframes
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &wireframeBuffers[currentFrame], offsets);

        vkCmdBindIndexBuffer(commandBuffer, wireframeIndexBuffers[currentFrame], 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndirect(commandBuffer, wireframeCountBuffers[currentFrame], 0, 1, sizeof(VkDrawIndirectCommand));
    }

    // skybox
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &skyboxBuffers[currentFrame], offsets);

        vkCmdBindIndexBuffer(commandBuffer, skyboxIndexBuffers[currentFrame], 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(skyboxIndices.size()), 1, 0, 0, 0);
    }

    // model
    {
        if (modelVertexBuffers.size() != modelPushConstants.size() || modelVertexBuffers.size() != modelDescriptorSets.size()) {
            throw std::runtime_error("model vertex buffers, push constants, and descriptor sets must have the same size!");
        }
        for (uint i = 0; i < modelVertexBuffers.size(); ++i) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, modelPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, modelPipelineLayout, 0, 1, &modelDescriptorSets[i][currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            vkCmdPushConstants(commandBuffer, modelPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ModelPushConstants), &modelPushConstants[i][currentFrame]);
            vkCmdDraw(commandBuffer, modelPushConstants[i][currentFrame].nTriangles * 3, 1, 0, 0);
        }
    }

#ifdef DEBUGTOOLS
    // debug particles
    {
        if (debugParticles.size() && debugParticleFlag) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugParticlePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugParticlePipelineLayout, 0, 1, &debugParticleDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &debugParticleBuffers[currentFrame], offsets);

            vkCmdDraw(commandBuffer, debugParticles.size(), 1, 0, 0);
        }
    }
#endif

    vkCmdEndRenderPass(commandBuffer);
    // particle
    {
        VkRenderPassBeginInfo renderpass_begininfo{};
        renderpass_begininfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderpass_begininfo.framebuffer = fluidsFramebuffers[imageIndex];
        std::array<VkClearValue, 2> clearvalues{};
        clearvalues[0].color = { {0,0,0,0} };
        clearvalues[1].color = { {1000,0,0,0} };
        renderpass_begininfo.clearValueCount = static_cast<uint32_t>(clearvalues.size());
        renderpass_begininfo.pClearValues = clearvalues.data();
        renderpass_begininfo.renderPass = fluidGraphicRenderPass;
        renderpass_begininfo.renderArea.extent = swapChainExtent;
        renderpass_begininfo.renderArea.offset = { 0,0 };
        vkCmdBeginRenderPass(commandBuffer, &renderpass_begininfo, VK_SUBPASS_CONTENTS_INLINE);

        // get particleCount data from particleCountBuffers
        //{
        //    VkDeviceSize bufferSize = sizeof(uint32_t);
        //
        //    VkBuffer stagingBuffer;
        //    VkDeviceMemory stagingBufferMemory;
        //    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        //    copyBuffer(particleCountBuffers[currentFrame], stagingBuffer, bufferSize);
        //
        //    uint32_t* data = nullptr;
        //    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, (void**)&data);
        //    particle_count = *data;
        //    vkUnmapMemory(device, stagingBufferMemory);
        //
        //    vkDestroyBuffer(device, stagingBuffer, nullptr);
        //    vkFreeMemory(device, stagingBufferMemory, nullptr);
        //}

        // particles
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particlePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &particleBuffers[currentFrame], offsets);

            vkCmdDrawIndirect(commandBuffer, particleCountBuffers[currentFrame], 0, 1, sizeof(VkDrawIndirectCommand));
        }
        vkCmdEndRenderPass(commandBuffer);

        VkImageMemoryBarrier imagebarrier{};
        imagebarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.subresourceRange.baseArrayLayer = 0;
        imagebarrier.subresourceRange.baseMipLevel = 0;
        imagebarrier.subresourceRange.layerCount = 1;
        imagebarrier.subresourceRange.levelCount = 1;
        VkMemoryBarrier memorybarrier{};
        memorybarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memorybarrier.srcAccessMask = 0;
        memorybarrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

        // transform filter depth image layout to be writed
        imagebarrier.image = filteredFluidDepthImage;
        imagebarrier.srcAccessMask = 0;
        imagebarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, filteredPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, filteredPipelineLayout, 0, 1, &filteredDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer, swapChainExtent.width / 8, swapChainExtent.height / 8, 1);
        imagebarrier.image = filteredFluidDepthImage;
        imagebarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, postprocessPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, postprocessPipelineLayout, 0, 1, &postprocessDescriptorSets[imageIndex], 0, nullptr);
        imagebarrier.image = swapChainImages[imageIndex];
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.srcAccessMask = 0;
        imagebarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);

        vkCmdDispatch(commandBuffer, swapChainExtent.width / 4, swapChainExtent.height / 4, 1);

        // transform swapchainimage layout from 'VK_IMAGE_LAYOUT_GENERAL' to 'VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL'
        // for imgui's demands
        imagebarrier.image = swapChainImages[imageIndex];
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);
    }
#endif

#ifdef RAYTRACE
    // raytrace
    {
        VkImageMemoryBarrier imagebarrier{};
        imagebarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.subresourceRange.baseArrayLayer = 0;
        imagebarrier.subresourceRange.baseMipLevel = 0;
        imagebarrier.subresourceRange.layerCount = 1;
        imagebarrier.subresourceRange.levelCount = 1;
        VkMemoryBarrier memorybarrier{};
        memorybarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memorybarrier.srcAccessMask = 0;
        memorybarrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, raytracePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, raytracePipelineLayout, 0, 1, &raytraceDescriptorSets[imageIndex], 0, nullptr);
        imagebarrier.image = swapChainImages[imageIndex];
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.srcAccessMask = 0;
        imagebarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);
        if (upscaleTimes != 0) vkCmdDispatch(commandBuffer, imageSize.back().x / 16 + 1, imageSize.back().y / 16 + 1, 1);
        else vkCmdDispatch(commandBuffer, swapChainExtent.width / 16 + 1, swapChainExtent.height / 16 + 1, 1);
        
        for (int i = upscaleTimes - 1; i >= 0; i--) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, upscalePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, upscalePipelineLayout, 0, 1, &upscaleDescriptorSets[imageIndex][i], 0, nullptr);
            imagebarrier.image = upscaleImages[imageIndex][i];
            imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imagebarrier.srcAccessMask = 0;
            imagebarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            imagebarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imagebarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);
            vkCmdDispatch(commandBuffer, imageSize[i].x / 16 + 1, imageSize[i].y / 16 + 1, 1);
        }

        // transform swapchainimage layout from 'VK_IMAGE_LAYOUT_GENERAL' to 'VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL'
        // for imgui's demands
        imagebarrier.image = swapChainImages[imageIndex];
        imagebarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imagebarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imagebarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        imagebarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imagebarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &imagebarrier);
    }
#endif

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    // imgui
    {
        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 控件
        {
            ImGui::Begin("fluid controller"); // Create a window

            ImGui::SliderFloat("transmittance", &renderUbo.transmittance, 0, 1);

            //simulateUbo.niu = 0.01f;
            ImGui::SliderFloat("niu", &simulateUbo.niu, 0.01f, 1.0f);

            //simulateUbo.niu = 0.01f;
            ImGui::SliderFloat("smoothness", &simulateUbo.smoothness, 0.0f, 1.0f);

            //simulateUbo.fx = 0.0f;
            static float fx = simulateUbo.fx * 1000;
            ImGui::SliderFloat("fx", &fx, -1.0f, 1.0f);
            simulateUbo.fx = fx * 0.001f;

            //simulateUbo.fy = 0.0f;
            static float fy = simulateUbo.fy * 1000;
            ImGui::SliderFloat("fy", &fy, -1.0f, 1.0f);
            simulateUbo.fy = fy * 0.001f;

            //simulateUbo.fz = 0.0f;
            static float fz = simulateUbo.fz * 1000;
            ImGui::SliderFloat("fz", &fz, -1.0f, 1.0f);
            simulateUbo.fz = fz * 0.001f;

            //simulateUbo.sigmas = 0.000002f;
            ImGui::SliderFloat("sigmas", &simulateUbo.sigmas, 0.0f, 0.001f);

            ImGui::Checkbox("run?", &isRun);

            ImGui::End();
        }
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(ImGuiCommandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = imGuiRenderPass;
        renderPassInfo.framebuffer = imGuiFrameBuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearColors{};
        clearColors[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearColors[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearColors.size());
        renderPassInfo.pClearValues = clearColors.data();

        vkCmdBeginRenderPass(ImGuiCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Record dear imgui primitives into command buffer
        ImGui_ImplVulkan_RenderDrawData(draw_data, ImGuiCommandBuffer);

        vkCmdEndRenderPass(ImGuiCommandBuffer);
        if (vkEndCommandBuffer(ImGuiCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}

void LBM::recordComputeCommandBuffer(VkCommandBuffer commandBuffer) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkMemoryBarrier memorybarrier{};
    memorybarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memorybarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    memorybarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording compute command buffer!");
    }

    if (!isInit) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, initPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
        vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        isInit = true;
    }

    if (isRun) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, surface0Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collideAndStreamPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
        vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, surface1Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
        vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, surface2Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
        vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, surface3Pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
        //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
        vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

        simulateUbo.t++;
    }

    // clear particleCount, wireframeCount and surfaceCount buffer
    VkDrawIndirectCommand initialCmd = {};
    initialCmd.vertexCount = 0;
    initialCmd.instanceCount = 1;
    initialCmd.firstVertex = 0;
    initialCmd.firstInstance = 0;

    vkCmdUpdateBuffer(commandBuffer, particleCountBuffers[currentFrame], 0, sizeof(VkDrawIndirectCommand), &initialCmd);
    vkCmdUpdateBuffer(commandBuffer, wireframeCountBuffers[currentFrame], 0, sizeof(VkDrawIndirectCommand), &initialCmd);
    vkCmdUpdateBuffer(commandBuffer, surfaceCountBuffers[currentFrame], 0, sizeof(VkDrawIndirectCommand), &initialCmd);

    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,  // <--- 使用全局内存屏障
        0, nullptr,         // Buffer 屏障设为 0
        0, nullptr          // Image 屏障设为 0
    );

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memorybarrier, 0, nullptr, 0, nullptr);
    //vkCmdDispatch(commandBuffer, sqrt(PARTICLE_COUNT) / 16, sqrt(PARTICLE_COUNT) / 16, 1);
    vkCmdDispatch(commandBuffer, Nxyz / 64 + 1, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record compute command buffer!");
    }
}

void LBM::createGuiRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorReference{};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    VkSubpassDependency subpassDependency{};
    subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpassDependency.dstSubpass = 0;
    subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassCreateInfo{};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = 1;
    renderPassCreateInfo.pAttachments = &colorAttachment;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &subpassDependency;

    if (vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &imGuiRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("could not create ImGui's render pass");
    }
}

void LBM::createGuiFrameBuffers() {
    imGuiFrameBuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::array<VkImageView, 1> attachments = {
            swapChainImageViews[i],
            //depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = imGuiRenderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &imGuiFrameBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void LBM::createGuiDescriptorPool()
{
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &imGuiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("could not create ImGui's descriptor pool");
    }
}

void LBM::createGuiCommandBuffers() {
    // create command pool for imgui
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &imGuiCommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    // create command buffers for imgui
    {
        imGuiCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = imGuiCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)imGuiCommandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, imGuiCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
}

void LBM::initImGui()
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    //设置微软雅黑字体,并指定字体大小
    ImFont* font = io.Fonts->AddFontFromFileTTF
    (
        "C:/Windows/Fonts/msyh.ttc",
        30,
        nullptr,
        //设置加载中文
        io.Fonts->GetGlyphRangesChineseFull()
    );
    //必须判断一下字体有没有加载成功
    IM_ASSERT(font != nullptr);

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsAndComputeFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imGuiDescriptorPool;
    init_info.RenderPass = imGuiRenderPass;
    init_info.Subpass = 0;
    init_info.MinImageCount = 2;
    init_info.ImageCount = swapChainImages.size();
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    //init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info);

    // upload fonts

    // end upload fonts
}

void LBM::cleanupImGui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, imGuiDescriptorPool, nullptr);
    for (auto frameBuffer : imGuiFrameBuffers) {
        vkDestroyFramebuffer(device, frameBuffer, nullptr);
    }
    vkDestroyRenderPass(device, imGuiRenderPass, nullptr);
    vkDestroyCommandPool(device, imGuiCommandPool, nullptr);
}

void LBM::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics synchronization objects for a frame!");
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute synchronization objects for a frame!");
        }
    }
}

void LBM::updateUniformBuffer(uint32_t currentImage) {
    // rendering
    {
        float zNear = 1.0f;
        float zFar = 10000.0f;
        float fovy = glm::radians(100.0f);
        float aspectRatio = (float)swapChainExtent.width / (float)swapChainExtent.height;

        glm::vec3 cameraPos = glm::vec3(0.0f);
        cameraPos.x = distance * sin(glm::radians(ry)) * cos(glm::radians(rx));
        cameraPos.y = distance * sin(glm::radians(ry)) * sin(glm::radians(rx));
        cameraPos.z = distance * cos(glm::radians(ry));
        glm::vec3 cameraView = glm::vec3(0.0f);
        glm::vec3 cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);

        renderUbo.Nx = Nx;
        renderUbo.Ny = Ny;
        renderUbo.Nz = Nz;
        renderUbo.Nxyz = Nx * Ny * Nz;
        renderUbo.t = currentTime;

        renderUbo.zNear = zNear;
        renderUbo.zFar = zFar;
        renderUbo.fovy = fovy;
        renderUbo.aspectRatio = aspectRatio;
        renderUbo.particleRadius = 1.0;
        renderUbo.cameraDistance = glm::length(cameraPos - cameraView);
        renderUbo.renderType = 0u;
        renderUbo.fluidType = 0u;
        renderUbo.fluidColor = glm::vec4(0.529f, 0.808f, 0.922f, 0.5f);
        renderUbo.cameraPos = cameraPos;
        renderUbo.cameraView = cameraView;
        renderUbo.cameraUp = cameraUp;

        renderUbo.model = glm::mat4(1.0f);
        renderUbo.view = glm::lookAt(cameraPos, cameraView, cameraUp);
        renderUbo.invView = glm::inverse(renderUbo.view);
        renderUbo.proj = glm::perspective(fovy, aspectRatio, zNear, zFar);
        renderUbo.proj[1][1] *= -1;
        renderUbo.invProj = glm::inverse(renderUbo.proj);
        memcpy(renderingUBOBuffersMapped[currentImage], &renderUbo, sizeof(renderUbo));
    }

    auto nu_from_Re = [](const float Re, const float x, const float u) { return x * u / Re; }; // kinematic shear viscosity nu = x*u/Re = [m^2/s]
    // simulating
    {
        simulateUbo.Nx = Nx;
        simulateUbo.Ny = Ny;
        simulateUbo.Nz = Nz;
        simulateUbo.Nxyz = Nxyz;
        //simulateUbo.particleCount = particle_count;
        simulateUbo.particleRho = 2.0f;
        //simulateUbo.niu = 0.000001f;
        //simulateUbo.niu = nu_from_Re(1000.0f, 128 - 2, 0.05f); // dynamic
        simulateUbo.tau = 3.0f * simulateUbo.niu + 0.5f;
        simulateUbo.inv_tau = 1.0f / simulateUbo.tau;
        //simulateUbo.isoVal = Nx * 0.4f;
        simulateUbo.isoVal = 0.502f;
        //simulateUbo.fx = 0.0f; // dynamic
        //simulateUbo.fy = 0.0f; // dynamic
        //simulateUbo.fz = -0.0002f; //dynamic
        //simulateUbo.sigmas = 0.000001f * 6.0f; // dynamic
        simulateUbo.rx = rx;
        simulateUbo.ry = ry;
        simulateUbo.distance = distance;
        simulateUbo.t = currentTime;
        memcpy(uniformBuffersMapped[currentImage], &simulateUbo, sizeof(simulateUbo));
    }

    // model
    {
        for (uint i = 0; i < modelUniformBuffersMapped.size(); ++i) {
            memcpy(modelUniformBuffersMapped[i][currentImage], &renderUbo, sizeof(renderUbo));
        }
    }

    if (isRun) currentTime += 1;
}

void LBM::updateDescriptorSets() {
    // universal
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            std::array<VkWriteDescriptorSet, 19> descriptorWrites{};

            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = sizeof(SimulateUBO);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            storageBufferInfoLastFrame.buffer = particleBuffers[i];
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range = sizeof(Particle) * Nxyz;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

            VkDescriptorBufferInfo velocityBufferInfo{};
            velocityBufferInfo.buffer = velocityBuffers[i];
            velocityBufferInfo.offset = 0;
            velocityBufferInfo.range = Nxyz * 3 * sizeof(uint16_t);
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = computeDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &velocityBufferInfo;

            VkDescriptorBufferInfo rhoBufferInfo{};
            rhoBufferInfo.buffer = rhoBuffers[i];
            rhoBufferInfo.offset = 0;
            rhoBufferInfo.range = Nxyz * sizeof(uint16_t);
            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = computeDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &rhoBufferInfo;

            VkDescriptorBufferInfo flagBufferInfo{};
            flagBufferInfo.buffer = flagBuffers[i];
            flagBufferInfo.offset = 0;
            flagBufferInfo.range = Nxyz * sizeof(uint32_t);
            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = computeDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &flagBufferInfo;

            VkDescriptorBufferInfo DDF1BufferInfo{};
            if (currentTime % 2 == 0) DDF1BufferInfo.buffer = DDF1Buffers[i];
            else DDF1BufferInfo.buffer = DDF2Buffers[i];
            //DDF1BufferInfo.buffer = DDF1Buffers[i];
            DDF1BufferInfo.offset = 0;
            DDF1BufferInfo.range = Nxyz * 5 * 4 * sizeof(uint16_t);
            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = computeDescriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pBufferInfo = &DDF1BufferInfo;

            VkDescriptorBufferInfo cellForceBufferInfo{};
            cellForceBufferInfo.buffer = cellForceBuffers[i];
            cellForceBufferInfo.offset = 0;
            cellForceBufferInfo.range = Nxyz * sizeof(float) * 3;
            descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[6].dstSet = computeDescriptorSets[i];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pBufferInfo = &cellForceBufferInfo;

            VkDescriptorBufferInfo phiBufferInfo{};
            phiBufferInfo.buffer = PhiBuffers[i];
            phiBufferInfo.offset = 0;
            phiBufferInfo.range = Nxyz * sizeof(float);
            descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[7].dstSet = computeDescriptorSets[i];
            descriptorWrites[7].dstBinding = 7;
            descriptorWrites[7].dstArrayElement = 0;
            descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[7].descriptorCount = 1;
            descriptorWrites[7].pBufferInfo = &phiBufferInfo;

            VkDescriptorBufferInfo wireframeVertexBufferInfo{};
            wireframeVertexBufferInfo.buffer = wireframeBuffers[i];
            wireframeVertexBufferInfo.offset = 0;
            wireframeVertexBufferInfo.range = Nxyz * sizeof(Vertex) * 2;
            descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[8].dstSet = computeDescriptorSets[i];
            descriptorWrites[8].dstBinding = 8;
            descriptorWrites[8].dstArrayElement = 0;
            descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[8].descriptorCount = 1;
            descriptorWrites[8].pBufferInfo = &wireframeVertexBufferInfo;

            VkDescriptorBufferInfo wireframeIndexBufferInfo{};
            wireframeIndexBufferInfo.buffer = wireframeIndexBuffers[i];
            wireframeIndexBufferInfo.offset = 0;
            wireframeIndexBufferInfo.range = Nxyz * sizeof(uint32_t) * 2;
            descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[9].dstSet = computeDescriptorSets[i];
            descriptorWrites[9].dstBinding = 9;
            descriptorWrites[9].dstArrayElement = 0;
            descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[9].descriptorCount = 1;
            descriptorWrites[9].pBufferInfo = &wireframeIndexBufferInfo;

            VkDescriptorBufferInfo surfaceVertexBufferInfo{};
            surfaceVertexBufferInfo.buffer = surfaceBuffers[i];
            surfaceVertexBufferInfo.offset = 0;
            surfaceVertexBufferInfo.range = Nxyz * sizeof(Vertex) * 3 * 5;
            descriptorWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[10].dstSet = computeDescriptorSets[i];
            descriptorWrites[10].dstBinding = 10;
            descriptorWrites[10].dstArrayElement = 0;
            descriptorWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[10].descriptorCount = 1;
            descriptorWrites[10].pBufferInfo = &surfaceVertexBufferInfo;

            VkDescriptorBufferInfo MassBufferInfo{};
            MassBufferInfo.buffer = MassBuffers[i];
            MassBufferInfo.offset = 0;
            MassBufferInfo.range = Nxyz * sizeof(float);
            descriptorWrites[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[11].dstSet = computeDescriptorSets[i];
            descriptorWrites[11].dstBinding = 11;
            descriptorWrites[11].dstArrayElement = 0;
            descriptorWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[11].descriptorCount = 1;
            descriptorWrites[11].pBufferInfo = &MassBufferInfo;

            VkDescriptorBufferInfo MassExBufferInfo{};
            MassExBufferInfo.buffer = MassExBuffers[i];
            MassExBufferInfo.offset = 0;
            MassExBufferInfo.range = Nxyz * sizeof(float);
            descriptorWrites[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[12].dstSet = computeDescriptorSets[i];
            descriptorWrites[12].dstBinding = 12;
            descriptorWrites[12].dstArrayElement = 0;
            descriptorWrites[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[12].descriptorCount = 1;
            descriptorWrites[12].pBufferInfo = &MassExBufferInfo;

            VkDescriptorBufferInfo particleCountBufferInfo{};
            particleCountBufferInfo.buffer = particleCountBuffers[i];
            particleCountBufferInfo.offset = 0;
            particleCountBufferInfo.range = sizeof(uint32_t);
            descriptorWrites[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[13].dstSet = computeDescriptorSets[i];
            descriptorWrites[13].dstBinding = 13;
            descriptorWrites[13].dstArrayElement = 0;
            descriptorWrites[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[13].descriptorCount = 1;
            descriptorWrites[13].pBufferInfo = &particleCountBufferInfo;

            VkDescriptorBufferInfo wireframeCountBufferInfo{};
            wireframeCountBufferInfo.buffer = wireframeCountBuffers[i];
            wireframeCountBufferInfo.offset = 0;
            wireframeCountBufferInfo.range = sizeof(uint32_t);
            descriptorWrites[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[14].dstSet = computeDescriptorSets[i];
            descriptorWrites[14].dstBinding = 14;
            descriptorWrites[14].dstArrayElement = 0;
            descriptorWrites[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[14].descriptorCount = 1;
            descriptorWrites[14].pBufferInfo = &wireframeCountBufferInfo;

            VkDescriptorBufferInfo surfaceCountBufferInfo{};
            surfaceCountBufferInfo.buffer = surfaceCountBuffers[i];
            surfaceCountBufferInfo.offset = 0;
            surfaceCountBufferInfo.range = sizeof(uint32_t);
            descriptorWrites[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[15].dstSet = computeDescriptorSets[i];
            descriptorWrites[15].dstBinding = 15;
            descriptorWrites[15].dstArrayElement = 0;
            descriptorWrites[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[15].descriptorCount = 1;
            descriptorWrites[15].pBufferInfo = &surfaceCountBufferInfo;

            VkDescriptorBufferInfo DDF2BufferInfo{};
            if (currentTime % 2 == 0) DDF2BufferInfo.buffer = DDF2Buffers[i];
            else DDF2BufferInfo.buffer = DDF1Buffers[i];
            //DDF2BufferInfo.buffer = DDF2Buffers[i];
            DDF2BufferInfo.offset = 0;
            DDF2BufferInfo.range = Nxyz * 5 * 4 * sizeof(uint16_t);
            descriptorWrites[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[16].dstSet = computeDescriptorSets[i];
            descriptorWrites[16].dstBinding = 16;
            descriptorWrites[16].dstArrayElement = 0;
            descriptorWrites[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[16].descriptorCount = 1;
            descriptorWrites[16].pBufferInfo = &DDF2BufferInfo;

            VkDescriptorBufferInfo stateBufferInfo{};
            stateBufferInfo.buffer = stateBuffers[i];
            stateBufferInfo.offset = 0;
            stateBufferInfo.range = Nxyz * 4 * sizeof(uint16_t);
            descriptorWrites[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[17].dstSet = computeDescriptorSets[i];
            descriptorWrites[17].dstBinding = 17;
            descriptorWrites[17].dstArrayElement = 0;
            descriptorWrites[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[17].descriptorCount = 1;
            descriptorWrites[17].pBufferInfo = &stateBufferInfo;

            VkDescriptorBufferInfo totalMassBufferInfo{};
            totalMassBufferInfo.buffer = totalMassBuffers[i];
            totalMassBufferInfo.offset = 0;
            totalMassBufferInfo.range = sizeof(float) * 2;
            descriptorWrites[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[18].dstSet = computeDescriptorSets[i];
            descriptorWrites[18].dstBinding = 18;
            descriptorWrites[18].dstArrayElement = 0;
            descriptorWrites[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[18].descriptorCount = 1;
            descriptorWrites[18].pBufferInfo = &totalMassBufferInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }

    // raytrace
    {
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

            // uniform buffer
            VkDescriptorBufferInfo renderingUBOBufferInfo{};
            renderingUBOBufferInfo.buffer = renderingUBOBuffers[0];
            renderingUBOBufferInfo.offset = 0;
            renderingUBOBufferInfo.range = sizeof(RenderingUBO);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = raytraceDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &renderingUBOBufferInfo;

            // framebuffer
            VkDescriptorImageInfo dstimageinfo{};
            if (upscaleTimes == 0) dstimageinfo.imageView = swapChainImageViews[i];
            else dstimageinfo.imageView = upscaleImagesView[i].back();
            dstimageinfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            dstimageinfo.sampler = nullptr;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = raytraceDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[1].pImageInfo = &dstimageinfo;

            // skybox
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = skyboxImageView;
            imageInfo.sampler = skyboxSampler;
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = raytraceDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &imageInfo;

            // phis
            VkDescriptorBufferInfo phiBufferInfo{};
            phiBufferInfo.buffer = PhiBuffers[0];
            phiBufferInfo.offset = 0;
            phiBufferInfo.range = Nxyz * sizeof(float);
            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = raytraceDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &phiBufferInfo;

            // flags
            VkDescriptorBufferInfo flagBufferInfo{};
            flagBufferInfo.buffer = flagBuffers[0];
            flagBufferInfo.offset = 0;
            flagBufferInfo.range = Nxyz * sizeof(uint32_t);
            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = raytraceDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &flagBufferInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // upscale
    {
        for (size_t i = 0; i < swapChainImages.size(); ++i) {
            for (size_t j = 0; j < upscaleTimes; ++j) {
                std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

                VkDescriptorImageInfo srcImageinfo{};
                srcImageinfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                srcImageinfo.imageView = upscaleImagesView[i][j];
                srcImageinfo.sampler = upscaleImagesSampler[i];
                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstSet = upscaleDescriptorSets[i][j];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[0].pImageInfo = &srcImageinfo;

                VkDescriptorImageInfo dstImageinfo{};
                dstImageinfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                if (j != 0) {
                    dstImageinfo.imageView = upscaleImagesView[i][j - 1];
                }
                else dstImageinfo.imageView = swapChainImageViews[i];
                dstImageinfo.sampler = filteredFluidDepthImageSampler;
                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstSet = upscaleDescriptorSets[i][j];
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[1].pImageInfo = &dstImageinfo;

                vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }
        }
    }

    // filtered
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            VkDescriptorImageInfo fluidDepthImageinfo{};
            fluidDepthImageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fluidDepthImageinfo.imageView = fluidDepthImageView;
            fluidDepthImageinfo.sampler = fluidDepthImageSampler;
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = filteredDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].pImageInfo = &fluidDepthImageinfo;

            VkDescriptorImageInfo filteredFluidDepthImageinfo{};
            filteredFluidDepthImageinfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            filteredFluidDepthImageinfo.imageView = filteredFluidDepthImageView;
            filteredFluidDepthImageinfo.sampler = filteredFluidDepthImageSampler;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = filteredDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[1].pImageInfo = &filteredFluidDepthImageinfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // postprocess
    {
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

            VkDescriptorBufferInfo renderingbufferinfo{};
            renderingbufferinfo.buffer = renderingUBOBuffers[0];
            renderingbufferinfo.offset = 0;
            renderingbufferinfo.range = sizeof(RenderingUBO);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].pBufferInfo = &renderingbufferinfo;

            VkDescriptorImageInfo depthimageinfo{};
            depthimageinfo.imageView = filteredFluidDepthImageView;
            depthimageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            depthimageinfo.sampler = filteredFluidDepthImageSampler;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].pImageInfo = &depthimageinfo;

            VkDescriptorImageInfo thickimageinfo{};
            thickimageinfo.imageView = thickImageView;
            thickimageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            thickimageinfo.sampler = thickImageSampler;
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].pImageInfo = &thickimageinfo;

            VkDescriptorImageInfo backgroundimageinfo{};
            backgroundimageinfo.imageView = backgroundImageView;
            backgroundimageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            backgroundimageinfo.sampler = backgroundImageSampler;
            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[3].pImageInfo = &backgroundimageinfo;

            VkDescriptorImageInfo dstimageinfo{};
            dstimageinfo.imageView = swapChainImageViews[i];
            dstimageinfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            dstimageinfo.sampler = nullptr;
            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[4].pImageInfo = &dstimageinfo;

            VkDescriptorImageInfo boxdepthimageinfo{};
            boxdepthimageinfo.imageView = depthImageView;
            boxdepthimageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            boxdepthimageinfo.sampler = depthImageSampler;
            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = postprocessDescriptorSets[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[5].pImageInfo = &boxdepthimageinfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
}

void LBM::drawFrame() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Compute submission
    vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

    updateUniformBuffer(currentFrame);
    // update descriptor sets
    for (uint i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        VkDescriptorBufferInfo DDF1BufferInfo{};
        if (currentTime % 2 == 0) DDF1BufferInfo.buffer = DDF1Buffers[i];
        else DDF1BufferInfo.buffer = DDF2Buffers[i];
        DDF1BufferInfo.offset = 0;
        DDF1BufferInfo.range = Nxyz * 5 * 4 * sizeof(uint16_t);

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 5;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &DDF1BufferInfo;

        VkDescriptorBufferInfo DDF2BufferInfo{};
        if (currentTime % 2 == 0) DDF2BufferInfo.buffer = DDF2Buffers[i];
        else DDF2BufferInfo.buffer = DDF1Buffers[i];
        DDF2BufferInfo.offset = 0;
        DDF2BufferInfo.range = Nxyz * 5 * 4 * sizeof(uint16_t);

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 16;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &DDF2BufferInfo;

        vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    vkResetCommandBuffer(computeCommandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit compute command buffer!");
    };

    // Graphics submission
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(commandBuffers[currentFrame], imGuiCommandBuffers[currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = { computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::array<VkCommandBuffer, 2> uploadingCommandBuffers = { commandBuffers[currentFrame], imGuiCommandBuffers[currentFrame] };
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = uploadingCommandBuffers.size();
    submitInfo.pCommandBuffers = uploadingCommandBuffers.data();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void LBM::voxelize_mesh(const Mesh* mesh, const std::vector<glm::vec3>& particles, const int flag) {
    VoxelizationPushConstants pcs{};

    VkDescriptorSetLayout voxelizationDescriptorSetLayout;
    VkDescriptorPool voxelizationDescriptorPool;
    std::vector<VkDescriptorSet> VoxelizationDescriptorSets;
    std::vector<VkBuffer> PointsBuffers;
    std::vector<VkDeviceMemory> PointsBuffersMemory;

    VkPipelineLayout voxelPipelineLayout;
    VkPipeline voxelPipeline;

    // initialize voxelization parameters
    {
        const float x0 = mesh->pmin.x - 2.0f, y0 = mesh->pmin.y - 2.0f, z0 = mesh->pmin.z - 2.0f, x1 = mesh->pmax.x + 2.0f, y1 = mesh->pmax.y + 2.0f, z1 = mesh->pmax.z + 2.0f; // use bounding box of mesh to speed up voxelization; add tolerance of 2 cells for re-voxelization of moving objects
        pcs.particle_number = float(particles.size());
        pcs.flag_type = flag;
    }

    // create voxelization descritpor set layout
    {
        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

        // flag
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // particles
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &voxelizationDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create model descriptor set layout!");
        }
    }

    // create point buffers
    {
        VkDeviceSize bufferSize = particles.size() * sizeof(glm::vec4);

        PointsBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        PointsBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;

        std::vector<glm::vec4> bufferData;
        bufferData.reserve(particles.size());
        for (const auto& p : particles) {
            bufferData.emplace_back(p, 1.0f); // 1.0f 是通常的 w 分量，如果是方向则用 0.0f
        }

        // create Points buffer
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, bufferData.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, PointsBuffers[i], PointsBuffersMemory[i]);
            copyBuffer(stagingBuffer, PointsBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // create voxelization descriptor set
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 10;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 10;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 100;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &voxelizationDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, voxelizationDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = voxelizationDescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        VoxelizationDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, VoxelizationDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = sizeof(SimulateUBO);

            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = VoxelizationDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

            VkDescriptorBufferInfo flagBufferInfo{};
            flagBufferInfo.buffer = flagBuffers[i];
            flagBufferInfo.offset = 0;
            flagBufferInfo.range = Nxyz * sizeof(int);

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = VoxelizationDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &flagBufferInfo;

            VkDescriptorBufferInfo pointsBufferInfo{};
            pointsBufferInfo.buffer = PointsBuffers[i];
            pointsBufferInfo.offset = 0;
            pointsBufferInfo.range = particles.size() * sizeof(glm::vec4);

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = VoxelizationDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &pointsBufferInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }

    // create voxelization compute pipeline
    {
        VkPushConstantRange voxelizationPushConstantRange{};
        voxelizationPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        voxelizationPushConstantRange.offset = 0;
        voxelizationPushConstantRange.size = sizeof(VoxelizationPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &voxelizationDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &voxelizationPushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &voxelPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        auto computeShaderCode = readFile("shaders/voxelize_model_comp.spv");
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = voxelPipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &voxelPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }

    updateUniformBuffer(0);
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, voxelPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, voxelPipelineLayout, 0, 1, &VoxelizationDescriptorSets[currentFrame], 0, nullptr);
    vkCmdPushConstants(commandBuffer, voxelPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VoxelizationPushConstants), &pcs);
    vkCmdDispatch(commandBuffer, particles.size() / 64 + 1, 1, 1);
    endSingleTimeCommands(commandBuffer);

    // clean up
    {
        vkDestroyDescriptorSetLayout(device, voxelizationDescriptorSetLayout, nullptr);
        vkDestroyPipelineLayout(device, voxelPipelineLayout, nullptr);
        vkDestroyPipeline(device, voxelPipeline, nullptr);
        vkDestroyDescriptorPool(device, voxelizationDescriptorPool, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, PointsBuffers[i], nullptr);
            vkFreeMemory(device, PointsBuffersMemory[i], nullptr);
        }
    }
    std::cout << "Model " << mesh->model_name << " voxelized successfully" << std::endl;
}

void LBM::render_mesh(const Mesh* mesh) {
    // create model vertex buffer
    {
        std::vector<VkBuffer> PointsBuffers;
        std::vector<VkDeviceMemory> PointsBuffersMemory;

        VkDeviceSize pointsBufferSize = mesh->triangle_number * sizeof(float) * 3;
        VkDeviceSize texCoorBufferSize = mesh->triangle_number * sizeof(float) * 2;
        VkDeviceSize bufferSize = mesh->triangle_number * mesh->frame_number * sizeof(float) * 9 + mesh->triangle_number * sizeof(float) * 6;

        PointsBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        PointsBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;

        // create Points buffer
        int index = 0;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        for (int i = 0; i < mesh->frame_number; ++i) {
            memcpy(static_cast<float*>(data) + 9 * index, mesh->p0 + index, (size_t)pointsBufferSize);
            memcpy(static_cast<float*>(data) + 9 * index + 3 * mesh->triangle_number, mesh->p1 + index, (size_t)pointsBufferSize);
            memcpy(static_cast<float*>(data) + 9 * index + 6 * mesh->triangle_number, mesh->p2 + index, (size_t)pointsBufferSize);
            index += mesh->triangle_number;
        }
        //index += 9 * mesh->triangle_number;
        //memcpy(static_cast<float*>(data) + index, mesh->p0 + index, (size_t)pointsBufferSize);
        //memcpy(static_cast<float*>(data) + index + 3 * mesh->triangle_number, mesh->p1 + index, (size_t)pointsBufferSize);
        //memcpy(static_cast<float*>(data) + index + 6 * mesh->triangle_number, mesh->p2 + index, (size_t)pointsBufferSize);
        memcpy(static_cast<float*>(data) + 9 * mesh->frame_number * mesh->triangle_number, mesh->tc0, (size_t)texCoorBufferSize);
        memcpy(static_cast<float*>(data) + 9 * mesh->frame_number * mesh->triangle_number + 2 * mesh->triangle_number, mesh->tc1, (size_t)texCoorBufferSize);
        memcpy(static_cast<float*>(data) + 9 * mesh->frame_number * mesh->triangle_number + 4 * mesh->triangle_number, mesh->tc2, (size_t)texCoorBufferSize);
        
        vkUnmapMemory(device, stagingBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, PointsBuffers[i], PointsBuffersMemory[i]);
            copyBuffer(stagingBuffer, PointsBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        modelVertexBuffers.push_back(PointsBuffers);
        modelVertexBuffersMemory.push_back(PointsBuffersMemory);
    }

    // create transformation matrix buffer
    {
        std::vector<VkBuffer> TMBuffers;
        std::vector<VkDeviceMemory> TMBuffersMemory;

        VkDeviceSize bufferSize = mesh->frame_number * sizeof(glm::mat4);

        TMBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        TMBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        // create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;

        // create TM buffer
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, mesh->tm, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, TMBuffers[i], TMBuffersMemory[i]);
            copyBuffer(stagingBuffer, TMBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        modelTMBuffers.push_back(TMBuffers);
        modelTMBuffersMemory.push_back(TMBuffersMemory);
    }

    // create model uniform buffer
    {
        std::vector<VkBuffer> uniformBuffers;
        std::vector<VkDeviceMemory> uniformBuffersMemory;
        std::vector<void*> uniformBuffersMapped;

        VkDeviceSize bufferSize = sizeof(RenderingUBO);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }

        modelUniformBuffers.push_back(uniformBuffers);
        modelUniformBuffersMemory.push_back(uniformBuffersMemory);
        modelUniformBuffersMapped.push_back(uniformBuffersMapped);
    }

    // create model push constant
    {
        std::vector<ModelPushConstants> pcs{};
        pcs.resize(MAX_FRAMES_IN_FLIGHT);

        for (uint i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            pcs[i].nTriangles = mesh->triangle_number;
            pcs[i].nFrames = mesh->frame_number;
        }

        modelPushConstants.push_back(pcs);
    }

    // create texture sampler
    {
        VkImage textureImage;
        VkDeviceMemory textureImageMemory;
        VkImageView textureImageView;
        VkSampler textureSampler;

        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(mesh->texture_path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        if (!pixels || texWidth <= 0 || texHeight <= 0) {
            throw std::runtime_error("failed to load texture image!");
        }
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);
        stbi_image_free(pixels);
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange);

        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        modelImages.push_back(textureImage);
        modelImageMemorys.push_back(textureImageMemory);
        modelImageViews.push_back(textureImageView);
        modelSamplers.push_back(textureSampler);
    }

    // create model descriptorsets
    {
        std::vector<VkDescriptorSet> modelDescriptorSet;
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, modelDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        modelDescriptorSet.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, modelDescriptorSet.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = modelUniformBuffers.back()[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range = sizeof(RenderingUBO);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = modelDescriptorSet[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

            VkDescriptorBufferInfo vertexBufferInfo{};
            vertexBufferInfo.buffer = modelVertexBuffers.back()[i];
            vertexBufferInfo.offset = 0;
            vertexBufferInfo.range = mesh->triangle_number * mesh->frame_number * sizeof(float) * 9 + mesh->triangle_number * sizeof(float) * 6;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = modelDescriptorSet[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &vertexBufferInfo;

            VkDescriptorBufferInfo tmBufferInfo{};
            tmBufferInfo.buffer = modelTMBuffers.back()[i];
            tmBufferInfo.offset = 0;
            tmBufferInfo.range = mesh->frame_number * sizeof(glm::mat4);
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = modelDescriptorSet[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &tmBufferInfo;

            VkDescriptorImageInfo textureInfo{};
            textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            textureInfo.imageView = modelImageViews.back();
            textureInfo.sampler = modelSamplers.back();
            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = modelDescriptorSet[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pImageInfo = &textureInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }

        modelDescriptorSets.push_back(modelDescriptorSet);
    }
}

VkCommandBuffer LBM::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void LBM::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void LBM::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        width,
        height,
        1
    };

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

void LBM::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    //barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    //barrier.subresourceRange.baseMipLevel = 0;
    //barrier.subresourceRange.levelCount = 1;
    //barrier.subresourceRange.baseArrayLayer = 0;
    //barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange = subresourceRange;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}

bool LBM::hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat LBM::findDepthFormat() {
    return findSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

VkFormat LBM::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VkShaderModule LBM::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

VkSurfaceFormatKHR LBM::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (auto& format : availableFormats) {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format.format, &formatProperties);
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            continue;
        }
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR LBM::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D LBM::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

SwapChainSupportDetails LBM::querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

bool LBM::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool LBM::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices LBM::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.graphicsAndComputeFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

std::vector<const char*> LBM::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    //extensions.push_back("VK_KHR_shader_buffer_float32_atomic_add");
    //extensions.push_back("VK_KHR_get_physical_device_properties2");
    //extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    return extensions;
}

bool LBM::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

std::vector<char> LBM::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL LBM::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

#ifdef DEBUGTOOLS
void LBM::debugParticle(Mesh* mesh) {
    debugParticleFlag = true;
    // create debug particle descriptor set layout
    {
        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &debugParticleDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // create debug particle pipeline layout
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &debugParticleDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &debugParticlePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // create debug particle pipeline
    {
        auto debugParticleVertShaderCode = readFile("shaders/debug_particle_vert.spv");
        auto debugParticleFragShaderCode = readFile("shaders/debug_particle_frag.spv");
        VkShaderModule debugParticleVertShaderModule = createShaderModule(debugParticleVertShaderCode);
        VkShaderModule debugParticleFragShaderModule = createShaderModule(debugParticleFragShaderCode);
        VkPipelineShaderStageCreateInfo debugParticleVertShaderStageInfo{};
        debugParticleVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        debugParticleVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        debugParticleVertShaderStageInfo.module = debugParticleVertShaderModule;
        debugParticleVertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo debugParticleFragShaderStageInfo{};
        debugParticleFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        debugParticleFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        debugParticleFragShaderStageInfo.module = debugParticleFragShaderModule;
        debugParticleFragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo debugParticleShaderStages[] = { debugParticleVertShaderStageInfo, debugParticleFragShaderStageInfo };
        
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        bindingDescription = Vertex::getBindingDescription();
        attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = debugParticleShaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = debugParticlePipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &debugParticlePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, debugParticleFragShaderModule, nullptr);
        vkDestroyShaderModule(device, debugParticleVertShaderModule, nullptr);
    }

    // create debug particle buffer
    {
        VkDeviceSize bufferSize = sizeof(Vertex) * debugParticles.size();

        debugParticleBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        debugParticleBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        //create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, debugParticles.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // create debug particle buffer
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, debugParticleBuffers[i], debugParticleBuffersMemory[i]);
            copyBuffer(stagingBuffer, debugParticleBuffers[i], bufferSize);
        }

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // create debug particle descriptor set
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, debugParticleDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        debugParticleDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, debugParticleDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            VkDescriptorBufferInfo renderingUBOBufferInfo{};
            renderingUBOBufferInfo.buffer = renderingUBOBuffers[i];
            renderingUBOBufferInfo.offset = 0;
            renderingUBOBufferInfo.range = sizeof(RenderingUBO);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = debugParticleDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &renderingUBOBufferInfo;

            VkDescriptorBufferInfo tmBufferInfo{};
            tmBufferInfo.buffer = modelTMBuffers.back()[i];
            tmBufferInfo.offset = 0;
            tmBufferInfo.range = mesh->frame_number * sizeof(glm::mat4);
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = debugParticleDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &tmBufferInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }
}

void LBM::cleanupDEBUGTOOLS() {
    // clean up debug particle
    if (debugParticleFlag) {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, debugParticleBuffers[i], nullptr);
            vkFreeMemory(device, debugParticleBuffersMemory[i], nullptr);
        }
        vkDestroyPipelineLayout(device, debugParticlePipelineLayout, nullptr);
        vkDestroyPipeline(device, debugParticlePipeline, nullptr);
        vkDestroyDescriptorSetLayout(device, debugParticleDescriptorSetLayout, nullptr);
    }
}
#endif