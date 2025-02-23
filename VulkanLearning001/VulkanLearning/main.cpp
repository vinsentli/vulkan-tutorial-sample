#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <set>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::string MODEL_PATH = "viking_room.obj";
const std::string TEXTURE_PATH = "viking_room.png";

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                                    void *pUserData){
    std::cerr << "validation layer : " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

static void framebufferResizeCallback(GLFWwindow * window, int width, int height);

const int PARTICLE_COUNT = 8192;

struct Particle{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;
    
    static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription bindingDescription{};
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Particle);
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            return bindingDescription;
        }
    
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions(){
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
        attributeDescriptions.resize(2);
        
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);
        
        return attributeDescriptions;
    }
};

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    
    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bingdingDescription = {};
        
        bingdingDescription.binding = 0;
        bingdingDescription.stride = sizeof(Vertex);
        bingdingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        
        return bingdingDescription;
    }
    
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions(){
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
        attributeDescriptions.resize(3);
        
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        
        return attributeDescriptions;
    }
    
    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct UniformBufferObject{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    
//    float deltaTime = 1.0f;
};

struct QueueFamilyIndices{
    int graphicsFamily = -1;
    int presentFamily = -1;
    
    bool isComplete(){
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
};

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
public:
    bool framebufferResized = false;
    
private:
    GLFWwindow* window = nullptr;
    VkInstance instance;
    VkDebugUtilsMessengerEXT callback;
	VkSurfaceKHR surface;
    //物理设备
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    //逻辑设备
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;
    
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;
    VkRenderPass renderPass;
    
    VkDescriptorSetLayout descriporSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkDescriptorPool computeDescriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    
    VkPipelineLayout pipelineLayout;
    VkPipelineLayout particlePipelineLayout;
    VkPipeline graphicsPipeline;
    VkPipeline particleGraphicsPipeline;
    
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    std::vector<void*> uniformBuffersMapped;
    
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    
    std::vector<VkFence> computeInFlightFences;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    
    size_t currentFrame = 0;
    float lastFrameTime = 0.0f;
    double lastTime = 0.0f;
    
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    uint32_t mipLevels;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
    
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    
    std::vector<VkBuffer> shaderStorageBuffers;
    std::vector<VkDeviceMemory> shaderStorageBuffersMemory;
    
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    
private:
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        
        lastTime = glfwGetTime();
    }
    
    void initVulkan() {
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createUniformBuffer();
        createDescriptorSetLayout();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createColorResouces();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createShaderStorageBuffers();
        createVertexBuffer();
        createIndexBuffer();
        createCommandBuffers();
        createComputeCommandBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createComputeDescriptorPool();
        createComputeDescriptorSets();
        createSyncObjects();
    }
	
	void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            
            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;
        }
        
        vkDeviceWaitIdle(device);
    }
	
	void cleanup() {
        if (enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
        }
        
        clearupSwapChain();
    
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT ; ++i){
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyFence(device, computeInFlightFences[i], nullptr);
            vkDestroyBuffer(device, shaderStorageBuffers[i], nullptr);
            vkFreeMemory(device, shaderStorageBuffersMemory[i], nullptr);
        }
        
        for (size_t i = 0; i < swapChainImages.size() ; ++i){
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        
        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);
        vkDestroyDescriptorSetLayout(device, descriporSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorPool(device, computeDescriptorPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    
    void clearupSwapChain(){
        for (auto imageView : swapChainImageViews){
            vkDestroyImageView(device, imageView, nullptr);
        }
        
        for (auto framebuffer : swapChainFramebuffers){
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);
        
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
    
    void recreateSwapChain(){
        int width = 0, height = 0;
        while (width == 0 || height == 0){
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        
        vkDeviceWaitIdle(device);
        
        clearupSwapChain();
        
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createColorResouces();
        createDepthResources();
        createFramebuffers();
        createCommandBuffers();
        createComputeCommandBuffers();
    }

    void createInstance(){
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        
        std::cout <<"available extensions : " << std::endl;
        for (const auto & extension : extensions){
            std::cout << "\t" << extension.extensionName << std::endl;
        }
        
        if (enableValidationLayers && !checkValidationLayerSupport()){
            throw  std::runtime_error("validation layers requested, but not available!");
        }
        
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        auto requiredExtensions = getRequiredExtensions();
        
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        createInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        
        if (enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
        
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS){
            throw std::runtime_error("failed to create instance!");
        }
    }
    
    void pickPhysicalDevice(){
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0){
            throw  std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        for (const auto& device : devices){
            if (isDeviceSuitable(device)){
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }
        
        if (physicalDevice == VK_NULL_HANDLE){
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }
    
    bool isDeviceSuitable(VkPhysicalDevice device){
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        
        //macOS不支持
//        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
//        deviceFeatures.geometryShader;
        
        QueueFamilyIndices indices = findQueueFamilies(device);
        
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        
        bool swapChainAdequate = false;
        if (extensionsSupported){
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        
        return indices.isComplete() && extensionsSupported && swapChainAdequate && deviceFeatures.samplerAnisotropy;
    }
    
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device){
        QueueFamilyIndices indices;
        
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        int i = 0;
        for (const auto& queueFamily : queueFamilies){
            if ((queueFamily.queueCount > 0) &&
                (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)){
                indices.graphicsFamily = i;
            }
            
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            
            if (queueFamily.queueCount > 0 && presentSupport){
                indices.presentFamily = i;
            }
            
            if (indices.isComplete()){
                break;
            }
            
            i++;
        }
        
        return indices;
    }
    
    std::vector<const char *> getRequiredExtensions(){
        uint32_t glfwExtensionCount = 0;
        const char ** glfwExtensions;
        
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        std::vector<const char *> requiredExtensions;
        for (uint32_t i = 0; i < glfwExtensionCount; i++){
            requiredExtensions.emplace_back(glfwExtensions[i]);
        }
        
        if (enableValidationLayers){
            requiredExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        
#if defined __APPLE__ && defined __arm64__
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        requiredExtensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
        
        return requiredExtensions;
    }
    
    bool checkValidationLayerSupport(){
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        
        std::cout <<"available layers:"<<std::endl;
        for (const auto & layerProperties : availableLayers){
            std::cout<<"\t"<<layerProperties.layerName <<std::endl;
        }
        
        for (const char * layerName : validationLayers){
            bool layerFound = false;
            
            for (const auto & layerProperties : availableLayers){
                if (strcmp(layerName, layerProperties.layerName) == 0){
                    layerFound = true;
                    break;
                }
            }
            
            if (!layerFound){
                return false;
            }
        }
        
        return true;
    }
    
    void setupDebugCallback(){
        if (!enableValidationLayers) return;
        
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr;
        
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS){
            throw std::runtime_error("failed to set up debug callback!");
        }
    }
    
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                          const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                          const VkAllocationCallbacks* pAllocator,
                                          VkDebugUtilsMessengerEXT* pCallback){
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,
                                                                              "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr){
            return func(instance, pCreateInfo, pAllocator, pCallback);
        } else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback,
                                      const VkAllocationCallbacks * pAllocator){
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr){
            func(instance, callback, pAllocator);
        }
    }
    
    void createLogicalDevice(){
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};
        
        float queuePriority = 1.0f;
        
        for (int queueFamily : uniqueQueueFamilies){
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.emplace_back(queueCreateInfo);
        }
        
        VkPhysicalDeviceFeatures deviceFeatues = {};
        deviceFeatues.samplerAnisotropy = VK_TRUE;
        deviceFeatues.sampleRateShading = VK_TRUE;
        
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatues;
        
        std::vector<const char *> extensions(deviceExtensions);
        
#if defined __APPLE__ && defined __arm64__
        extensions.emplace_back("VK_KHR_portability_subset");
#endif
        
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        
        if (enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }
        
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS){
            throw std::runtime_error("failed to create logical device!");
        }
        
        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.graphicsFamily, 0, &computeQueue);
        vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
    }
    
    void createSurface(){
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
            throw std::runtime_error("failed to create window surface!");
        }
    }
    
    bool checkDeviceExtensionSupport(VkPhysicalDevice device){
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        
        std::cout<<"device extension properties:\n";
        
        for (const auto& extension : availableExtensions){
            std::cout<<"\t"<<extension.extensionName<<std::endl;
            
            auto count = requiredExtensions.erase(extension.extensionName);
            if (count > 0){
                std::cout << "Support device extension : " << extension.extensionName << std::endl;
            }
        }
        
        return requiredExtensions.empty();
    }
    
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device){
        SwapChainSupportDetails details;
        
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        
        if (formatCount != 0){
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        
        if (presentModeCount != 0){
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        
        return details;
    }
    
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & availableFormats){
        if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED){
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        }
        
        for (const auto& availableFormat : availableFormats){
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM
                && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
                return availableFormat;
            }
        }
        
        return availableFormats[0];
    }
    
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> & availablePresentModes){
        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;
        
        for (const auto& availablePresentMode : availablePresentModes){
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
                return availablePresentMode;
            } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR){
                bestMode = availablePresentMode;
            }
        }
        
        return bestMode;
    }
    
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR & capabilites){
        if (capabilites.currentExtent.width != std::numeric_limits<uint32_t>::max()){
            return capabilites.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            
            VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
            
            return actualExtent;
        }
    }
    
    void createSwapChain(){
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 &&
            imageCount > swapChainSupport.capabilities.maxImageCount){
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        
        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {(uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily};
        
        if (indices.graphicsFamily != indices.presentFamily){
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }
        
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS){
            throw std::runtime_error("failed to create swap chain!");
        }
        
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
    
    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size());
        
        for (size_t i = 0; i < swapChainImages.size(); ++i){
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }
    
    void createRenderPass(){
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = msaaSamples;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkAttachmentDescription depthAttachment = {};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        
        VkAttachmentDescription colorAttachmentResolve = {};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkAttachmentReference depthAttachmentRef = {};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        
        VkAttachmentReference colorAttachmentResolveRef = {};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        
        std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
        
		VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS){
            throw std::runtime_error("failed to create render pass!");
        }
    }
    
    void createDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        
        VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriporSetLayout) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    
    void createComputeDescriptorSetLayout(){
        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings;
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        layoutInfo.pBindings = layoutBindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    
    void createDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriporSetLayout);
        
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();
        
        descriptorSets.resize(swapChainImages.size());
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate descriptor set!");
        }
        
        for (size_t i = 0 ; i < swapChainImages.size() ; ++i){
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
            
            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;
            
            std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;
            descriptorWrites[0].pImageInfo = nullptr;
            descriptorWrites[0].pTexelBufferView = nullptr;
            
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = nullptr;
            descriptorWrites[1].pImageInfo = &imageInfo;
            descriptorWrites[1].pTexelBufferView = nullptr;
            
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
    
    void createComputeDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = computeDescriptorPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts = layouts.data();
        
        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate descriptor set!");
        }
        
        for (size_t i = 0 ; i < MAX_FRAMES_IN_FLIGHT ; ++i){
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
            
            std::array<VkWriteDescriptorSet, 3> descriptorWrites = {};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;
            descriptorWrites[0].pImageInfo = nullptr;
            descriptorWrites[0].pTexelBufferView = nullptr;
            
            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            storageBufferInfoLastFrame.buffer = shaderStorageBuffers[(i-1)%MAX_FRAMES_IN_FLIGHT];
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range = sizeof(Particle) * PARTICLE_COUNT;
            
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;
            
            VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
            storageBufferInfoCurrentFrame.buffer = shaderStorageBuffers[i];
            storageBufferInfoCurrentFrame.offset = 0;
            storageBufferInfoCurrentFrame.range = sizeof(Particle) * PARTICLE_COUNT;
            
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = computeDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;
            
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
    
    void createDescriptorPool(){
        std::array<VkDescriptorPoolSize, 2> poolSizes = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
        
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    
    void createComputeDescriptorPool(){
        std::array<VkDescriptorPoolSize, 2> poolSizes = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2;
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    
    void createComputePipeline(){
        auto compShaderCode = readFile("particle_comp.spv");
        
        VkShaderModule compShaderModule = createShaderModule(compShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = compShaderModule;
        computeShaderStageInfo.pName = "main";
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS){
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS){
            throw std::runtime_error("failed to create compute pipeline!");
        }
        
        vkDestroyShaderModule(device, compShaderModule, nullptr);
    }
    
    void createGraphicsPipeline(){
        graphicsPipeline = createGraphicsPipeline("vert.spv", "frag.spv",
                                                  Vertex::getBindingDescription(),
                                                  Vertex::getAttributeDescriptions(),
                                                  &descriporSetLayout, pipelineLayout);
        
//        particleGraphicsPipeline = createGraphicsPipeline("particle_vert.spv", "particle_frag.spv",
//                                                  Vertex::getBindingDescription(),
//                                                  Vertex::getAttributeDescriptions(),
//                                                          nullptr, particlePipelineLayout);
    }
        
    VkPipeline createGraphicsPipeline(const std::string & vertex_shader_name,
                                const std::string & fragment_shader_name,
                                const VkVertexInputBindingDescription & bindingDescription,
                                const std::vector<VkVertexInputAttributeDescription> & attributeDescription,
                                      VkDescriptorSetLayout* descriptor_set_layout,
                                      VkPipelineLayout &pipeline_layout){
        auto vertShaderCode = readFile(vertex_shader_name);
        auto fragShaderCode = readFile(fragment_shader_name);
        
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        
        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();
        
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        
        VkRect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
        
        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
        
        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = .2f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
        
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT ;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        
        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; //Optional
        colorBlending.blendConstants[1] = 0.0f; //Optional
        colorBlending.blendConstants[2] = 0.0f; //Optional
        colorBlending.blendConstants[3] = 0.0f; //Optional
        
        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        
        VkPipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = descriptor_set_layout ? 1 : 0;
        pipelineLayoutInfo.pSetLayouts = descriptor_set_layout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline_layout) != VK_SUCCESS){
            throw std::runtime_error("failed to create pipeline layout!");
        }
        
        VkPipelineDepthStencilStateCreateInfo depthStencil = {};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
        
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages= shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil; //Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState; // Optional
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;
        
        VkPipeline pipeline;
        
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS){
            throw std::runtime_error("failed to create graphics pipeline!");
        }
 
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        
        return pipeline;
    }
    
    VkShaderModule createShaderModule(const std::vector<char>& code){
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS){
            throw std::runtime_error("failed to create shader module!");
        }
        
        return shaderModule;
    }
    
    void createColorResouces(){
        VkFormat colorFormat = swapChainImageFormat;
        
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        
        colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        
        transitionImageLayout(colorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
    }
    
    void createDepthResources(){
        VkFormat depthFormat = findDepthFormat();
        
        createImage(swapChainExtent.width, swapChainExtent.height,1, msaaSamples,
                    depthFormat,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
        
        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
    }
    
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                 VkFormatFeatureFlags features){
        for (VkFormat format : candidates){
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features){
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features){
                return format;
            }
        }
        
        throw std::runtime_error("failed to find supported format!");
    }
    
    VkFormat findDepthFormat(){
        return findSupportedFormat({
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    
    bool hasStencilComponent(VkFormat format){
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
    
    void createTextureImage(){
        int texWidth, texHeight, texChannels;
        
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        
        if (!pixels){
            throw std::runtime_error("failed to load texture image!");
        }
        
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);
        
        stbi_image_free(pixels);
        
        createImage(texWidth, texHeight, mipLevels,VK_SAMPLE_COUNT_1_BIT,
                    VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage, textureImageMemory);
        
        transitionImageLayout(textureImage,
                              VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              mipLevels);
        
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        
        generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
        
        transitionImageLayout(textureImage,
                              VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              mipLevels);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels){
        //Check if image format supports linear blitting
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);
        
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)){
            throw std::runtime_error("texture image format does not support linear bliting!");
        }
        
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;
        
        for (uint32_t i = 1; i < mipLevels; ++i) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            
            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
            
            VkImageBlit blit = {};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1 , mipHeight > 1 ? mipHeight / 2 : 1,1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            
            vkCmdBlitImage(commandBuffer,
                           image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);
            
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            //barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            //官网上写要使用VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL，但是ValidateLayer会报错，改成VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            
            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
            
            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }
        
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        //barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //官网上写要使用VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL，但是ValidateLayer会报错，改成VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        endSignleTimeCommands(commandBuffer);
    }
    
    void createTextureImageView(){
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    }
    
    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspectFlags, uint32_t mipLevels){
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        
        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS){
            throw std::runtime_error("failed to create texture image view!");
        }
        
        return imageView;
    }
    
    void createTextureSampler(){
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0;
        //samplerInfo.minLod = static_cast<float>(mipLevels / 2);
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS){
            throw std::runtime_error("failed to create texture sampler!");
        }
    }
    
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples,
                     VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image,
                     VkDeviceMemory & imageMemory){
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = static_cast<uint32_t>(width);
        imageInfo.extent.height = static_cast<uint32_t>(height);
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;
        imageInfo.flags = 0;
        
        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS){
            throw std::runtime_error("failed to create image!");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate image memory!");
        }
        
        vkBindImageMemory(device, image, imageMemory, 0);
    }
    
    void copyBufferToImage(VkBuffer buffer, VkImage  image, uint32_t width, uint32_t height){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        
        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        
        endSignleTimeCommands(commandBuffer);
    }
    
    VkCommandBuffer beginSingleTimeCommands(){
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
        
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        return commandBuffer;
    }
    
    void endSignleTimeCommands(VkCommandBuffer commandBuffer){
        vkEndCommandBuffer(commandBuffer);
        
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    
    void createShaderStorageBuffers(){
        shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        
        // Initialize particles
        std::default_random_engine rndEngine((unsigned)time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

        // Initial particle positions on a circle
        std::vector<Particle> particles(PARTICLE_COUNT);
        for (auto& particle : particles) {
            float r = 0.25f * sqrt(rndDist(rndEngine));
            float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846;
            float x = r * cos(theta) * HEIGHT / WIDTH;
            float y = r * sin(theta);
            particle.position = glm::vec2(x, y);
            particle.velocity = glm::normalize(glm::vec2(x,y)) * 0.00025f;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }
        
        VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, particles.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i){
            createBuffer(bufferSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         shaderStorageBuffers[i], shaderStorageBuffersMemory[i]);
            
            copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
        }
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    void createVertexBuffer(){
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    void createIndexBuffer(){
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
        
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    void createUniformBuffer(){
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        
        auto size = std::max(swapChainImages.size(), (size_t)MAX_FRAMES_IN_FLIGHT);
        
        uniformBuffers.resize(size);
        uniformBuffersMemory.resize(size);
        uniformBuffersMapped.resize(size);
        
        for (size_t i = 0; i < size ; ++i){
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformBuffersMemory[i]);
            
//            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }
    
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        
        endSignleTimeCommands(commandBuffer);
    }
    
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout, uint32_t mipLevels){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags desinationStage;
        
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        
        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            
            if (hasStencilComponent(format)){
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            desinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            desinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout ==
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            desinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout ==
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            desinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        
        vkCmdPipelineBarrier(commandBuffer, sourceStage, desinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        endSignleTimeCommands(commandBuffer);
    }
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0 ; i < memProperties.memoryTypeCount; ++i){
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties)){
                return i;
            }
        }
        
        throw std::runtime_error("failed to find suitable memory type!");
    }
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer & buffer, VkDeviceMemory & bufferMemory){
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS){
            throw std::runtime_error("failed to create buffer!");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate buffer memory!");
        }
        
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    static std::vector<char> readFile(const std::string & filename){
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()){
            throw std::runtime_error("failed to open file!");
        }
        
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        
        return buffer;
    }
    
    void createFramebuffers(){
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size() ; ++i){
            std::array<VkImageView,3 > attachments = {colorImageView, depthImageView, swapChainImageViews[i]};
            
            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS){
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    
    void createCommandPool(){
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
        
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS){
            throw std::runtime_error("failed to create commnad pool!");
        }
    }

    void createCommandBuffers(){
        commandBuffers.resize(swapChainFramebuffers.size());
        
        createCommandBuffers(commandBuffers);
    }
    
    void createComputeCommandBuffers(){
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        createCommandBuffers(computeCommandBuffers);
    }
    
    void createCommandBuffers(std::vector<VkCommandBuffer> & command_buffers){
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)command_buffers.size();
        
        if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
    
    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer){
        //Bind Compute Pipeline
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
            vkCmdDispatch(commandBuffer, PARTICLE_COUNT / 256, 1, 1);
        }
        
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
            throw std::runtime_error("failed to record command buffer!");
        }
    }
    
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, VkPipeline pipeline, bool is_particle) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;
        
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        
        std::array<VkClearValue, 2> clearValues;
        clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};
        
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        //Bind Graphic Pipeline
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            
            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            
            if (is_particle)
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);
            else
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            
            if (!is_particle){
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);
            }
            
            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapChainExtent.width;
            viewport.height = (float)swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            
            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
            
            if (is_particle){
                vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);
            } else {
                vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            }
        }
        
        vkCmdEndRenderPass(commandBuffer);
        
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
            throw std::runtime_error("failed to record command buffer!");
        }
    }
    
    void createSyncObjects(){
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT ; ++i){
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS){
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }
    
    void updateUniformBuffer(uint32_t currentImage){
#if 1
        static auto startTime = std::chrono::high_resolution_clock::now();
        
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f , 10.0f);
        ubo.proj[1][1] *= -1;
        
        void* data;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
#endif
    }
    
    void updateComputeUniformBuffer(uint32_t currentImage){
#if 0
        UniformBufferObject ubo{};
        ubo.deltaTime = lastFrameTime * 2.0f;
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
#endif
    }
    
    void drawFrame(){
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        //Compute submission
//        vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
//        
//        updateComputeUniformBuffer((uint32_t)currentFrame);
//        
//        vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
//        vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
//        recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);
//        
//        submitInfo.commandBufferCount = 1;
//        submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
//        submitInfo.signalSemaphoreCount = 1;
//        submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];
//        
//        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS){
//            throw std::runtime_error("failed to submit compute command buffer!");
//        }
        
        //Graphics submission
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR){
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        updateUniformBuffer(imageIndex);
        
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex, graphicsPipeline, false);
        
//        VkSemaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame]};
//        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        
        submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS){
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        
        //Present
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional
        
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS){
            throw std::runtime_error("failed to present swap chain image!");
        }
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
    VkSampleCountFlagBits getMaxUsableSampleCount(){
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        
        VkSampleCountFlags counts = std::min(physicalDeviceProperties.limits.framebufferColorSampleCounts,
                                             physicalDeviceProperties.limits.framebufferDepthSampleCounts);
        
        if (counts & VK_SAMPLE_COUNT_64_BIT) {
            return VK_SAMPLE_COUNT_64_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_32_BIT) {
            return VK_SAMPLE_COUNT_32_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_16_BIT) {
            return VK_SAMPLE_COUNT_16_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_8_BIT) {
            return VK_SAMPLE_COUNT_8_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_4_BIT) {
            return VK_SAMPLE_COUNT_4_BIT;
        }
        if (counts & VK_SAMPLE_COUNT_2_BIT) {
            return VK_SAMPLE_COUNT_2_BIT;
        }
        
        return VK_SAMPLE_COUNT_1_BIT;
    }
    
    void loadModel(){
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warning;
        std::string err;
        
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warning , &err, MODEL_PATH.c_str())){
            throw std::runtime_error(err);
        }
        
        std::unordered_map<Vertex, uint32_t> uniqueVertices = {};
        
        for (const auto & shape : shapes){
            for (const auto & index : shape.mesh.indices){
                Vertex vertex = {};
                
                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };
                
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                };
                
                vertex.color = {1.0f, 1.0f, 1.0f};
                
                if (uniqueVertices.count(vertex) == 0){
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.emplace_back(vertex);
                }
                
                indices.emplace_back(uniqueVertices[vertex]);
            }
        }
    }
};

static void framebufferResizeCallback(GLFWwindow * window, int width, int height){
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
