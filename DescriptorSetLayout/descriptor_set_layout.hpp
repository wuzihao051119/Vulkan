#pragma once

#define NDEBUG

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char *> validationLayers = {
  "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
  glm::vec2 position;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription();
  static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
  { { -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
  { { 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
  { { 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } },
  { { -0.5f, 0.5f }, {1.0f, 1.0f, 1.0f } },
};

const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };

class DescriptorSetLayout {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();

  void cleanupSwapChain();
  void recreateSwapChain();

  void createInstance();
  void createSurface();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();
  void createImageViews();
  void createFramebuffers();
  void createDescriptorSetLayout();
  void createRenderPass();
  void createPipeline();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();

  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
  void updateUniformBuffer(uint32_t currentImage);
  void drawFrame();

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
  bool isDeviceSuitable(VkPhysicalDevice device);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
  std::vector<const char *> getRequiredExtensions();
  bool checkValidationLayerSupport();
  VkShaderModule createShaderModule(const std::vector<char> &code);
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
  VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentMode);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallback, void *pUserData);
  static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
  static std::vector<char> readFile(const std::string &filename);

private:
  GLFWwindow *window;

  VkInstance instance;
  VkSurfaceKHR surface;

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;

  VkSwapchainKHR swapChain;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;

  VkQueue graphicsQueue;
  VkQueue presentQueue;

  VkCommandPool commandPool;

  VkDescriptorSetLayout descriptorSetLayout;
  VkRenderPass renderPass;
  VkPipelineLayout pipelineLayout;
  VkPipeline pipeline;

  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;
  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  std::vector<VkCommandBuffer> commandBuffers;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  uint32_t currentFrame = 0;

  bool framebufferResized = false;
};
