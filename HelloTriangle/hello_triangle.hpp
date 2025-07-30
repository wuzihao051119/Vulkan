#pragma once

#include <memory>

#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#define LOGI(...) spdlog::info(__VA_ARGS__);
#define LOGW(...) spdlog::warn(__VA_ARGS__);
#define LOGE(...) spdlog::error("{}", fmt::format(__VA_ARGS__));
#define LOGD(...) spdlog::debug(__VA_ARGS__);

class HelloTriangle {
  struct SwapchainDimensions {
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
  };

  struct PerFrame {
    VkFence queue_submit_fence = VK_NULL_HANDLE;
    VkCommandPool primary_command_pool = VK_NULL_HANDLE;
    VkCommandBuffer primary_command_buffer = VK_NULL_HANDLE;
    VkSemaphore swapchain_acquire_semaphore = VK_NULL_HANDLE;
    VkSemaphore swapchain_release_semaphore = VK_NULL_HANDLE;
  };

  struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_callback = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    SwapchainDimensions swapchain_dimensions;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    int32_t graphics_queue_index = -1;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkFramebuffer> swapchain_framebuffers;
    std::vector<VkSemaphore> recycled_semaphores;
    std::vector<PerFrame> per_frame;
  };

  struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
  };

public:
  HelloTriangle();
  virtual ~HelloTriangle();

  void create_window();
  bool prepare();
  void update();
  bool resize(const uint32_t width, const uint32_t height);

private:
  bool validate_extensions(const std::vector<const char *> &required,
                           const std::vector<VkExtensionProperties> &available);

  void init_instance();
  void init_device();
  void init_vertex_buffer();
  void init_per_frame(PerFrame &per_frame);
  void teardown_per_frame(PerFrame &per_frame);
  void init_swapchain();
  void init_render_pass();
  VkShaderModule load_shader_module(const std::string &path);
  void init_pipeline();
  void init_framebuffers();
  VkResult acquire_next_image(uint32_t *image);
  void render_triangle(uint32_t swapchain_index);
  VkResult present_image(uint32_t index);

  VkSurfaceFormatKHR select_surface_format(VkPhysicalDevice gpu, VkSurfaceKHR surface, std::vector<VkFormat> const &preferred_formats = {
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_B8G8R8A8_SRGB,
    VK_FORMAT_A8B8G8R8_SRGB_PACK32,
  });
  std::vector<uint32_t> read_shader_binary_u32(const std::string &filename);

public:
  GLFWwindow *window;

private:
  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vertex_buffer_memory = VK_NULL_HANDLE;
  VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

  Context context;
  std::unique_ptr<VkInstance> vk_instance;
};

std::unique_ptr<HelloTriangle> create_hello_triangle();
