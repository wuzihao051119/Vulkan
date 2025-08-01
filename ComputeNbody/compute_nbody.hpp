#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define VK_CHECK(x) x

class ComputeNbody {
  struct SwapchainExtent {
    float width;
    float height;
    VkFormat format;
    VkColorSpaceKHR colorSpace;
  };

  struct Particle {
    glm::vec4 position;
    glm::vec4 velocity;
  };

  struct Context {
    // Device
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t graphics_queue_family_index = -1;
    uint32_t compute_queue_family_index = -1;

    // Swapchain
    SwapchainExtent swapchain_extent;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    // Queue
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;

    // Descriptor
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    VkDescriptorSetLayout graphics_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet graphics_descriptor_set = VK_NULL_HANDLE;

    VkDescriptorSetLayout compute_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet compute_descriptor_set = VK_NULL_HANDLE;

    // Command
    VkCommandPool compute_command_pool = VK_NULL_HANDLE;
    VkCommandBuffer compute_command_buffer = VK_NULL_HANDLE;

    // Semaphore
    VkSemaphore graphics_semaphore;
    VkSemaphore compute_semaphore;

    // Buffer
    VkBuffer graphics_uniform_buffer;
    VkImageView graphics_particle_image;
    VkSampler grahpics_particle_sampler;
    VkFormat graphics_particle_format;
    VkImageView graphics_gradient_image;
    VkSampler graphics_gradient_sampler;
    VkFormat graphics_gradient_format;

    VkBuffer compute_storage_buffer;
    VkBuffer compute_uniform_buffer;

    // Pipeline
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;

    VkPipelineLayout graphics_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline = VK_NULL_HANDLE;

    VkPipelineLayout compute_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipeline_calculate = VK_NULL_HANDLE;
    VkPipeline compute_pipeline_integrate = VK_NULL_HANDLE;
  };

public:
  ComputeNbody();
  virtual ~ComputeNbody();

private:
  void create_window();
  void create_instance();
  void create_surface();
  void create_device();
  void create_swapchain();
  void create_queue();
  void create_descriptor_pool();
  bool is_depth_stencil_format(VkFormat &format);
  VkDescriptorBufferInfo create_descriptor(VkBuffer &buffer, VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
  VkDescriptorImageInfo create_descriptor(VkImageView &image_view, VkSampler &sampler, VkFormat &format, VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  void create_graphics_descriptor();
  void create_compute_descriptor();
  void create_semaphore();
  VkPipelineShaderStageCreateInfo load_shader(const char *name, const char *path, VkShaderStageFlagBits shader_stage_flag);
  void create_graphics_pipeline();
  void create_compute_pipeline();
  void create_compute_command();

  void load_assets();
  void build_command_buffers();
  void build_compute_command_buffer();
  void setup_descriptor_pool();
  void setup_descriptor_set_layout();
  void setup_descriptor_set();
  void prepare_graphics();
  void prepare_compute();
  void draw();

public:
  GLFWwindow *window;

private:
  Context context;
};
