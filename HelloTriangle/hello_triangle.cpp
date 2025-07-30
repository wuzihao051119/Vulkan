#define VK_NO_PROTOTYPES
#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION

#define VK_USE_PLATFORM_XCB_KHR

#include "hello_triangle.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include <volk/volk.h>

#define VK_CHECK(x)                                                         \
  do {                                                                      \
    VkResult err = x;                                                       \
    if (err)                                                                \
      throw std::runtime_error("Detected Vulkan error: " + to_string(err)); \
  } while (0)

static void error_callback(int error, const char *description) {
  LOGE("GLFW Error (code: {}): {}", error, description);
}

static const std::string to_string(VkResult result) {
  switch (result) {
#define STR(r) \
  case VK_##r: \
    return #r
		STR(NOT_READY);
		STR(TIMEOUT);
		STR(EVENT_SET);
		STR(EVENT_RESET);
		STR(INCOMPLETE);
		STR(ERROR_OUT_OF_HOST_MEMORY);
		STR(ERROR_OUT_OF_DEVICE_MEMORY);
		STR(ERROR_INITIALIZATION_FAILED);
		STR(ERROR_DEVICE_LOST);
		STR(ERROR_MEMORY_MAP_FAILED);
		STR(ERROR_LAYER_NOT_PRESENT);
		STR(ERROR_EXTENSION_NOT_PRESENT);
		STR(ERROR_FEATURE_NOT_PRESENT);
		STR(ERROR_INCOMPATIBLE_DRIVER);
		STR(ERROR_TOO_MANY_OBJECTS);
		STR(ERROR_FORMAT_NOT_SUPPORTED);
		STR(ERROR_FRAGMENTED_POOL);
		STR(ERROR_UNKNOWN);
		STR(ERROR_OUT_OF_POOL_MEMORY);
		STR(ERROR_INVALID_EXTERNAL_HANDLE);
		STR(ERROR_FRAGMENTATION);
		STR(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
		STR(PIPELINE_COMPILE_REQUIRED);
		STR(ERROR_SURFACE_LOST_KHR);
		STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
		STR(SUBOPTIMAL_KHR);
		STR(ERROR_OUT_OF_DATE_KHR);
		STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
		STR(ERROR_VALIDATION_FAILED_EXT);
		STR(ERROR_INVALID_SHADER_NV);
#undef STR
  default:
    return "UNKNOWN_ERROR";
  }
}

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
  VkDebugUtilsMessageTypeFlagsEXT message_type,
  const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
  void *user_data
) {
  (void)user_data;

  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    LOGE("{} Validation Layer: Error: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage)
  else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    LOGE("{} Validation Layer: Warning: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage)
  else if (message_type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
    LOGI("{} Validation Layer: Performance warning: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage)
  else
    LOGI("{} Validation Layer: Information: {}: {}", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage)
  return VK_FALSE;
}
#endif

HelloTriangle::HelloTriangle() {}

HelloTriangle::~HelloTriangle() {
  vkDeviceWaitIdle(context.device);

  for (auto &framebuffer : context.swapchain_framebuffers)
    vkDestroyFramebuffer(context.device, framebuffer, nullptr);

  for (auto &per_frame : context.per_frame)
    teardown_per_frame(per_frame);

  context.per_frame.clear();

  for (auto semaphore : context.recycled_semaphores)
    vkDestroySemaphore(context.device, semaphore, nullptr);

  if (context.pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(context.device, context.pipeline, nullptr);

  if (context.pipeline_layout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(context.device, context.pipeline_layout, nullptr);

  if (context.render_pass != VK_NULL_HANDLE)
    vkDestroyRenderPass(context.device, context.render_pass, nullptr);

  for (VkImageView image_view : context.swapchain_image_views)
    vkDestroyImageView(context.device, image_view, nullptr);

  if (context.swapchain != VK_NULL_HANDLE)
    vkDestroySwapchainKHR(context.device, context.swapchain, nullptr);

  if (context.surface != VK_NULL_HANDLE)
    vkDestroySurfaceKHR(context.instance, context.surface, nullptr);

  if (vertex_buffer_allocation != VK_NULL_HANDLE)
    vmaDestroyBuffer(context.vma_allocator, vertex_buffer, nullptr);

  if (context.vma_allocator != VK_NULL_HANDLE)
    vmaDestroyAllocator(context.vma_allocator);

  if (context.device != VK_NULL_HANDLE)
    vkDestroyDevice(context.device, nullptr);

  if (context.debug_callback != VK_NULL_HANDLE)
    vkDestroyDebugUtilsMessengerEXT(context.instance, context.debug_callback, nullptr);

  vk_instance.reset();
}

void HelloTriangle::create_window() {
  if (!glfwInit())
    throw std::runtime_error("GLFW couldn't be initialized.");

  glfwSetErrorCallback(error_callback);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(1280, 720, "Hello Triangle", nullptr, nullptr);
  if (!window)
    throw std::runtime_error("GLFW window conldn't be created");

  glfwSetWindowCloseCallback(window, [](GLFWwindow *window) {
    glfwDestroyWindow(window);
    glfwTerminate();
  });
}

bool HelloTriangle::prepare() {
  assert(window != nullptr);

  init_instance();
  vk_instance = std::make_unique<VkInstance>(context.instance);

  glfwCreateWindowSurface(context.instance, window, nullptr, &context.surface);

  int width, height;
  glfwGetWindowSize(window, &width, &height);
  context.swapchain_dimensions.width = static_cast<uint32_t>(width);
  context.swapchain_dimensions.height = static_cast<uint32_t>(height);

  if (!context.surface)
    throw std::runtime_error("Failed to create window surface.");

  init_device();
  init_vertex_buffer();
  init_swapchain();
  init_render_pass();
  init_pipeline();
  init_framebuffers();

  return true;
}

void HelloTriangle::update() {
  uint32_t index;
  auto res = acquire_next_image(&index);

  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    resize(context.swapchain_dimensions.width, context.swapchain_dimensions.height);
    res = acquire_next_image(&index);
  }

  if (res != VK_SUCCESS) {
    vkQueueWaitIdle(context.queue);
    return;
  }

  render_triangle(index);
  res = present_image(index);

  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR)
    resize(context.swapchain_dimensions.width, context.swapchain_dimensions.height);
  else if (res != VK_SUCCESS)
    LOGE("Failed to present swapchain image.");
}

bool HelloTriangle::resize(const uint32_t width, const uint32_t height) {
  if (context.device == VK_NULL_HANDLE)
    return false;

  VkSurfaceCapabilitiesKHR surface_properties;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.gpu, context.surface, &surface_properties));

  if (surface_properties.currentExtent.width == context.swapchain_dimensions.width &&
    surface_properties.currentExtent.height == context.swapchain_dimensions.height)
    return false;

  vkDeviceWaitIdle(context.device);

  for (auto &framebuffer : context.swapchain_framebuffers)
    vkDestroyFramebuffer(context.device, framebuffer, nullptr);

  init_swapchain();
  init_framebuffers();

  return true;
}

bool HelloTriangle::validate_extensions(
  const std::vector<const char *> &required,
  const std::vector<VkExtensionProperties> &available
) {
  for (auto extension : required) {
    bool found = false;
    for (auto &available_extension : available) {
      if (strcmp(available_extension.extensionName, extension) == 0) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

void HelloTriangle::init_instance() {
  LOGI("Initializing vulkan instance.");

  if (volkInitialize())
    throw std::runtime_error("Failed to initialize volk.");

  uint32_t instance_extension_count;
  VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr));

  std::vector<VkExtensionProperties> available_instance_extensions(instance_extension_count);
  VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_instance_extensions.data()));

  std::vector<const char *> required_instance_extensions { VK_KHR_SURFACE_EXTENSION_NAME };

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
  bool has_debug_utils = false;
  for (const auto &ext: available_instance_extensions) {
    if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
      has_debug_utils = true;
      required_instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      break;
    }
  }

  if (!has_debug_utils) {
    LOGW("{} not supported or available", VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    LOGW("Make sure to compile the sample in debug mode and/or enable the validation layers");
  }
#endif

#if defined(VKB_ENABLE_PORTABILITY)
  required_instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  bool portability_enumeration_available = false;
  if (std::ranges::any_of(
    available_instance_extensions,
    [](VkExtensionProperties const &extension) {
      return strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
    }
  )) {
    required_instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    portability_enumeration_available = true;
  }
#endif

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
  required_instance_extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WIN32_KHR)
  required_instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
  required_instance_extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
  required_instance_extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
  required_instance_extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  required_instance_extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DISPLAY_KHR)
  required_instance_extensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#else
#  pragma error Platform not supported
#endif

  if (!validate_extensions(required_instance_extensions, available_instance_extensions))
    throw std::runtime_error("Required instance extensions are missing.");

  std::vector<const char *> requested_instance_layers {};

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
  char const *validationLayer = "VK_LAYER_KHRONOS_validation";

  uint32_t instance_layer_count;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr));

  std::vector<VkLayerProperties> supported_instance_layers(instance_layer_count);
  VK_CHECK(vkEnumerateInstanceLayerProperties(&instance_layer_count, supported_instance_layers.data()));

  if (std::ranges::any_of(
    supported_instance_layers,
    [&validationLayer](auto const &lp) {
      return strcmp(lp.layerName, validationLayer) == 0;
    }
  )) {
    requested_instance_layers.push_back(validationLayer);
    LOGI("Enabled Validation Layer {}", validationLayer);
  } else
    LOGW("Validation Layer {} is not available", validationLayer);
#endif

  VkApplicationInfo app {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "Hello Triangle",
    .pEngineName = "Vulkan Samples",
    .apiVersion= VK_API_VERSION_1_1,
  };

  VkInstanceCreateInfo instance_info {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &app,
    .enabledLayerCount = static_cast<uint32_t>(requested_instance_layers.size()),
    .ppEnabledLayerNames = requested_instance_layers.data(),
    .enabledExtensionCount = static_cast<uint32_t>(required_instance_extensions.size()),
    .ppEnabledExtensionNames = required_instance_extensions.data(),
  };

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
  VkDebugUtilsMessengerCreateInfoEXT debug_utils_create_info = {
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
  };

  if (has_debug_utils) {
    debug_utils_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    debug_utils_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debug_utils_create_info.pfnUserCallback = debug_callback;

    instance_info.pNext = &debug_utils_create_info;
  }
#endif

#if defined(VKB_ENABLE_PORTABILITY)
  if (portability_enumeration_available)
    instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  VK_CHECK(vkCreateInstance(&instance_info, nullptr, &context.instance));
  volkLoadInstance(context.instance);

#if defined(VKB_DEBUG) || defined(VKB_VALIDATION_LAYERS)
  if (has_debug_utils)
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(context.instance, &debug_utils_create_info, nullptr, &context.debug_callback));
#endif
}

void HelloTriangle::init_device() {
  LOGI("Initializing vulkan device.");

  uint32_t gpu_count;
  VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &gpu_count, nullptr));

  if (gpu_count < 1)
    throw std::runtime_error("No physical device found.");

  std::vector<VkPhysicalDevice> gpus(gpu_count);
  VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &gpu_count, gpus.data()));

  for (size_t i = 0; i < gpu_count && context.graphics_queue_index < 0; i++) {
    context.gpu = gpus[i];

    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(context.gpu, &queue_family_count, nullptr);

    if (queue_family_count < 1)
      throw std::runtime_error("No queue family found.");

    std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(context.gpu, &queue_family_count, queue_family_properties.data());

    for (uint32_t i = 0; i < queue_family_count; i++) {
      VkBool32 supports_present;
      vkGetPhysicalDeviceSurfaceSupportKHR(context.gpu, i, context.surface, &supports_present);

      if ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && supports_present) {
        context.graphics_queue_index = i;
        break;
      }
    }
  }

  if (context.graphics_queue_index < 0)
    throw std::runtime_error("Did not find suitable device with a queue that supports graphics and presentation.");

  uint32_t device_extension_count;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, nullptr));
  std::vector<VkExtensionProperties> device_extensions(device_extension_count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, device_extensions.data()));

  std::vector<const char *> required_device_extensions { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

  if (!validate_extensions(required_device_extensions, device_extensions))
    throw std::runtime_error("Required device extensions are missing.");

#if defined(VKB_ENABLE_PORTABILITY)
  if (std::ranges::any_of(
    device_extensions,
    [](VkExtensionProperties const &extension) {
      return strcmp(extension.extensionName, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0;
    }
  )) {
    required_device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
  }
#endif

  const float queue_priority = 1.0f;

  VkDeviceQueueCreateInfo queue_info {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = static_cast<uint32_t>(context.graphics_queue_index),
    .queueCount = 1,
    .pQueuePriorities = &queue_priority,
  };

  VkDeviceCreateInfo device_info {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queue_info,
    .enabledExtensionCount = static_cast<uint32_t>(required_device_extensions.size()),
    .ppEnabledExtensionNames = required_device_extensions.data(),
  };

  VK_CHECK(vkCreateDevice(context.gpu, &device_info, nullptr, &context.device));
  volkLoadDevice(context.device);

  vkGetDeviceQueue(context.device, context.graphics_queue_index, 0, &context.queue);

  VmaVulkanFunctions vma_vulkan_func {
    .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
    .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
  };

  VmaAllocatorCreateInfo allocator_info {
    .physicalDevice = context.gpu,
    .device = context.device,
    .pVulkanFunctions = &vma_vulkan_func,
    .instance = context.instance,
  };

  VkResult result = vmaCreateAllocator(&allocator_info, &context.vma_allocator);
  if (result != VK_SUCCESS)
    throw std::runtime_error("Could not create allocator for VMA allocator");
}

void HelloTriangle::init_vertex_buffer() {
  const std::vector<Vertex> vertices = {
    { { 0.5f, -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f } },
  };

  const VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

  VkBufferCreateInfo buffer_info {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = buffer_size,
    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
  };

  VmaAllocationCreateInfo buffer_alloc_ci {
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO,
    .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  };

  VmaAllocationInfo buffer_alloc_info {};
  vmaCreateBuffer(context.vma_allocator, &buffer_info, &buffer_alloc_ci, &vertex_buffer, &vertex_buffer_allocation, &buffer_alloc_info);

  if (buffer_alloc_info.pMappedData)
    memcpy(buffer_alloc_info.pMappedData, vertices.data(), buffer_size);
  else
    throw std::runtime_error("Could not map vertex buffer.");
}

void HelloTriangle::init_swapchain() {
  VkSurfaceCapabilitiesKHR surface_properties;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.gpu, context.surface, &surface_properties));

  VkSurfaceFormatKHR format = select_surface_format(context.gpu, context.surface);

  VkExtent2D swapchain_size {};
  if (surface_properties.currentExtent.width == 0xFFFFFFFF) {
    swapchain_size.width = context.swapchain_dimensions.width;
    swapchain_size.height = context.swapchain_dimensions.height;
  } else
    swapchain_size = surface_properties.currentExtent;

  VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;

  uint32_t desired_swapchain_images = surface_properties.minImageCount + 1;
  if ((surface_properties.maxImageCount > 0) && (desired_swapchain_images > surface_properties.maxImageCount))
    desired_swapchain_images = surface_properties.maxImageCount;

  VkSurfaceTransformFlagBitsKHR pre_transform;
  if (surface_properties.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  else
    pre_transform = surface_properties.currentTransform;

  VkSwapchainKHR old_swapchain = context.swapchain;

  VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
    composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
    composite = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
    composite = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
  else if (surface_properties.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
    composite = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;

  VkSwapchainCreateInfoKHR info {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .surface = context.surface,
    .minImageCount = desired_swapchain_images,
    .imageFormat = format.format,
    .imageColorSpace = format.colorSpace,
    .imageExtent = swapchain_size,
    .imageArrayLayers = 1,
    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .preTransform = pre_transform,
    .compositeAlpha = composite,
    .presentMode = swapchain_present_mode,
    .clipped = true,
    .oldSwapchain = old_swapchain,
  };

  VK_CHECK(vkCreateSwapchainKHR(context.device, &info, nullptr, &context.swapchain));

  if (old_swapchain != VK_NULL_HANDLE) {
    for (VkImageView image_view : context.swapchain_image_views)
      vkDestroyImageView(context.device, image_view, nullptr);

    for (auto &per_frame : context.per_frame)
      teardown_per_frame(per_frame);

      context.swapchain_image_views.clear();

      vkDestroySwapchainKHR(context.device, old_swapchain, nullptr);
  }

  context.swapchain_dimensions = { swapchain_size.width, swapchain_size.height, format.format };

  uint32_t image_count;
  VK_CHECK(vkGetSwapchainImagesKHR(context.device, context.swapchain, &image_count, nullptr));

  std::vector<VkImage> swapchain_images(image_count);
  VK_CHECK(vkGetSwapchainImagesKHR(context.device, context.swapchain, &image_count, swapchain_images.data()));

  context.per_frame.clear();
  context.per_frame.resize(image_count);

  for (size_t i = 0; i < image_count; i++)
    init_per_frame(context.per_frame[i]);

  for (size_t i = 0; i < image_count; i++) {
    VkImageViewCreateInfo view_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = swapchain_images[i],
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = context.swapchain_dimensions.format,
      .subresourceRange = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
      },
    };

    VkImageView image_view;
    VK_CHECK(vkCreateImageView(context.device, &view_info, nullptr, &image_view));

    context.swapchain_image_views.push_back(image_view);
  }
}

void HelloTriangle::init_render_pass() {
  VkAttachmentDescription attachment {
    .format = context.swapchain_dimensions.format,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
  };

  VkAttachmentReference color_ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

  VkSubpassDescription subpass {
    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
    .colorAttachmentCount = 1,
    .pColorAttachments = &color_ref,
  };

  VkSubpassDependency dependency {
    .srcSubpass = VK_SUBPASS_EXTERNAL,
    .dstSubpass = 0,
    .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  };

  dependency.srcAccessMask = 0;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo rp_info {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &attachment,
    .subpassCount = 1,
    .pSubpasses = &subpass,
    .dependencyCount = 1,
    .pDependencies = &dependency,
  };

  VK_CHECK(vkCreateRenderPass(context.device, &rp_info, nullptr, &context.render_pass));
}

VkShaderModule HelloTriangle::load_shader_module(const std::string &path) {
  auto spirv = read_shader_binary_u32(path);

  VkShaderModuleCreateInfo module_info {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spirv.size() * sizeof(uint32_t),
    .pCode = spirv.data(),
  };

  VkShaderModule shader_module;
  VK_CHECK(vkCreateShaderModule(context.device, &module_info, nullptr, &shader_module));

  return shader_module;
}

void HelloTriangle::init_pipeline() {
  VkPipelineLayoutCreateInfo layout_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
  };
  VK_CHECK(vkCreatePipelineLayout(context.device, &layout_info, nullptr, &context.pipeline_layout));

  VkPipelineInputAssemblyStateCreateInfo input_assembly {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };

  VkVertexInputBindingDescription binding_description {
    .binding = 0,
    .stride = sizeof(Vertex),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions {
    {
      { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, position) },
      { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) },
    },
  };

  VkPipelineVertexInputStateCreateInfo vertex_input {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &binding_description,
    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size()),
    .pVertexAttributeDescriptions = attribute_descriptions.data(),
  };

  VkPipelineRasterizationStateCreateInfo raster {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .cullMode = VK_CULL_MODE_BACK_BIT,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
    .lineWidth = 1.0f,
  };

  VkPipelineColorBlendAttachmentState blend_attachment = {
    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  VkPipelineColorBlendStateCreateInfo blend {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &blend_attachment,
  };

  VkPipelineViewportStateCreateInfo viewport {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .scissorCount = 1,
  };

  VkPipelineDepthStencilStateCreateInfo depth_stencil {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  };

  VkPipelineMultisampleStateCreateInfo multisample {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  std::array<VkDynamicState, 2> dynamics { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

  VkPipelineDynamicStateCreateInfo dynamic {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .dynamicStateCount = 2,
    .pDynamicStates = dynamics.data(),
  };

  std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages {};

  shader_stages[0] = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_VERTEX_BIT,
    .module = load_shader_module("triangle.vert.spv"),
    .pName = "main",
  };

  shader_stages[1] = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
    .module = load_shader_module("triangle.frag.spv"),
    .pName = "main",
  };

  VkGraphicsPipelineCreateInfo pipe {
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = static_cast<uint32_t>(shader_stages.size()),
    .pStages = shader_stages.data(),
    .pVertexInputState = &vertex_input,
    .pInputAssemblyState = &input_assembly,
    .pViewportState = &viewport,
    .pRasterizationState = &raster,
    .pMultisampleState = &multisample,
    .pDepthStencilState = &depth_stencil,
    .pColorBlendState = &blend,
    .pDynamicState = &dynamic,
    .layout = context.pipeline_layout,
    .renderPass = context.render_pass,
  };
  VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipe, nullptr, &context.pipeline));

  vkDestroyShaderModule(context.device, shader_stages[0].module, nullptr);
  vkDestroyShaderModule(context.device, shader_stages[1].module, nullptr);
}

void HelloTriangle::init_per_frame(PerFrame &per_frame) {
  VkFenceCreateInfo info {
    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    .flags = VK_FENCE_CREATE_SIGNALED_BIT,
  };
  VK_CHECK(vkCreateFence(context.device, &info, nullptr, &per_frame.queue_submit_fence));

  VkCommandPoolCreateInfo cmd_pool_info {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    .queueFamilyIndex = static_cast<uint32_t>(context.graphics_queue_index),
  };
  VK_CHECK(vkCreateCommandPool(context.device, &cmd_pool_info, nullptr, &per_frame.primary_command_pool));

  VkCommandBufferAllocateInfo cmd_buf_info {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = per_frame.primary_command_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  VK_CHECK(vkAllocateCommandBuffers(context.device, &cmd_buf_info, &per_frame.primary_command_buffer));
}

void HelloTriangle::teardown_per_frame(PerFrame &per_frame) {
  if (per_frame.queue_submit_fence != VK_NULL_HANDLE) {
    vkDestroyFence(context.device, per_frame.queue_submit_fence, nullptr);
    per_frame.queue_submit_fence = VK_NULL_HANDLE;
  }

  if (per_frame.primary_command_buffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(context.device, per_frame.primary_command_pool, 1, &per_frame.primary_command_buffer);
    per_frame.primary_command_buffer = VK_NULL_HANDLE;
  }

  if (per_frame.primary_command_pool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(context.device, per_frame.primary_command_pool, nullptr);
    per_frame.primary_command_pool = VK_NULL_HANDLE;
  }

  if (per_frame.swapchain_acquire_semaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(context.device, per_frame.swapchain_acquire_semaphore, nullptr);
    per_frame.swapchain_acquire_semaphore = VK_NULL_HANDLE;
  }

  if (per_frame.swapchain_release_semaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(context.device, per_frame.swapchain_release_semaphore, nullptr);
    per_frame.swapchain_release_semaphore = VK_NULL_HANDLE;
  }
}

void HelloTriangle::init_framebuffers() {
  context.swapchain_framebuffers.clear();

  for (auto &image_view : context.swapchain_image_views) {
    VkFramebufferCreateInfo fb_info {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = context.render_pass,
      .attachmentCount = 1,
      .pAttachments = &image_view,
      .width = context.swapchain_dimensions.width,
      .height = context.swapchain_dimensions.height,
      .layers = 1,
    };

    VkFramebuffer framebuffer;
    VK_CHECK(vkCreateFramebuffer(context.device, &fb_info, nullptr, &framebuffer));

    context.swapchain_framebuffers.push_back(framebuffer);
  }
}

VkResult HelloTriangle::acquire_next_image(uint32_t *image) {
  VkSemaphore acquire_semaphore;
  if (context.recycled_semaphores.empty()) {
    VkSemaphoreCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    VK_CHECK(vkCreateSemaphore(context.device, &info, nullptr, &acquire_semaphore));
  } else {
    acquire_semaphore = context.recycled_semaphores.back();
    context.recycled_semaphores.pop_back();
  }

  VkResult res = vkAcquireNextImageKHR(context.device, context.swapchain, UINT64_MAX, acquire_semaphore, VK_NULL_HANDLE, image);

  if (res != VK_SUCCESS) {
    context.recycled_semaphores.push_back(acquire_semaphore);
    return res;
  }

  if (context.per_frame[*image].queue_submit_fence != VK_NULL_HANDLE) {
    vkWaitForFences(context.device, 1, &context.per_frame[*image].queue_submit_fence, true, UINT64_MAX);
    vkResetFences(context.device, 1, &context.per_frame[*image].queue_submit_fence);
  }

  if (context.per_frame[*image].primary_command_pool != VK_NULL_HANDLE)
    vkResetCommandPool(context.device, context.per_frame[*image].primary_command_pool, 0);

  VkSemaphore old_semaphore = context.per_frame[*image].swapchain_acquire_semaphore;

  if (old_semaphore != VK_NULL_HANDLE)
    context.recycled_semaphores.push_back(old_semaphore);

  context.per_frame[*image].swapchain_acquire_semaphore = acquire_semaphore;

  return VK_SUCCESS;
}

void HelloTriangle::render_triangle(uint32_t swapchain_index) {
  VkFramebuffer framebuffer = context.swapchain_framebuffers[swapchain_index];
  VkCommandBuffer cmd = context.per_frame[swapchain_index].primary_command_buffer;
  VkCommandBufferBeginInfo begin_info {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(cmd, &begin_info);

  VkClearValue clear_value {
    .color = {
      {0.01f, 0.01f, 0.033f, 1.0f},
    },
  };

  VkRenderPassBeginInfo rp_begin {
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    .renderPass = context.render_pass,
    .framebuffer = framebuffer,
    .renderArea = {
      .extent = {
        .width = context.swapchain_dimensions.width,
        .height = context.swapchain_dimensions.height,
      },
    },
    .clearValueCount = 1,
    .pClearValues = &clear_value,
  };
  vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, context.pipeline);

  VkViewport vp {
    .width = static_cast<float>(context.swapchain_dimensions.width),
    .height = static_cast<float>(context.swapchain_dimensions.height),
    .minDepth = 0.0f,
    .maxDepth = 1.0f,
  };
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D scissor {
    .extent = {
      .width = context.swapchain_dimensions.width,
      .height = context.swapchain_dimensions.height,
    },
  };
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  VkDeviceSize offset = { 0 };
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);

  vkCmdDraw(cmd, 3, 1, 0, 0);

  vkCmdEndRenderPass(cmd);

  VK_CHECK(vkEndCommandBuffer(cmd));

  if (context.per_frame[swapchain_index].swapchain_release_semaphore == VK_NULL_HANDLE) {
    VkSemaphoreCreateInfo semaphore_info {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    VK_CHECK(vkCreateSemaphore(context.device, &semaphore_info, nullptr, &context.per_frame[swapchain_index].swapchain_release_semaphore));
  }

  VkPipelineStageFlags wait_stage { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
  VkSubmitInfo info {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &context.per_frame[swapchain_index].swapchain_acquire_semaphore,
    .pWaitDstStageMask = &wait_stage,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &context.per_frame[swapchain_index].swapchain_release_semaphore,
  };
  VK_CHECK(vkQueueSubmit(context.queue, 1, &info, context.per_frame[swapchain_index].queue_submit_fence));
}

VkResult HelloTriangle::present_image(uint32_t index) {
  VkPresentInfoKHR present {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &context.per_frame[index].swapchain_release_semaphore,
    .swapchainCount = 1,
    .pSwapchains = &context.swapchain,
    .pImageIndices = &index,
  };
  return vkQueuePresentKHR(context.queue, &present);
}

VkSurfaceFormatKHR HelloTriangle::select_surface_format(VkPhysicalDevice gpu, VkSurfaceKHR surface, std::vector<VkFormat> const &preferred_formats) {
  uint32_t surface_format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &surface_format_count, nullptr);
  assert(0 < surface_format_count);
  std::vector<VkSurfaceFormatKHR> supported_surface_formats(surface_format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &surface_format_count, supported_surface_formats.data());

  auto it = std::ranges::find_if(supported_surface_formats,
    [&preferred_formats](VkSurfaceFormatKHR surface_format) {
      return std::ranges::any_of(preferred_formats,
        [&surface_format](VkFormat format) {
          return format == surface_format.format;
        }
      );
    }
  );

  return it != supported_surface_formats.end() ? *it : supported_surface_formats[0];
}

std::vector<uint32_t> HelloTriangle::read_shader_binary_u32(const std::string &filename) {
  std::ifstream file { filename, std::ios::binary | std::ios::ate };
  if (!file.is_open())
    throw std::runtime_error("Failed to open file for reading at path: " + filename);

  std::error_code ec;
  size_t count = std::filesystem::file_size(filename, ec);
  if (ec)
    count = 0;

  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(count);
  file.read(reinterpret_cast<char *>(buffer.data()), count);

  assert(buffer.size() % sizeof(uint32_t) == 0);
  auto spirv = std::vector<uint32_t>(reinterpret_cast<uint32_t *>(buffer.data()), reinterpret_cast<uint32_t *>(buffer.data()) + buffer.size() / sizeof(uint32_t));
  return spirv;
}

std::unique_ptr<HelloTriangle> create_hello_triangle() {
  return std::make_unique<HelloTriangle>();
}
