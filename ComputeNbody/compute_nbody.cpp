#include "compute_nbody.hpp"
#include "vulkan/vulkan_core.h"

#include <array>
#include <vector>

ComputeNbody::ComputeNbody() {}

ComputeNbody::~ComputeNbody() {
  vkDestroyBuffer(context.device, context.graphics_uniform_buffer, nullptr);
  vkDestroyBuffer(context.device, context.compute_uniform_buffer, nullptr);
  vkDestroyBuffer(context.device, context.compute_storage_buffer, nullptr);

  vkDestroyImageView(context.device, context.graphics_particle_image, nullptr);
  vkDestroyImageView(context.device, context.graphics_gradient_image, nullptr);

  vkDestroySampler(context.device, context.grahpics_particle_sampler, nullptr);
  vkDestroySampler(context.device, context.graphics_gradient_sampler, nullptr);

  vkDestroyPipelineLayout(context.device, context.graphics_pipeline_layout, nullptr);
  vkDestroyPipeline(context.device, context.graphics_pipeline, nullptr);

  vkDestroyPipelineLayout(context.device, context.compute_pipeline_layout, nullptr);
  vkDestroyPipeline(context.device, context.compute_pipeline_calculate, nullptr);
  vkDestroyPipeline(context.device, context.compute_pipeline_integrate, nullptr);

  vkDestroyRenderPass(context.device, context.render_pass, nullptr);
  vkDestroyPipelineCache(context.device, context.pipeline_cache, nullptr);

  vkDestroyDescriptorPool(context.device, context.descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(context.device, context.graphics_descriptor_set_layout, nullptr);
  vkDestroyDescriptorSetLayout(context.device, context.compute_descriptor_set_layout, nullptr);

  vkDestroySemaphore(context.device, context.graphics_semaphore, nullptr);
  vkDestroySemaphore(context.device, context.compute_semaphore, nullptr);

  vkDestroySwapchainKHR(context.device, context.swapchain, nullptr);
  vkDestroyDevice(context.device, nullptr);

  vkDestroySurfaceKHR(context.instance, context.surface, nullptr);
  vkDestroyInstance(context.instance, nullptr);
}

void ComputeNbody::create_window() {
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NO_API);

  int width = 1280;
  int height = 720;
  glfwCreateWindow(width, height, "ComputeNbody", nullptr, nullptr);
}

void ComputeNbody::create_instance() {
  uint32_t instance_extension_count;
  vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
  std::vector<VkExtensionProperties> available_extension_properties(instance_extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_extension_properties.data());

  uint32_t instance_layer_count;
  vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
  std::vector<VkLayerProperties> available_layer_properties(instance_layer_count);
  vkEnumerateInstanceLayerProperties(&instance_layer_count, available_layer_properties.data());

  std::vector<const char *> request_layer_names;
  std::vector<const char *> request_extension_names;

  VkApplicationInfo application_info {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "Compute Nbody",
    .pEngineName = "Vulkan",
    .apiVersion = VK_API_VERSION_1_3,
  };

  VkInstanceCreateInfo instance_create_info {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &application_info,
    .enabledLayerCount = static_cast<uint32_t>(request_layer_names.size()),
    .ppEnabledLayerNames = request_layer_names.data(),
    .enabledExtensionCount = static_cast<uint32_t>(request_extension_names.size()),
    .ppEnabledExtensionNames = request_extension_names.data(),
  };

  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &context.instance));
}

void ComputeNbody::create_surface() {
  glfwCreateWindowSurface(context.instance, window, nullptr, &context.surface);
}

void ComputeNbody::create_device() {
  uint32_t physical_device_count;
  vkEnumeratePhysicalDevices(context.instance, &physical_device_count, nullptr);
  std::vector<VkPhysicalDevice> gpus(physical_device_count);
  vkEnumeratePhysicalDevices(context.instance, &physical_device_count, gpus.data());

  for (uint32_t physical_device_index = 0; physical_device_index < physical_device_count && context.graphics_queue_family_index < 0; physical_device_index++) {
    context.gpu = gpus[physical_device_index];

    uint32_t queue_family_property_count;
    vkGetPhysicalDeviceQueueFamilyProperties(context.gpu, &queue_family_property_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
    vkGetPhysicalDeviceQueueFamilyProperties(context.gpu, &queue_family_property_count, queue_family_properties.data());

    for (uint32_t queue_family_property_index = 0; queue_family_property_index < queue_family_property_count; queue_family_property_index++) {
      if (queue_family_properties[queue_family_property_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        context.graphics_queue_family_index = queue_family_property_index;
        break;
      }
    }
  }

  uint32_t device_extension_count;
  vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, nullptr);
  std::vector<VkExtensionProperties> available_extension_properties(device_extension_count);
  vkEnumerateDeviceExtensionProperties(context.gpu, nullptr, &device_extension_count, available_extension_properties.data());

  std::vector<const char *> device_extension_names;

  float queue_priority = 1.0f;

  VkDeviceQueueCreateInfo device_queue_create_info {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = context.graphics_queue_family_index,
    .queueCount = 1,
    .pQueuePriorities = &queue_priority,
  };

  VkDeviceCreateInfo device_create_info {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &device_queue_create_info,
    .enabledExtensionCount = static_cast<uint32_t>(device_extension_names.size()),
    .ppEnabledExtensionNames = device_extension_names.data(),
  };

  VK_CHECK(vkCreateDevice(context.gpu, &device_create_info, nullptr, &context.device));
}

void ComputeNbody::create_swapchain() {
  VkSwapchainCreateInfoKHR swapchain_create_info {
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .surface = context.surface,
  };

  VK_CHECK(vkCreateSwapchainKHR(context.device, &swapchain_create_info, nullptr, &context.swapchain));
}

void ComputeNbody::create_queue() {
  VK_CHECK(vkGetDeviceQueue(context.device, context.graphics_queue_family_index, 0, &context.graphics_queue));
  VK_CHECK(vkGetDeviceQueue(context.device, context.compute_queue_family_index, 0, &context.compute_queue));
}

void ComputeNbody::create_descriptor_pool() {
  std::vector<VkDescriptorPoolSize> descriptor_pool_size {
    VkDescriptorPoolSize {
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 2,
    },
    VkDescriptorPoolSize {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
    },
    VkDescriptorPoolSize {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 2,
    },
  };

  VkDescriptorPoolCreateInfo descriptor_pool_create_info {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 2,
    .poolSizeCount = static_cast<uint32_t>(descriptor_pool_size.size()),
    .pPoolSizes = descriptor_pool_size.data(),
  };

  VK_CHECK(vkCreateDescriptorPool(context.device, &descriptor_pool_create_info, nullptr, &context.descriptor_pool));
}

VkDescriptorBufferInfo ComputeNbody::create_descriptor(VkBuffer &buffer, VkDeviceSize size, VkDeviceSize offset) {
  VkDescriptorBufferInfo descriptor_buffer_info {
    .buffer = buffer,
    .offset = offset,
    .range = size,
  };

  return descriptor_buffer_info;
}

bool ComputeNbody::is_depth_stencil_format(VkFormat &format) {
  return format == VK_FORMAT_D16_UNORM_S8_UINT ||
         format == VK_FORMAT_D24_UNORM_S8_UINT ||
         format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

VkDescriptorImageInfo ComputeNbody::create_descriptor(VkImageView &image_view, VkSampler &sampler, VkFormat &format, VkDescriptorType descriptor_type) {
  VkDescriptorImageInfo descriptor_image_info {
    .sampler = sampler,
    .imageView = image_view,
  };

  switch (descriptor_type) {
  case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
  case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
    if (is_depth_stencil_format(format))
      descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    else
      descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    break;
  case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    break;
  default:
    descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    break;
  }

  return descriptor_image_info;
}

void ComputeNbody::create_graphics_descriptor() {
  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings {
    VkDescriptorSetLayoutBinding {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    },
    VkDescriptorSetLayoutBinding {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    },
    VkDescriptorSetLayoutBinding {
      .binding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    },
  };

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size()),
    .pBindings = descriptor_set_layout_bindings.data(),
  };

  VK_CHECK(vkCreateDescriptorSetLayout(context.device, &descriptor_set_layout_create_info, nullptr, &context.graphics_descriptor_set_layout));

  VkDescriptorSetAllocateInfo descriptor_set_allocate_info {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = context.descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &context.graphics_descriptor_set_layout,
  };

  VK_CHECK(vkAllocateDescriptorSets(context.device, &descriptor_set_allocate_info, &context.graphics_descriptor_set));

  VkDescriptorImageInfo particle_image_descriptor = create_descriptor(context.graphics_particle_image, context.grahpics_particle_sampler, context.graphics_particle_format);
  VkDescriptorImageInfo gradient_image_descriptor = create_descriptor(context.graphics_gradient_image, context.graphics_gradient_sampler, context.graphics_gradient_format);
  VkDescriptorBufferInfo buffer_descriptor = create_descriptor(context.graphics_uniform_buffer);

  std::vector<VkWriteDescriptorSet> graphics_write_descriptor_sets {
    VkWriteDescriptorSet {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = context.graphics_descriptor_set,
      .dstBinding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &particle_image_descriptor,
    },
    VkWriteDescriptorSet {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = context.graphics_descriptor_set,
      .dstBinding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &gradient_image_descriptor,
    },
    VkWriteDescriptorSet {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = context.graphics_descriptor_set,
      .dstBinding = 2,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &buffer_descriptor,
    },
  };

  vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(graphics_write_descriptor_sets.size()), graphics_write_descriptor_sets.data(), 0, nullptr);
}

void ComputeNbody::create_compute_descriptor() {
  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings {
    VkDescriptorSetLayoutBinding {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    },
    VkDescriptorSetLayoutBinding {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    },
  };

  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size()),
    .pBindings = descriptor_set_layout_bindings.data(),
  };

  VK_CHECK(vkCreateDescriptorSetLayout(context.device, &descriptor_set_layout_create_info, nullptr, &context.compute_descriptor_set_layout));

  VkDescriptorSetAllocateInfo descriptor_set_allocate_info {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = context.descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts = &context.compute_descriptor_set_layout,
  };

  VK_CHECK(vkAllocateDescriptorSets(context.device, &descriptor_set_allocate_info, &context.compute_descriptor_set));

  VkDescriptorBufferInfo storage_buffer_descriptor = create_descriptor(context.compute_storage_buffer);
  VkDescriptorBufferInfo uniform_buffer_descriptor = create_descriptor(context.compute_uniform_buffer);

  std::vector<VkWriteDescriptorSet> compute_write_descriptor_sets {
    VkWriteDescriptorSet {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = context.compute_descriptor_set,
      .dstBinding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &storage_buffer_descriptor,
    },
    VkWriteDescriptorSet {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = context.compute_descriptor_set,
      .dstBinding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &uniform_buffer_descriptor,
    },
  };

  vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(compute_write_descriptor_sets.size()), compute_write_descriptor_sets.data(), 0, nullptr);
}

void ComputeNbody::create_semaphore() {
  VkSemaphoreCreateInfo semaphore_create_info {
    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
  };

  VK_CHECK(vkCreateSemaphore(context.device, &semaphore_create_info, nullptr, &context.graphics_semaphore));
  VK_CHECK(vkCreateSemaphore(context.device, &semaphore_create_info, nullptr, &context.compute_semaphore));
}

VkPipelineShaderStageCreateInfo ComputeNbody::load_shader(const char *name, const char *path, VkShaderStageFlagBits shader_stage_flag) {
  std::vector<uint32_t> shader_code;
  VkShaderModule shader_module;

  VkShaderModuleCreateInfo vertex_shader_module_create_info {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = shader_code.size(),
    .pCode = shader_code.data(),
  };

  VK_CHECK(vkCreateShaderModule(context.device, &vertex_shader_module_create_info, nullptr, &shader_module));

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = shader_stage_flag,
    .module = shader_module,
    .pName = name,
  };

  return pipeline_shader_stage_create_info;
}

void ComputeNbody::create_graphics_pipeline() {
  VkPipelineLayoutCreateInfo pipeline_layout_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &context.graphics_descriptor_set_layout,
  };

  VK_CHECK(vkCreatePipelineLayout(context.device, &pipeline_layout_create_info, nullptr, &context.graphics_pipeline_layout));

  std::vector<VkVertexInputBindingDescription> vertex_input_binding_descriptors {
    VkVertexInputBindingDescription {
      .binding = 0,
      .stride = sizeof(Particle),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    },
  };

  std::vector<VkVertexInputAttributeDescription> vertex_input_attribute_descriptors {
    VkVertexInputAttributeDescription {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = offsetof(Particle, position),
    },
    VkVertexInputAttributeDescription {
      .location = 1,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = offsetof(Particle, velocity),
    },
  };

  VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = static_cast<uint32_t>(vertex_input_binding_descriptors.size()),
    .pVertexBindingDescriptions = vertex_input_binding_descriptors.data(),
    .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_input_attribute_descriptors.size()),
    .pVertexAttributeDescriptions = vertex_input_attribute_descriptors.data(),
  };

  VkPipelineInputAssemblyStateCreateInfo pipeline_input_assembly_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    .primitiveRestartEnable = VK_FALSE,
  };

  VkViewport viewport {
    .width = context.swapchain_extent.width,
    .height = context.swapchain_extent.height,
    .minDepth = 0.0f,
    .maxDepth = 1.0f,
  };

  VkRect2D scissor {
    .extent = {
      .width = static_cast<uint32_t>(context.swapchain_extent.width),
      .height = static_cast<uint32_t>(context.swapchain_extent.height),
    },
  };

  VkPipelineViewportStateCreateInfo pipeline_viewport_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor,
  };

  VkPipelineRasterizationStateCreateInfo pipeline_rasterization_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .cullMode = VK_CULL_MODE_NONE,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
  };

  VkPipelineMultisampleStateCreateInfo pipeline_multisample_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  VkPipelineDepthStencilStateCreateInfo pipeline_depth_stencil_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    .depthTestEnable = VK_FALSE,
    .depthCompareOp = VK_COMPARE_OP_ALWAYS,
    .stencilTestEnable = VK_FALSE,
  };

  VkPipelineColorBlendAttachmentState pipeline_color_blend_attachment_state {
    .blendEnable = VK_TRUE,
    .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
    .colorBlendOp = VK_BLEND_OP_ADD,
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA,
    .alphaBlendOp = VK_BLEND_OP_ADD,
    .colorWriteMask = 0xF,
  };

  VkPipelineColorBlendStateCreateInfo pipeline_color_blend_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .attachmentCount = 1,
    .pAttachments = &pipeline_color_blend_attachment_state,
  };

  std::vector<VkDynamicState> dynamic_state {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };

  VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .dynamicStateCount = static_cast<uint32_t>(dynamic_state.size()),
    .pDynamicStates = dynamic_state.data(),
  };

  std::array<VkPipelineShaderStageCreateInfo, 2> pipeline_shader_stage_create_info;

  pipeline_shader_stage_create_info[0] = load_shader("compute_nbody", "particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
  pipeline_shader_stage_create_info[1] = load_shader("compute_nbody", "particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

  VkGraphicsPipelineCreateInfo graphics_pipeline_create_info {
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .stageCount = static_cast<uint32_t>(pipeline_shader_stage_create_info.size()),
    .pStages = pipeline_shader_stage_create_info.data(),
    .pVertexInputState = &pipeline_vertex_input_state_create_info,
    .pInputAssemblyState = &pipeline_input_assembly_state_create_info,
    .pViewportState = &pipeline_viewport_state_create_info,
    .pRasterizationState = &pipeline_rasterization_state_create_info,
    .pMultisampleState = &pipeline_multisample_state_create_info,
    .pDepthStencilState = &pipeline_depth_stencil_state_create_info,
    .pColorBlendState = &pipeline_color_blend_state_create_info,
    .pDynamicState = &pipeline_dynamic_state_create_info,
    .renderPass = context.render_pass,
  };

  VK_CHECK(vkCreateGraphicsPipelines(context.device, context.pipeline_cache, 1, &graphics_pipeline_create_info, nullptr, &context.graphics_pipeline));
}

void ComputeNbody::create_compute_pipeline() {
  VkPipelineLayoutCreateInfo pipeline_layout_create_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &context.compute_descriptor_set_layout,
  };

  VK_CHECK(vkCreatePipelineLayout(context.device, &pipeline_layout_create_info, nullptr, &context.compute_pipeline_layout));

  VkPipelineShaderStageCreateInfo pipeline_calculate_shader_stage_create_info = load_shader("compute_nbody", "particle_calculate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

  VkComputePipelineCreateInfo compute_pipeline_create_info {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = pipeline_calculate_shader_stage_create_info,
    .layout = context.compute_pipeline_layout,
  };

  VK_CHECK(vkCreateComputePipelines(context.device, context.pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &context.compute_pipeline_calculate));

  VkPipelineShaderStageCreateInfo pipeline_integrate_shader_stage_create_info = load_shader("compute_nbody", "particle_integrate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

  compute_pipeline_create_info.stage = pipeline_integrate_shader_stage_create_info;

  VK_CHECK(vkCreateComputePipelines(context.device, context.pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &context.compute_pipeline_integrate));
}

void ComputeNbody::create_compute_command() {
  VkCommandPoolCreateInfo command_pool_create_info {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = context.compute_queue_family_index,
  };

  VK_CHECK(vkCreateCommandPool(context.device, &command_pool_create_info, nullptr, &context.compute_command_pool));

  VkCommandBufferAllocateInfo command_buffer_allocate_info {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = context.compute_command_pool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };

  VK_CHECK(vkAllocateCommandBuffers(context.device, &command_buffer_allocate_info, &context.compute_command_buffer));

  VkSubmitInfo submit_info {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &context.compute_semaphore,
  };

  VK_CHECK(vkQueueSubmit(context.compute_queue, 1, &submit_info, VK_NULL_HANDLE));
  VK_CHECK(vkQueueWaitIdle(context.compute_queue));

  if (context.graphics_queue_family_index != context.compute_queue_family_index) {
    VkCommandBuffer transfer_command;

    VkCommandBufferAllocateInfo command_buffer_allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = context.compute_command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };

    VK_CHECK(vkAllocateCommandBuffers(context.device, &command_buffer_allocate_info, &transfer_command));

    VkCommandBufferBeginInfo command_buffer_begin_info {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    VK_CHECK(vkBeginCommandBuffer(transfer_command, &command_buffer_begin_info));

    VkBufferMemoryBarrier acquire_buffer_barrier {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .srcQueueFamilyIndex = context.graphics_queue_family_index,
      .dstQueueFamilyIndex = context.compute_queue_family_index,
      .buffer = context.compute_storage_buffer,
      .offset = 0,
      // .size = context.compute_storage_buffer.getsize(),
    };

    vkCmdPipelineBarrier(transfer_command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &acquire_buffer_barrier, 0, nullptr);

    VkBufferMemoryBarrier release_buffer_barrier {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = 0,
      .srcQueueFamilyIndex = context.compute_queue_family_index,
      .dstQueueFamilyIndex = context.graphics_queue_family_index,
      .buffer = context.compute_storage_buffer,
      .offset = 0,
      // .size = context.compute_storage_buffer.getsize(),
    };

    vkCmdPipelineBarrier(transfer_command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &release_buffer_barrier, 0, nullptr);

    VK_CHECK(vkEndCommandBuffer(transfer_command));

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &transfer_command,
    };

    VkFenceCreateInfo fence_create_info {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };

    VkFence fence;
    VK_CHECK(vkCreateFence(context.device, &fence_create_info, nullptr, &fence));

    VK_CHECK(vkQueueSubmit(context.compute_queue, 1, &submit_info, fence));

    VK_CHECK(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(context.device, fence, nullptr);

    vkFreeCommandBuffers(context.device, context.compute_command_pool, 1, &transfer_command);
  }
}

void ComputeNbody::load_assets() {

}

void ComputeNbody::build_command_buffers() {

}

void ComputeNbody::build_compute_command_buffer() {

}

void ComputeNbody::setup_descriptor_pool() {

}

void ComputeNbody::setup_descriptor_set_layout() {

}

void ComputeNbody::setup_descriptor_set() {

}

void ComputeNbody::prepare_graphics() {

}

void ComputeNbody::prepare_compute() {

}

void ComputeNbody::draw() {

}
