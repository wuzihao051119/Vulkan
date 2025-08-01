set(PROJECT_NAME DescriptorSetLayout)
set(CURRENT_INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin/${PROJECT_NAME})

project(${PROJECT_NAME} LANGUAGES CXX)

add_executable(${PROJECT_NAME}
  descriptor_set_layout.cpp descriptor_set_layout.hpp
  main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  Vulkan::Vulkan
  glfw
)

install(TARGETS DescriptorSetLayout
  RUNTIME DESTINATION ${CURRENT_INSTALL_DIR}
)

find_program(GLSLANG_VALIDATOR glslangValidator)

set(SHADER_FILES shader.vert shader.frag)

foreach(SHADER ${SHADER_FILES})
  get_filename_component(FILENAME ${SHADER} NAME)
  set(SPIRV_OUTPUT ${FILENAME}.spv)

  add_custom_command(
    OUTPUT ${SPIRV_OUTPUT}
    COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPIRV_OUTPUT}
    DEPENDS ${SHADER}
  )
  list(APPEND SPIRV_FILES ${CMAKE_CURRENT_BINARY_DIR}/${SPIRV_OUTPUT})
endforeach()

add_custom_target(${PROJECT_NAME}Shaders ALL DEPENDS ${SPIRV_FILES})

install(FILES ${SPIRV_FILES}
  DESTINATION ${CURRENT_INSTALL_DIR}
)
