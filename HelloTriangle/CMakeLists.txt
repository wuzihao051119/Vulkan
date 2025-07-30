set(PROJECT_NAME HelloTriangle)
set(CURRENT_INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin/${PROJECT_NAME})

project(${PROJECT_NAME} LANGUAGES CXX)

add_executable(HelloTriangle
  hello_triangle.cpp hello_triangle.hpp
  main.cpp
)

target_include_directories(HelloTriangle PRIVATE
  ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(HelloTriangle PRIVATE
  Vulkan::Vulkan
  spdlog::spdlog
  glfw
)

install(TARGETS HelloTriangle
  RUNTIME DESTINATION ${CURRENT_INSTALL_DIR}
)

find_program(GLSLANG_VALIDATOR glslangValidator)

set(SHADER_FILES triangle.vert triangle.frag)

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

add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})

install(FILES ${SPIRV_FILES}
  DESTINATION ${CURRENT_INSTALL_DIR}
)
