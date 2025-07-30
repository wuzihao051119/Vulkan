cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER g++)

project(Vulkan LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(spdlog REQUIRED)
find_package(glfw3 REQUIRED)

add_subdirectory(HelloTriangle)
