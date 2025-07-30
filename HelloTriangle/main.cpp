#include "hello_triangle.hpp"

int main(int argc, char *argv[]) {
  std::unique_ptr<HelloTriangle> application = create_hello_triangle();
  application->create_window();
  application->prepare();
  application->update();

  while (!glfwWindowShouldClose(application->window)) {
    glfwPollEvents();
  }

  return 0;
}
