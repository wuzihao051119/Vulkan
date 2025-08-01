#include "descriptor_set_layout.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
  DescriptorSetLayout app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
