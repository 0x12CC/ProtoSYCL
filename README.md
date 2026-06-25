# ProtoSYCL

ProtoSYCL is a sample compiler implementation of SYCL 2020. It is primarily intended for developing SYCL test cases. In its current state, it can pass 68 of the 72 test categories in the [CTS](https://github.com/KhronosGroup/SYCL-CTS).

> [!WARNING]
> This SYCL implementation is a research project and is **NOT** intended for use in a production environment.

## Requirements

ProtoSYCL requires the following dependencies:

* **CMake**: Version 3.20 or higher.
* **LLVM/Clang**: Tested with pre-built version 22.1.4.

ProtoSYCL runs on Linux and macOS. Windows support is not implemented.

## Getting Started

ProtoSYCL is designed to be integrated into C++ projects using CMake's `FetchContent` module. This makes it easy to include ProtoSYCL as a dependency in your project.

### CMake Integration

The following configuration fetches ProtoSYCL, sets the project's C++ compiler to the driver (`sycl++`), and links the necessary runtime libraries:

```cmake
# Create a SYCL project.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(sycl_app VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch ProtoSYCL.
include(FetchContent)
FetchContent_Declare(
    ProtoSYCL
    GIT_REPOSITORY https://github.com/0x12CC/ProtoSYCL.git
    GIT_TAG main
)
FetchContent_MakeAvailable(ProtoSYCL)

# Set the C++ compiler to sycl++.
FetchContent_GetProperties(ProtoSYCL BINARY_DIR PROTOSYCL_BINARY_DIR)
set(CMAKE_CXX_COMPILER "${PROTOSYCL_BINARY_DIR}/sycl++")

# Create an example SYCL application using ProtoSYCL.
add_executable(sycl_app src/main.cpp)
target_link_libraries(sycl_app PRIVATE ProtoSYCL)
```

### Writing a Kernel

ProtoSYCL supports most SYCL 2020 features, including Unified Shared Memory (USM). The following example demonstrates a basic parallel vector assignment submitted to a SYCL queue:


```c++
#include <cassert>
#include <cstddef>
#include <print>
#include <sycl/sycl.hpp>

int main() {
  // Create a SYCL queue to submit work to the device.
  sycl::queue q;

  // Allocate shared memory for use in the kernel.
  constexpr std::size_t size = 1024;
  int *const data = sycl::malloc_shared<int>(size, q);
  assert(data != nullptr);

  // Write to the pointer in parallel using a SYCL kernel.
  q.parallel_for(size, [=](const sycl::item<1> it) {
    const std::size_t idx = it.get_linear_id();
    data[idx] = idx;
  });
  q.wait();

  // Check the results.
  for (std::size_t i = 0; i < size; i++)
    assert(data[i] == i);

  // Print a success message and free the allocated memory.
  std::println("SYCL parallel_for executed successfully!");
  sycl::free(data, q);
}
```

### Compilation and Execution

Building is handled via the standard CMake workflow:

```
mkdir -p build
cmake -G Ninja -B build
cmake --build build
./build/sycl_app
```

## License

ProtoSYCL is licensed under the MIT License. See the `LICENSE` file for details.
