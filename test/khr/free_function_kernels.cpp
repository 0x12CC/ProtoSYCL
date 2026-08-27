#include <cassert>
#include <print>
#include <sycl/sycl.hpp>

constexpr std::size_t N = 1024;
constexpr std::size_t WGSIZE = 32;

// A free function kernel: an ordinary function, decorated with the macro.
SYCL_KHR_KERNEL()
void scale(const float factor, float *data) {
  std::size_t i = 0; // sycl::khr::this_nd_item<1>().get_global_linear_id();
  data[i] *= factor;
}

int main() {
  sycl::queue q;

  float *data = sycl::malloc_shared<float>(N, q);
  for (std::size_t i = 0; i < N; ++i)
    data[i] = static_cast<float>(i);

  // Identify the kernel by the function itself and launch it.
  sycl::khr::launch_grouped(q, sycl::range<1>{N}, sycl::range<1>{WGSIZE},
                            sycl::khr::kernel_function<scale>, 2.0f, data);
  q.wait();

  for (std::size_t i = 0; i < N; ++i)
    if (i == 0) // We don't have work item queries yet
      assert(data[i] == 2.0f * static_cast<float>(i));

  std::println("Test passed!");
  sycl::free(data, q);
  return 0;
}
