#include <cassert>
#include <print>
#include <sycl/sycl.hpp>

void run_lambda_kernel(sycl::queue q, sycl::nd_range<1> range) {
  q.parallel_for<struct LambdaKernel>(
      range, [](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(4)]] {});
  q.wait();
}

void run_lambda_template_kernel(sycl::queue q, sycl::nd_range<1> range) {
  q.parallel_for<struct LambdaTemplateKernel>(
      range, [](auto it) [[sycl::reqd_work_group_size(4)]] {});
  q.wait();
}

int main() {
  sycl::queue q;

  // Check that the application has two kernel IDs.
  const auto ids = sycl::get_kernel_ids();
  assert(ids.size() == 2 && "Expected exactly 2 kernel IDs");

  // Check that both kernels are compatible with the current device.
  for (const auto &id : ids) {
    const bool compatible = sycl::is_compatible({id}, q.get_device());
    assert(compatible &&
           "Expected kernel to be compatible with current device");
  }

  // Check that a kernel bundle includes both kernels.
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  assert(bundle.get_kernel_ids().size() == 2 &&
         "Expected kernel bundle to contain exactly 2 kernel IDs");

  // Lambda with non-template call operator, right work group size.
  run_lambda_kernel(q, {8, 4});

  // Lambda with non-template call operator, wrong work group size.
  try {
    run_lambda_kernel(q, {8, 8});
    assert(false && "Expected an exception when launching LambdaKernel");
  } catch (...) {
  }

  // Lambda with template call operator, right work group size.
  run_lambda_template_kernel(q, {8, 4});

  // Lambda with template call operator, wrong work group size.
  try {
    run_lambda_template_kernel(q, {8, 8});
    assert(false &&
           "Expected an exception when launching LambdaTemplateKernel");
  } catch (...) {
  }

  std::println("Attribute test passed!");
  return 0;
}
