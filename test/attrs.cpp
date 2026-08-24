#include <cassert>
#include <print>
#include <sycl/sycl.hpp>

#if defined(PROTOSYCL_REFLECTION)
// GCC/reflection path: the property is expressed as a P3394 annotation on a
// named functor and read from the library via C++26 reflection (no compiler
// plugin). For a non-template call operator the annotation may sit on
// operator() directly; for a template call operator g++ 16.1 cannot read
// annotations off the template, so it sits on the functor type.
namespace {
using sycl::detail::kernel_attributes::reqd_work_group_size_1d;

struct LambdaKernel {
  [[=reqd_work_group_size_1d{4}]] void operator()(sycl::nd_item<1>) const {}
};

struct [[=reqd_work_group_size_1d{4}]] LambdaTemplateKernel {
  template <typename ItemT> void operator()(ItemT) const {}
};
} // namespace

void run_lambda_kernel(sycl::queue q, sycl::nd_range<1> range) {
  q.parallel_for<LambdaKernel>(range, LambdaKernel{});
  q.wait();
}

void run_lambda_template_kernel(sycl::queue q, sycl::nd_range<1> range) {
  q.parallel_for<LambdaTemplateKernel>(range, LambdaTemplateKernel{});
  q.wait();
}
#else
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
#endif

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
