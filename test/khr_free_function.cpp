#include <cassert>
#include <print>
#include <sycl/sycl.hpp>

// Free-function kernels declared with the prototype SYCL_KHR_KERNEL form. The
// khr compile-time property is attached as a P3394 annotation and read back via
// reflection; each macro also emits a static registrar so the kernel shows up
// in get_kernel_ids().
SYCL_KHR_KERNEL(wgs1_kernel, (sycl::khr::property::reqd_work_group_size<4>),
                sycl::nd_item<1> it) {
  (void)it;
}

SYCL_KHR_KERNEL(wgs2_kernel, (sycl::khr::property::reqd_work_group_size<8, 4>),
                sycl::nd_item<2> it) {
  (void)it;
}

SYCL_KHR_KERNEL(sg_kernel, (sycl::khr::property::reqd_sub_group_size<16>),
                sycl::nd_item<1> it) {
  (void)it;
}

SYCL_KHR_KERNEL(hint_kernel, (sycl::khr::property::work_group_size_hint<32, 2>),
                sycl::nd_item<2> it) {
  (void)it;
}

// Compile-time vs runtime property classification (per the KHR spec taxonomy).
static_assert(
    sycl::khr::is_property_v<sycl::khr::property::reqd_work_group_size_t<4>>);
static_assert(
    sycl::khr::is_property_v<sycl::khr::property::reqd_sub_group_size_t<16>>);
static_assert(
    sycl::khr::is_property_v<sycl::khr::property::work_group_size_hint_t<32>>);
static_assert(sycl::khr::is_property_key_v<
              sycl::khr::property::key::reqd_sub_group_size>);
static_assert(sycl::khr::is_property_key_compile_time_v<
              sycl::khr::property::key::reqd_sub_group_size>);
static_assert(sycl::khr::is_property_key_compile_time_v<
              sycl::khr::property::key::work_group_size_hint>);

int main() {
  using namespace sycl::detail::kernel_attributes;

  // All four free-function kernels are enumerable.
  const auto ids = sycl::get_kernel_ids();
  assert(ids.size() == 4 && "Expected exactly 4 free-function kernel IDs");

  // wgs1_kernel: 1D required work-group size of 4.
  {
    const auto id = sycl::get_kernel_id<wgs1_kernel_KernelName>();
    const auto attributes = sycl::detail::get_kernel_attributes(id);
    assert(attributes.size() == 1);
    const auto &wgs = std::get<reqd_work_group_size_1d>(attributes[0]);
    assert(wgs.x == 4);
  }

  // wgs2_kernel: 2D required work-group size of {8, 4}.
  {
    const auto id = sycl::get_kernel_id<wgs2_kernel_KernelName>();
    const auto attributes = sycl::detail::get_kernel_attributes(id);
    assert(attributes.size() == 1);
    const auto &wgs = std::get<reqd_work_group_size_2d>(attributes[0]);
    assert(wgs.x == 8 && wgs.y == 4);
  }

  // sg_kernel: required sub-group size of 16.
  {
    const auto id = sycl::get_kernel_id<sg_kernel_KernelName>();
    const auto attributes = sycl::detail::get_kernel_attributes(id);
    assert(attributes.size() == 1);
    const auto &sg = std::get<reqd_sub_group_size>(attributes[0]);
    assert(sg.size == 16);
  }

  // hint_kernel: work-group size hint {32, 2} (stored, not enforced).
  {
    const auto id = sycl::get_kernel_id<hint_kernel_KernelName>();
    const auto attributes = sycl::detail::get_kernel_attributes(id);
    assert(attributes.size() == 1);
    const auto &hint = std::get<work_group_size_hint>(attributes[0]);
    assert(hint.rank == 2 && hint.x == 32 && hint.y == 2);
  }

  std::println("KHR free-function kernel test passed!");
  return 0;
}
