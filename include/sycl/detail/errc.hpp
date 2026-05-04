#pragma once

namespace sycl {

enum class errc {
  success = 0,
  runtime,
  kernel,
  accessor,
  nd_range,
  event,
  kernel_argument,
  build,
  invalid,
  memory_allocation,
  platform,
  profiling,
  feature_not_supported,
  kernel_not_supported,
  backend_mismatch
};

namespace detail {
[[noreturn]] void throw_sycl_exception(errc, const char * = "");
} // namespace detail

} // namespace sycl
