#include "sycl/detail/context.hpp"

namespace sycl::detail {

context get_default_context() {
  static context default_context{};
  return default_context;
}

} // namespace sycl::detail
