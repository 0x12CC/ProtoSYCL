#pragma once

namespace sycl::detail {

template <typename ReturnType> struct descriptor {
  using return_type = ReturnType;
};

} // namespace sycl::detail
