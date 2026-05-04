#include "sycl/detail/exception.hpp"

namespace sycl {

std::error_code make_error_code(errc e) noexcept {
  return {static_cast<int>(e), detail::error_category_v};
}

const std::error_category &sycl_category() noexcept {
  return detail::error_category_v;
}

exception_list
detail::create_exception_list(std::vector<std::exception_ptr> exceptions) {
  exception_list list;
  list.m_elements = std::move(exceptions);
  return list;
}

[[noreturn]] void detail::throw_sycl_exception(errc code, const char *message) {
  throw exception(code, message);
}

} // namespace sycl
