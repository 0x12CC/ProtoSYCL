#pragma once

#include <typeindex>

namespace sycl {

class kernel_id {
public:
  const char *get_name() const noexcept { return m_info.name(); }

  friend bool operator==(const kernel_id &, const kernel_id &) = default;

private:
  template <typename> friend kernel_id get_kernel_id();
  friend std::hash<kernel_id>;

  kernel_id(const std::type_info &info) : m_info{info} {}

  std::type_index m_info;
};

template <typename KernelName> kernel_id get_kernel_id();

} // namespace sycl

template <> struct std::hash<sycl::kernel_id> {
  std::size_t operator()(const sycl::kernel_id &id) const noexcept {
    return std::hash<std::type_index>{}(id.m_info);
  }
};
