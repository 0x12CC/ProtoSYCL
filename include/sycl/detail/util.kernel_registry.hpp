#pragma once

#include <cstdint>
#include <unordered_map>
#include <variant>
#include <vector>

#include "aspect.hpp"
#include "kernel_id.hpp"

namespace sycl::detail {

namespace kernel_attributes {

struct reqd_work_group_size_1d {
  std::size_t x;
};

struct reqd_work_group_size_2d {
  std::size_t x;
  std::size_t y;
};

struct reqd_work_group_size_3d {
  std::size_t x;
  std::size_t y;
  std::size_t z;
};

struct reqd_sub_group_size {
  std::uint32_t size;
};

struct device_has {
  sycl::aspect feature;
};

using attribute =
    std::variant<reqd_work_group_size_1d, reqd_work_group_size_2d,
                 reqd_work_group_size_3d, reqd_sub_group_size, device_has>;

} // namespace kernel_attributes

using kernel_attribute_list = std::vector<kernel_attributes::attribute>;

std::unordered_map<kernel_id, std::type_index> &get_kernel_registry();

kernel_attribute_list get_kernel_attributes(const kernel_id &id);

template <typename KernelName, typename KernelFunction>
kernel_id register_kernel() {
  const kernel_id id = get_kernel_id<KernelName>();
  get_kernel_registry().insert({id, typeid(KernelFunction)});
  return id;
}

extern "C" void sycl_register_attribute(const char *name, const char *attribute,
                                        int argCount, ...);

template <typename KernelName, typename KernelFunction>
[[clang::annotate("sycl_kernel_registration")]] inline const sycl::kernel_id
    kernel_registry_v = register_kernel<KernelName, KernelFunction>();

} // namespace sycl::detail
