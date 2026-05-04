#pragma once

#include <functional>
#include <string>
#include <vector>

#include "aspect.hpp"
#include "backend.hpp"
#include "errc.hpp"
#include "kernel_id.hpp"
#include "memory_order.hpp"
#include "memory_scope.hpp"
#include "range.hpp"
#include "util.descriptor.hpp"

namespace sycl {

class device;
class platform;

namespace info {

enum class device_type {
  cpu,         // Maps to OpenCL CL_DEVICE_TYPE_CPU
  gpu,         // Maps to OpenCL CL_DEVICE_TYPE_GPU
  accelerator, // Maps to OpenCL CL_DEVICE_TYPE_ACCELERATOR
  custom,      // Maps to OpenCL CL_DEVICE_TYPE_CUSTOM
  automatic,   // Maps to OpenCL CL_DEVICE_TYPE_DEFAULT
  host,
  all // Maps to OpenCL CL_DEVICE_TYPE_ALL
};

enum class partition_property {
  no_partition,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

enum class partition_affinity_domain {
  not_applicable,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

enum class local_mem_type { none, local, global };

enum class fp_config {
  denorm,
  inf_nan,
  round_to_nearest,
  round_to_zero,
  round_to_inf,
  fma,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type { none, read_only, read_write };

enum class execution_capability { exec_kernel, exec_native_kernel };

namespace device {

struct device_type : detail::descriptor<info::device_type> {};
struct vendor_id : detail::descriptor<std::uint32_t> {};
struct max_compute_units : detail::descriptor<std::uint32_t> {};
struct max_work_item_dimensions : detail::descriptor<std::uint32_t> {};
template <int Dimensions = 3>
struct max_work_item_sizes : detail::descriptor<range<Dimensions>> {};
struct max_work_group_size : detail::descriptor<std::size_t> {};
struct max_num_sub_groups : detail::descriptor<std::uint32_t> {};
struct sub_group_sizes : detail::descriptor<std::vector<std::size_t>> {};
struct preferred_vector_width_char : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_short : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_int : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_long : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_float : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_double : detail::descriptor<std::uint32_t> {};
struct preferred_vector_width_half : detail::descriptor<std::uint32_t> {};
struct native_vector_width_char : detail::descriptor<std::uint32_t> {};
struct native_vector_width_short : detail::descriptor<std::uint32_t> {};
struct native_vector_width_int : detail::descriptor<std::uint32_t> {};
struct native_vector_width_long : detail::descriptor<std::uint32_t> {};
struct native_vector_width_float : detail::descriptor<std::uint32_t> {};
struct native_vector_width_double : detail::descriptor<std::uint32_t> {};
struct native_vector_width_half : detail::descriptor<std::uint32_t> {};
struct max_clock_frequency : detail::descriptor<std::uint32_t> {};
struct address_bits : detail::descriptor<std::uint32_t> {};
struct max_mem_alloc_size : detail::descriptor<std::uint64_t> {};
// Deprecated
struct image_support : detail::descriptor<bool> {};
struct max_read_image_args : detail::descriptor<std::uint32_t> {};
struct max_write_image_args : detail::descriptor<std::uint32_t> {};
struct image2d_max_height : detail::descriptor<std::size_t> {};
struct image2d_max_width : detail::descriptor<std::size_t> {};
struct image3d_max_height : detail::descriptor<std::size_t> {};
struct image3d_max_width : detail::descriptor<std::size_t> {};
struct image3d_max_depth : detail::descriptor<std::size_t> {};
struct image_max_buffer_size : detail::descriptor<std::size_t> {};
struct max_samplers : detail::descriptor<std::uint32_t> {};
struct max_parameter_size : detail::descriptor<std::size_t> {};
struct mem_base_addr_align : detail::descriptor<std::uint32_t> {};
struct half_fp_config : detail::descriptor<std::vector<fp_config>> {};
struct single_fp_config : detail::descriptor<std::vector<fp_config>> {};
struct double_fp_config : detail::descriptor<std::vector<fp_config>> {};
struct global_mem_cache_type : detail::descriptor<info::global_mem_cache_type> {
};
struct global_mem_cache_line_size : detail::descriptor<std::uint32_t> {};
struct global_mem_cache_size : detail::descriptor<std::uint64_t> {};
struct global_mem_size : detail::descriptor<std::uint64_t> {};
// Deprecated
struct max_constant_buffer_size : detail::descriptor<std::uint64_t> {};
// Deprecated
struct max_constant_args : detail::descriptor<std::uint32_t> {};
struct local_mem_type : detail::descriptor<info::local_mem_type> {};
struct local_mem_size : detail::descriptor<std::uint64_t> {};
struct error_correction_support : detail::descriptor<bool> {};
struct host_unified_memory : detail::descriptor<bool> {};
struct atomic_memory_order_capabilities
    : detail::descriptor<std::vector<memory_order>> {};
struct atomic_fence_order_capabilities
    : detail::descriptor<std::vector<memory_order>> {};
struct atomic_memory_scope_capabilities
    : detail::descriptor<std::vector<memory_scope>> {};
struct atomic_fence_scope_capabilities
    : detail::descriptor<std::vector<memory_scope>> {};
struct profiling_timer_resolution : detail::descriptor<std::size_t> {};
// Deprecated
struct is_endian_little : detail::descriptor<bool> {};
struct is_available : detail::descriptor<bool> {};
// Deprecated
struct is_compiler_available : detail::descriptor<bool> {};
// Deprecated
struct is_linker_available : detail::descriptor<bool> {};
struct execution_capabilities
    : detail::descriptor<std::vector<execution_capability>> {};
// Deprecated
struct queue_profiling : detail::descriptor<bool> {};
// Deprecated
struct built_in_kernels : detail::descriptor<std::vector<std::string>> {};
struct built_in_kernel_ids : detail::descriptor<std::vector<kernel_id>> {};
struct platform : detail::descriptor<platform> {};
struct name : detail::descriptor<std::string> {};
struct vendor : detail::descriptor<std::string> {};
struct driver_version : detail::descriptor<std::string> {};
struct profile : detail::descriptor<std::string> {};
struct version : detail::descriptor<std::string> {};
struct backend_version : detail::descriptor<std::string> {};
struct aspects : detail::descriptor<std::vector<aspect>> {};
// Deprecated
struct extensions : detail::descriptor<std::vector<std::string>> {};
struct printf_buffer_size : detail::descriptor<std::size_t> {};
struct preferred_interop_user_sync : detail::descriptor<bool> {};
struct parent_device : detail::descriptor<sycl::device> {};
struct partition_max_sub_devices : detail::descriptor<std::uint32_t> {};
struct partition_properties
    : detail::descriptor<std::vector<partition_property>> {};
struct partition_affinity_domains
    : detail::descriptor<std::vector<partition_affinity_domain>> {};
struct partition_type_property : detail::descriptor<partition_property> {};
struct partition_type_affinity_domain
    : detail::descriptor<partition_affinity_domain> {};

} // namespace device

} // namespace info

namespace detail {

using device_selector = std::function<int(const device &)>;

template <typename Fn>
concept DeviceSelector = std::is_invocable_r_v<int, Fn, const device &>;

} // namespace detail

class device {
public:
  device() = default;

  template <detail::DeviceSelector DeviceSelector>
  explicit device(const DeviceSelector &deviceSelector) {
    auto result = deviceSelector(*this);
    if (result < 0)
      sycl::detail::throw_sycl_exception(
          errc::runtime,
          "No device found matching the device selector criteria");
  }

  /* -- common interface members -- */

  device(const device &rhs) = default;
  device(device &&rhs) = default;
  device &operator=(const device &rhs) = default;
  device &operator=(device &&rhs) = default;
  ~device() = default;

  friend bool operator==(const device &lhs, const device &rhs) { return true; }

  friend bool operator!=(const device &lhs, const device &rhs) {
    return !(lhs == rhs);
  }

  backend get_backend() const noexcept { return backend::proto_sycl; }

  bool is_cpu() const { return true; }

  bool is_gpu() const { return false; }

  bool is_accelerator() const { return false; }

  platform get_platform() const;

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

  bool has(aspect asp) const {
    if (asp == aspect::online_compiler)
      return false;
    else if (asp == aspect::online_linker)
      return false;
    return true;
  }

  // Deprecated
  bool has_extension(const std::string &extension) const {
    std::ignore = extension;
    return false;
  }

  template <info::partition_property Prop>
  std::vector<device> create_sub_devices(std::size_t count) const
    requires(Prop == info::partition_property::partition_equally)
  {
    sycl::detail::throw_sycl_exception(errc::feature_not_supported);
  }

  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(const std::vector<std::size_t> &counts) const
    requires(Prop == info::partition_property::partition_by_counts)
  {
    sycl::detail::throw_sycl_exception(errc::feature_not_supported);
  }

  template <info::partition_property Prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const
    requires(Prop == info::partition_property::partition_by_affinity_domain)
  {
    sycl::detail::throw_sycl_exception(errc::feature_not_supported);
  }

  static std::vector<device>
  get_devices(info::device_type deviceType = info::device_type::all) {
    if (deviceType == info::device_type::all ||
        deviceType == info::device_type::cpu)
      return {device{}};
    return {};
  }
};

// Predefined device selectors
inline detail::device_selector default_selector_v = [](const device &) -> int {
  return 0;
};
inline detail::device_selector cpu_selector_v = [](const device &) -> int {
  return 0;
};
inline detail::device_selector gpu_selector_v = [](const device &) -> int {
  return -1;
};
inline detail::device_selector accelerator_selector_v =
    [](const device &) -> int { return -1; };

// Predefined types for compatibility with old SYCL 1.2.1 device selectors
// Deprecated in SYCL 2020
struct default_selector {
  int operator()(const device &) const { return 0; }
};
struct cpu_selector {
  int operator()(const device &) const { return 0; }
};
struct gpu_selector {
  int operator()(const device &) const { return -1; }
};
struct accelerator_selector {
  int operator()(const device &) const { return -1; }
};

// Returns a selector that selects a device based on desired aspects
inline detail::device_selector
aspect_selector(const std::vector<aspect> &aspectList,
                const std::vector<aspect> &denyList = {}) {
  return [aspectList, denyList](const device &dev) -> int {
    for (const auto &deny_aspect : denyList) {
      if (!dev.has(deny_aspect))
        continue;
      return -1;
    }
    for (const auto &req_aspect : aspectList) {
      if (!dev.has(req_aspect))
        return -1;
    }
    return 0;
  };
}
template <typename... AspectList>
detail::device_selector aspect_selector(AspectList... aspectList) {
  return aspect_selector(std::vector<aspect>{aspectList...});
}
template <aspect... AspectList> detail::device_selector aspect_selector() {
  return aspect_selector(std::vector<aspect>{AspectList...});
}

template <aspect Aspect> struct any_device_has : std::true_type {};
template <aspect Aspect> struct all_devices_have : std::true_type {};

template <> struct any_device_has<aspect::online_compiler> : std::false_type {};
template <> struct any_device_has<aspect::online_linker> : std::false_type {};

template <>
struct all_devices_have<aspect::online_compiler> : std::false_type {};
template <> struct all_devices_have<aspect::online_linker> : std::false_type {};

template <aspect A>
inline constexpr bool any_device_has_v = any_device_has<A>::value;
template <aspect A>
inline constexpr bool all_devices_have_v = all_devices_have<A>::value;

template <>
inline typename info::device::device_type::return_type
device::get_info<info::device::device_type>() const {
  return info::device_type::cpu;
}

template <>
inline typename info::device::max_compute_units::return_type
device::get_info<info::device::max_compute_units>() const {
  return 1;
}

template <>
inline typename info::device::sub_group_sizes::return_type
device::get_info<info::device::sub_group_sizes>() const {
  return {32};
}

template <>
inline typename info::device::is_available::return_type
device::get_info<info::device::is_available>() const {
  return true;
}

template <>
typename info::device::is_compiler_available::return_type inline device::
    get_info<info::device::is_compiler_available>() const {
  return false;
}

template <>
inline typename info::device::is_linker_available::return_type
device::get_info<info::device::is_linker_available>() const {
  return false;
}

template <>
inline typename info::device::built_in_kernels::return_type
device::get_info<info::device::built_in_kernels>() const {
  return {};
}

template <>
inline typename info::device::built_in_kernel_ids::return_type
device::get_info<info::device::built_in_kernel_ids>() const {
  return {};
}

template <>
inline typename info::device::name::return_type
device::get_info<info::device::name>() const {
  return "<sycl device>";
}

template <>
inline typename info::device::vendor::return_type
device::get_info<info::device::vendor>() const {
  return "<sycl device vendor>";
}

template <>
inline typename info::device::version::return_type
device::get_info<info::device::version>() const {
  return "v0.1.0";
}

template <>
inline typename info::device::max_work_item_sizes<1>::return_type
device::get_info<info::device::max_work_item_sizes<1>>() const {
  return range{16};
}

template <>
inline typename info::device::max_work_item_sizes<2>::return_type
device::get_info<info::device::max_work_item_sizes<2>>() const {
  return range{4, 4};
}

template <>
inline typename info::device::max_work_item_sizes<3>::return_type
device::get_info<info::device::max_work_item_sizes<3>>() const {
  return range{2, 4, 4};
}

template <>
inline typename info::device::max_work_group_size::return_type
device::get_info<info::device::max_work_group_size>() const {
  return 8;
}

template <>
inline typename info::device::max_num_sub_groups::return_type
device::get_info<info::device::max_num_sub_groups>() const {
  return 1;
}

template <>
inline typename info::device::mem_base_addr_align::return_type
device::get_info<info::device::mem_base_addr_align>() const {
  return sizeof(double);
}

template <>
inline typename info::device::atomic_memory_order_capabilities::return_type
device::get_info<info::device::atomic_memory_order_capabilities>() const {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::device::atomic_fence_order_capabilities::return_type
device::get_info<info::device::atomic_fence_order_capabilities>() const {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::device::atomic_memory_scope_capabilities::return_type
device::get_info<info::device::atomic_memory_scope_capabilities>() const {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

template <>
inline typename info::device::atomic_fence_scope_capabilities::return_type
device::get_info<info::device::atomic_fence_scope_capabilities>() const {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

template <>
inline typename info::device::max_mem_alloc_size::return_type
device::get_info<info::device::max_mem_alloc_size>() const {
  return 8192;
}

} // namespace sycl

template <> struct std::hash<sycl::device> {
  std::size_t operator()(const sycl::device &d) const noexcept {
    return std::hash<std::string>{}(d.get_info<sycl::info::device::name>());
  }
};
