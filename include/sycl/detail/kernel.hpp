#pragma once

#include <limits>

#include "bundle_state.hpp"
#include "backend.hpp"
#include "context.hpp"
#include "device.hpp"
#include "exception.hpp"
#include "range.hpp"
#include "util.descriptor.hpp"

namespace sycl {

namespace info {

namespace kernel {
struct num_args : detail::descriptor<std::uint32_t> {};
struct attributes : detail::descriptor<std::string> {};
} // namespace kernel

namespace kernel_device_specific {
struct global_work_size : detail::descriptor<range<3>> {};
struct work_group_size : detail::descriptor<std::size_t> {};
struct compile_work_group_size : detail::descriptor<range<3>> {};
struct preferred_work_group_size_multiple : detail::descriptor<std::size_t> {};
struct private_mem_size : detail::descriptor<std::size_t> {};
struct max_num_sub_groups : detail::descriptor<std::uint32_t> {};
struct compile_num_sub_groups : detail::descriptor<std::uint32_t> {};
struct max_sub_group_size : detail::descriptor<std::uint32_t> {};
struct compile_sub_group_size : detail::descriptor<std::uint32_t> {};
} // namespace kernel_device_specific

} // namespace info

template <bundle_state State> class kernel_bundle;

template <bundle_state State>
kernel_bundle<State> get_kernel_bundle(const context &);

class kernel {
public:
  friend bool operator==(const kernel &, const kernel &) = default;

  backend get_backend() const noexcept { return get_context().get_backend(); }

  context get_context() const { return m_ctx; }

  kernel_bundle<bundle_state::executable> get_kernel_bundle() const;

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_info(const device &dev) const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  template <bundle_state State> friend class kernel_bundle;
  friend std::hash<kernel>;

  kernel(kernel_id id, context ctx) : m_id{id}, m_ctx{ctx} {}

  kernel_id m_id;
  context m_ctx;
};

template <>
inline typename info::kernel::num_args::return_type
kernel::get_info<info::kernel::num_args>() const {
  throw sycl::exception{errc::invalid};
}

template <>
inline typename info::kernel::attributes::return_type
kernel::get_info<info::kernel::attributes>() const {
  return "";
}

template <>
inline typename info::kernel_device_specific::global_work_size::return_type
kernel::get_info<info::kernel_device_specific::global_work_size>(
    const device &dev) const {
  throw sycl::exception{errc::invalid};
}

template <>
inline typename info::kernel_device_specific::work_group_size::return_type
kernel::get_info<info::kernel_device_specific::work_group_size>(
    const device &dev) const {
  return dev.get_info<info::device::max_work_group_size>();
}

template <>
inline typename info::kernel_device_specific::compile_work_group_size::return_type
kernel::get_info<info::kernel_device_specific::compile_work_group_size>(
    const device &dev) const {
  return {0, 0, 0};
}

template <>
inline typename info::kernel_device_specific::preferred_work_group_size_multiple::
    return_type
    kernel::get_info<
        info::kernel_device_specific::preferred_work_group_size_multiple>(
        const device &dev) const {
  return 1;
}

template <>
inline typename info::kernel_device_specific::private_mem_size::return_type
kernel::get_info<info::kernel_device_specific::private_mem_size>(
    const device &dev) const {
  return std::numeric_limits<std::size_t>::max();
}

template <>
inline typename info::kernel_device_specific::max_num_sub_groups::return_type
kernel::get_info<info::kernel_device_specific::max_num_sub_groups>(
    const device &dev) const {
  return 1;
}

template <>
inline typename info::kernel_device_specific::compile_num_sub_groups::return_type
kernel::get_info<info::kernel_device_specific::compile_num_sub_groups>(
    const device &dev) const {
  return 0;
}

template <>
inline typename info::kernel_device_specific::max_sub_group_size::return_type
kernel::get_info<info::kernel_device_specific::max_sub_group_size>(
    const device &dev) const {
  return 1;
}

template <>
inline typename info::kernel_device_specific::compile_sub_group_size::return_type
kernel::get_info<info::kernel_device_specific::compile_sub_group_size>(
    const device &dev) const {
  return 0;
}

} // namespace sycl

template <> struct std::hash<sycl::kernel> {
  std::size_t operator()(const sycl::kernel &kernel) const noexcept {
    return std::hash<sycl::kernel_id>{}(kernel.m_id);
  }
};
