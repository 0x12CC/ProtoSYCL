#pragma once

#include <functional>
#include <optional>

#include "device.hpp"
#include "memory_order.hpp"
#include "memory_scope.hpp"
#include "platform.hpp"
#include "property_list.hpp"
#include "util.descriptor.hpp"

namespace sycl {

class context;
class exception_list;

namespace detail {

struct queue_impl;

context get_default_context();

} // namespace detail

using async_handler = std::function<void(exception_list)>;

namespace info::context {

struct platform : detail::descriptor<sycl::platform> {};
struct devices : detail::descriptor<std::vector<sycl::device>> {};
struct atomic_memory_order_capabilities
    : detail::descriptor<std::vector<memory_order>> {};
struct atomic_fence_order_capabilities
    : detail::descriptor<std::vector<memory_order>> {};
struct atomic_memory_scope_capabilities
    : detail::descriptor<std::vector<memory_scope>> {};
struct atomic_fence_scope_capabilities
    : detail::descriptor<std::vector<memory_scope>> {};

} // namespace info::context

class context {
public:
  explicit context(const property_list &propList = {}) : m_props{propList} {
    check_device_count();
  }

  explicit context(async_handler asyncHandler,
                   const property_list &propList = {})
      : m_asyncHandler{asyncHandler}, m_props{propList} {
    check_device_count();
  }

  explicit context(const device &dev, const property_list &propList = {})
      : m_devices{dev}, m_props{propList} {
    check_device_count();
  }

  explicit context(const device &dev, async_handler asyncHandler,
                   const property_list &propList = {})
      : m_devices{dev}, m_asyncHandler{asyncHandler}, m_props{propList} {
    check_device_count();
  }

  explicit context(const platform &plt, const property_list &propList = {})
      : m_devices{plt.get_devices()}, m_props{propList} {
    check_device_count();
  }

  explicit context(const platform &plt, async_handler asyncHandler,
                   const property_list &propList = {})
      : m_devices{plt.get_devices()}, m_asyncHandler{asyncHandler},
        m_props{propList} {
    check_device_count();
  }

  explicit context(const std::vector<device> &deviceList,
                   const property_list &propList = {})
      : m_devices{deviceList}, m_props{propList} {
    check_device_count();
  }

  explicit context(const std::vector<device> &deviceList,
                   async_handler asyncHandler,
                   const property_list &propList = {})
      : m_devices{deviceList}, m_asyncHandler{asyncHandler}, m_props{propList} {
    check_device_count();
  }

  /* -- property interface members -- */

  friend bool operator==(const context &lhs, const context &rhs) {
    return lhs.m_id.get() == rhs.m_id.get();
  }

  backend get_backend() const noexcept { return backend::proto_sycl; }

  platform get_platform() const { return platform{}; }

  std::vector<device> get_devices() const { return m_devices; }

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  friend detail::queue_impl;
  friend std::hash<context>;

  void check_device_count() const {
    if (m_devices.empty())
      detail::throw_sycl_exception(
          sycl::errc::invalid,
          "Context must be associated with at least one device");
  }

  std::shared_ptr<std::monostate> m_id{std::make_shared<std::monostate>()};
  std::vector<device> m_devices{device{}};
  std::optional<async_handler> m_asyncHandler;
  property_list m_props;
};

template <>
inline typename info::context::platform::return_type
context::get_info<info::context::platform>() const {
  return get_platform();
}

template <>
inline typename info::context::devices::return_type
context::get_info<info::context::devices>() const {
  return get_devices();
}

template <>
inline typename info::context::atomic_memory_order_capabilities::return_type
context::get_info<info::context::atomic_memory_order_capabilities>() const {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::context::atomic_fence_order_capabilities::return_type
context::get_info<info::context::atomic_fence_order_capabilities>() const {
  return {memory_order::relaxed, memory_order::acquire, memory_order::release,
          memory_order::acq_rel, memory_order::seq_cst};
}

template <>
inline typename info::context::atomic_memory_scope_capabilities::return_type
context::get_info<info::context::atomic_memory_scope_capabilities>() const {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

template <>
inline typename info::context::atomic_fence_scope_capabilities::return_type
context::get_info<info::context::atomic_fence_scope_capabilities>() const {
  return {memory_scope::work_item, memory_scope::sub_group,
          memory_scope::work_group, memory_scope::device, memory_scope::system};
}

} // namespace sycl

template <> struct std::hash<sycl::context> {
  std::size_t operator()(const sycl::context &ctx) const noexcept {
    return std::hash<decltype(ctx.m_id)>{}(ctx.m_id);
  }
};
