#pragma once

#include <memory>
#include <variant>

#include "backend.hpp"
#include "device.hpp"
#include "util.descriptor.hpp"

namespace sycl {

class device;

namespace info::platform {

struct version : detail::descriptor<std::string> {};
struct name : detail::descriptor<std::string> {};
struct vendor : detail::descriptor<std::string> {};
// Deprecated
struct extensions : detail::descriptor<std::vector<std::string>> {};

} // namespace info::platform

class platform {
public:
  platform() = default;

  template <detail::DeviceSelector DeviceSelector>
  explicit platform(const DeviceSelector &deviceSelector) {
    device{deviceSelector};
  }

  /* -- common interface members -- */

  backend get_backend() const noexcept { return backend::proto_sycl; }

  std::vector<device>
  get_devices(info::device_type type = info::device_type::all) const {
    if (type == info::device_type::all || type == info::device_type::cpu ||
        type == info::device_type::automatic)
      return {device{}};
    return {};
  }

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

  bool has(aspect asp) const {
    std::ignore = asp;
    return true;
  }

  // Deprecated
  bool has_extension(const std::string &extension) const {
    std::ignore = extension;
    return false;
  }

  static std::vector<platform> get_platforms() { return {platform{}}; }

  friend bool operator==(const platform &, const platform &) = default;

private:
  friend std::hash<platform>;

  std::shared_ptr<std::monostate> m_id{std::make_shared<std::monostate>()};
};

template <>
inline typename info::platform::name::return_type
platform::get_info<info::platform::name>() const {
  return "<sycl platform>";
}

template <>
inline typename info::platform::vendor::return_type
platform::get_info<info::platform::vendor>() const {
  return "<sycl platform vendor>";
}

template <>
inline typename info::platform::version::return_type
platform::get_info<info::platform::version>() const {
  return "0.1.0";
}

template <>
inline typename info::platform::extensions::return_type
platform::get_info<info::platform::extensions>() const {
  return {};
}

} // namespace sycl

template <> struct std::hash<sycl::platform> {
  std::size_t operator()(const sycl::platform &platform) const noexcept {
    return std::hash<decltype(platform.m_id)>{}(platform.m_id);
  }
};
