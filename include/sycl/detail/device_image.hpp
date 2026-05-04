#pragma once

#include <memory>
#include <variant>

#include "bundle_state.hpp"
#include "device.hpp"
#include "kernel_id.hpp"

namespace sycl {

template <bundle_state State> class device_image;

namespace detail {
template <bundle_state State> device_image<State> create_device_image();
}

template <bundle_state State> class device_image {
public:
  friend bool operator==(const device_image &, const device_image &) = default;

  bool has_kernel(const kernel_id &kernelId) const noexcept { return true; }

  bool has_kernel(const kernel_id &kernelId, const device &dev) const noexcept {
    return true;
  }

private:
  friend device_image<State> detail::create_device_image();
  friend std::hash<device_image>;

  device_image() {}

  std::shared_ptr<std::monostate> m_id{std::make_shared<std::monostate>()};
};

template <bundle_state State>
device_image<State> detail::create_device_image() {
  return device_image<State>{};
}

} // namespace sycl

template <sycl::bundle_state State>
struct std::hash<sycl::device_image<State>> {
  std::size_t
  operator()(const sycl::device_image<State> &device_image) const noexcept {
    return std::hash<decltype(device_image.m_id)>{}(device_image.m_id);
  }
};
