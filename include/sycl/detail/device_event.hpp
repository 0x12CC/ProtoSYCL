#pragma once

namespace sycl {

class device_event;

namespace detail {
device_event make_device_event();
}

class device_event {
public:
  void wait() noexcept {}

private:
  friend device_event detail::make_device_event();

  device_event(/*__unspecified__*/) = default;
};

inline device_event detail::make_device_event() { return device_event{}; }

} // namespace sycl
