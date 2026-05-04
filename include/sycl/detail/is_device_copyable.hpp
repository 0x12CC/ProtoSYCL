#pragma once

#include <type_traits>

namespace sycl {

template <typename T>
struct is_device_copyable : std::is_trivially_copyable<T> {};

template <typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;

} // namespace sycl
