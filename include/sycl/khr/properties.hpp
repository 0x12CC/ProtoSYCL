#pragma once

// Minimal implementation of the sycl_khr_properties spelling for compile-time
// kernel properties (KhronosGroup/SYCL-Docs PR #980). Provides the compile-time
// kernel-placed attributes that the reflection based free-function-kernel path
// can cover: reqd_work_group_size, reqd_sub_group_size, and work_group_size_hint
// (each a variable template), their keys, and the classification traits. The
// full `properties` container is intentionally out of scope for this prototype.
//
// Note: vec_type_hint (deprecated, type-valued) and device_has (transitive /
// call-graph, compiler-only) are deliberately not modelled here.

#include <cstddef>
#include <type_traits>

#define SYCL_KHR_PROPERTIES 1

namespace sycl::khr {

namespace property {

namespace key {
// Each property has an associated "key" type used to query a properties list.
struct reqd_work_group_size {};
struct reqd_sub_group_size {};
struct work_group_size_hint {};
} // namespace key

// Compile-time properties: variable templates whose (unspecified) type carries
// the values as compile-time constants. Per the spec, a compile-time property
// contains no runtime-supplied values.
template <std::size_t... Dims> struct reqd_work_group_size_t {
  using key_t = key::reqd_work_group_size;
  static constexpr std::size_t rank = sizeof...(Dims);
};
template <std::size_t... Dims>
inline constexpr reqd_work_group_size_t<Dims...> reqd_work_group_size{};

template <std::size_t Size> struct reqd_sub_group_size_t {
  using key_t = key::reqd_sub_group_size;
  static constexpr std::size_t size = Size;
};
template <std::size_t Size>
inline constexpr reqd_sub_group_size_t<Size> reqd_sub_group_size{};

template <std::size_t... Dims> struct work_group_size_hint_t {
  using key_t = key::work_group_size_hint;
  static constexpr std::size_t rank = sizeof...(Dims);
};
template <std::size_t... Dims>
inline constexpr work_group_size_hint_t<Dims...> work_group_size_hint{};

} // namespace property

// --- Classification traits (subset of the spec) ---------------------------

template <typename T> struct is_property : std::false_type {};
template <std::size_t... Dims>
struct is_property<property::reqd_work_group_size_t<Dims...>> : std::true_type {
};
template <std::size_t Size>
struct is_property<property::reqd_sub_group_size_t<Size>> : std::true_type {};
template <std::size_t... Dims>
struct is_property<property::work_group_size_hint_t<Dims...>> : std::true_type {
};
template <typename T> inline constexpr bool is_property_v = is_property<T>::value;

template <typename T> struct is_property_key : std::false_type {};
template <>
struct is_property_key<property::key::reqd_work_group_size> : std::true_type {};
template <>
struct is_property_key<property::key::reqd_sub_group_size> : std::true_type {};
template <>
struct is_property_key<property::key::work_group_size_hint> : std::true_type {};
template <typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;

// Distinguishes compile-time property keys (no runtime-supplied values) from
// runtime ones. All the kernel attributes modelled here are compile-time; a
// runtime property would leave this at std::false_type.
template <typename T>
struct is_property_key_compile_time : std::false_type {};
template <>
struct is_property_key_compile_time<property::key::reqd_work_group_size>
    : std::true_type {};
template <>
struct is_property_key_compile_time<property::key::reqd_sub_group_size>
    : std::true_type {};
template <>
struct is_property_key_compile_time<property::key::work_group_size_hint>
    : std::true_type {};
template <typename T>
inline constexpr bool is_property_key_compile_time_v =
    is_property_key_compile_time<T>::value;

} // namespace sycl::khr
