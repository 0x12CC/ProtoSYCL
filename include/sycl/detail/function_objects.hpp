#pragma once

#include <functional>
#include <type_traits>

#include "half.hpp"

namespace sycl {

namespace detail {

template <typename T>
struct is_arithmetic
    : std::bool_constant<std::is_arithmetic_v<T> || std::is_same_v<T, half>> {};

template <class T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

template <typename T>
concept Arithmetic = detail::is_arithmetic_v<T>;

template <typename T>
concept Boolean = std::is_same_v<std::remove_cv_t<T>, bool>;

} // namespace detail

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity {
  static constexpr bool value = false;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

using std::plus;
template <typename OpT, typename T>
struct has_known_identity<plus<OpT>, T>
    : std::bool_constant<detail::is_arithmetic_v<T>> {};
template <typename OpT, detail::Arithmetic T>
struct known_identity<plus<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = T{};
};

using std::multiplies;
template <typename OpT, typename T>
struct has_known_identity<multiplies<OpT>, T>
    : std::bool_constant<detail::is_arithmetic_v<T>> {};
template <typename OpT, detail::Arithmetic T>
struct known_identity<multiplies<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = T{1};
};

using std::bit_and;
template <typename OpT, typename T>
struct has_known_identity<bit_and<OpT>, T>
    : std::bool_constant<std::is_integral_v<T>> {};
template <typename OpT, detail::Arithmetic T>
struct known_identity<bit_and<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = ~T{};
};

using std::bit_or;
template <typename OpT, typename T>
struct has_known_identity<bit_or<OpT>, T>
    : std::bool_constant<std::is_integral_v<T>> {};
template <typename OpT, std::integral T> struct known_identity<bit_or<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = T{};
};

using std::bit_xor;
template <typename OpT, typename T>
struct has_known_identity<bit_xor<OpT>, T>
    : std::bool_constant<std::is_integral_v<T>> {};
template <typename OpT, std::integral T>
struct known_identity<bit_xor<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = T{};
};

using std::logical_and;
template <typename OpT, typename T>
struct has_known_identity<logical_and<OpT>, T>
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template <typename OpT, detail::Boolean T>
struct known_identity<logical_and<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = true;
};

using std::logical_or;
template <typename OpT, typename T>
struct has_known_identity<logical_or<OpT>, T>
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};
template <typename OpT, detail::Boolean T>
struct known_identity<logical_or<OpT>, T> {
  static constexpr std::remove_cv_t<T> value = false;
};

template <typename T = void> struct minimum {
  T operator()(const T &x, const T &y) const { return x < y ? x : y; }
};
template <> struct minimum<void> {
  template <typename T, typename U>
  std::common_type_t<T &&, U &&> operator()(T &&x, U &&y) const {
    return x < y ? std::forward<T>(x) : std::forward<U>(y);
  }
};
template <typename OpT, typename T>
struct has_known_identity<minimum<OpT>, T>
    : std::bool_constant<detail::is_arithmetic_v<T>> {};
template <typename OpT, detail::Arithmetic T>
struct known_identity<minimum<OpT>, T> {
  static constexpr std::remove_cv_t<T> value =
      std::is_floating_point_v<T> ? std::numeric_limits<T>::infinity()
                                  : std::numeric_limits<T>::max();
};

template <typename T = void> struct maximum {
  T operator()(const T &x, const T &y) const { return x >= y ? x : y; }
};
template <> struct maximum<void> {
  template <typename T, typename U>
  std::common_type_t<T &&, U &&> operator()(T &&x, U &&y) const {
    return x > y ? std::forward<T>(x) : std::forward<U>(y);
  }
};
template <typename OpT, typename T>
struct has_known_identity<maximum<OpT>, T>
    : std::bool_constant<detail::is_arithmetic_v<T>> {};
template <typename OpT, detail::Arithmetic T>
struct known_identity<maximum<OpT>, T> {
  static constexpr std::remove_cv_t<T> value =
      std::is_floating_point_v<T> ? -std::numeric_limits<T>::infinity()
                                  : std::numeric_limits<T>::lowest();
};

} // namespace sycl
