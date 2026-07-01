#pragma once

#include "math.hpp"

namespace sycl {
#define SYCL_UNARY_MATH_FN(NAME, DEFINITION)                                   \
  inline bool NAME(float value) { return DEFINITION(value); }                  \
  inline bool NAME(double value) { return DEFINITION(value); }                 \
  inline bool NAME(half value) {                                               \
    return DEFINITION(static_cast<float>(value));                              \
  }                                                                            \
  template <detail::NonScalar NonScalar> auto NAME(NonScalar value) {          \
    if constexpr (sycl::detail::is_marray_v<NonScalar>) {                      \
      typename detail::switch_element_type<NonScalar, bool>::type result;      \
      for (std::size_t i = 0; i < value.size(); i++)                           \
        result[i] = DEFINITION(value[i]);                                      \
      return result;                                                           \
    } else {                                                                   \
      using ElementT = typename NonScalar::value_type;                         \
      using ReturnT = std::conditional_t<                                      \
          std::is_same_v<ElementT, float>, std::int32_t,                       \
          std::conditional_t<std::is_same_v<ElementT, double>, std::int64_t,   \
                             std::int16_t>>;                                   \
      typename detail::switch_element_type<NonScalar, ReturnT>::type result;   \
      for (std::size_t i = 0; i < value.size(); i++)                           \
        result[i] = DEFINITION(static_cast<ElementT>(value[i])) ? -1 : 0;      \
      return result;                                                           \
    }                                                                          \
  }

SYCL_UNARY_MATH_FN(isfinite, std::isfinite)
SYCL_UNARY_MATH_FN(isinf, std::isinf)
SYCL_UNARY_MATH_FN(isnan, std::isnan)
SYCL_UNARY_MATH_FN(isnormal, std::isnormal)
SYCL_UNARY_MATH_FN(signbit, std::signbit)
#undef SYCL_UNARY_MATH_FN

#define SYCL_BINARY_MATH_FN(NAME, DEFINITION)                                  \
  inline bool NAME(float lhs, float rhs) { return DEFINITION(lhs, rhs); }      \
  inline bool NAME(double lhs, double rhs) { return DEFINITION(lhs, rhs); }    \
  inline bool NAME(half lhs, half rhs) {                                       \
    return DEFINITION(static_cast<float>(lhs), static_cast<float>(rhs));       \
  }                                                                            \
  template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>        \
  auto NAME(NonScalar1 lhs, NonScalar2 rhs) {                                  \
    if constexpr (sycl::detail::is_marray_v<NonScalar1>) {                     \
      typename detail::switch_element_type<NonScalar1, bool>::type result;     \
      for (std::size_t i = 0; i < lhs.size(); i++)                             \
        result[i] = DEFINITION(lhs[i], rhs[i]);                                \
      return result;                                                           \
    } else {                                                                   \
      using ElementT = typename NonScalar1::value_type;                        \
      using ReturnT = std::conditional_t<                                      \
          std::is_same_v<ElementT, float>, std::int32_t,                       \
          std::conditional_t<std::is_same_v<ElementT, double>, std::int64_t,   \
                             std::int16_t>>;                                   \
      typename detail::switch_element_type<NonScalar1, ReturnT>::type result;  \
      for (std::size_t i = 0; i < lhs.size(); i++)                             \
        result[i] = DEFINITION(static_cast<ElementT>(lhs[i]),                  \
                               static_cast<ElementT>(rhs[i]))                  \
                        ? -1                                                   \
                        : 0;                                                   \
      return result;                                                           \
    }                                                                          \
  }

SYCL_BINARY_MATH_FN(isequal, [](auto x, auto y) { return x == y; });
SYCL_BINARY_MATH_FN(isnotequal, [](auto x, auto y) { return x != y; });
SYCL_BINARY_MATH_FN(isgreater, [](auto x, auto y) { return x > y; });
SYCL_BINARY_MATH_FN(isgreaterequal, [](auto x, auto y) { return x >= y; });
SYCL_BINARY_MATH_FN(isless, [](auto x, auto y) { return x < y; });
SYCL_BINARY_MATH_FN(islessequal, [](auto x, auto y) { return x <= y; });
SYCL_BINARY_MATH_FN(islessgreater,
                    [](auto x, auto y) { return x < y || x > y; });
SYCL_BINARY_MATH_FN(isordered, [](auto x, auto y) {
  return !std::isnan(x) && !std::isnan(y);
});
SYCL_BINARY_MATH_FN(isunordered, [](auto x, auto y) {
  return std::isnan(x) || std::isnan(y);
});
#undef SYCL_BINARY_MATH_FN

namespace detail {

template <typename T>
concept RelationalGenInt =
    (detail::is_marray_v<T> && std::is_same_v<typename T::value_type, bool>) ||
    (detail::is_vec_v<T> &&
     (std::is_same_v<typename T::value_type, std::int8_t> ||
      std::is_same_v<typename T::value_type, std::int16_t> ||
      std::is_same_v<typename T::value_type, std::int32_t> ||
      std::is_same_v<typename T::value_type, std::int64_t>));

template <typename T>
concept RelationalDeprecatedGenInt =
    std::is_same_v<T, signed char> || std::is_same_v<T, short> ||
    std::is_same_v<T, int> || std::is_same_v<T, long> ||
    std::is_same_v<T, long long> ||
    (detail::is_marray_v<T> &&
     std::is_same_v<typename T::value_type, signed char>) ||
    (detail::is_marray_v<T> && std::is_same_v<typename T::value_type, short>) ||
    (detail::is_marray_v<T> && std::is_same_v<typename T::value_type, int>) ||
    (detail::is_marray_v<T> && std::is_same_v<typename T::value_type, long>) ||
    (detail::is_marray_v<T> &&
     std::is_same_v<typename T::value_type, long long>);

} // namespace detail

template <detail::RelationalGenInt GenInt> auto any(GenInt x) {
  if constexpr (detail::is_marray_v<GenInt>) {
    for (std::size_t i = 0; i < x.size(); i++)
      if (x[i])
        return true;
    return false;
  } else {
    for (std::size_t i = 0; i < x.size(); i++)
      if (x[i] & (1ull << (sizeof(typename GenInt::value_type) * 8 - 1)))
        return 1;
    return 0;
  }
}

template <detail::RelationalDeprecatedGenInt GenInt>
[[deprecated]] bool any(GenInt x) {
  if constexpr (detail::is_marray_v<GenInt>) {
    for (std::size_t i = 0; i < x.size(); i++)
      if (x[i] & (1ull << (sizeof(typename GenInt::value_type) * 8 - 1)))
        return true;
    return false;
  } else {
    return x & (1ull << (sizeof(GenInt) * 8 - 1)) ? true : false;
  }
}

template <detail::RelationalGenInt GenInt> auto all(GenInt x) {
  if constexpr (detail::is_marray_v<GenInt>) {
    for (std::size_t i = 0; i < x.size(); i++)
      if (!x[i])
        return false;
    return true;
  } else {
    for (std::size_t i = 0; i < x.size(); i++)
      if (!(x[i] & (1ull << (sizeof(typename GenInt::value_type) * 8 - 1))))
        return 0;
    return 1;
  }
}

template <detail::RelationalDeprecatedGenInt GenInt>
[[deprecated]] bool all(GenInt x) {
  if constexpr (detail::is_marray_v<GenInt>) {
    for (std::size_t i = 0; i < x.size(); i++)
      if (!(x[i] & (1ull << (sizeof(typename GenInt::value_type) * 8 - 1))))
        return false;
    return true;
  } else {
    return x & (1ull << (sizeof(GenInt) * 8 - 1)) ? true : false;
  }
}

template <typename GenType1, typename GenType2, typename GenType3>
auto bitselect(GenType1 a, GenType2 b, GenType3 c) {
  if constexpr (detail::is_marray_v<GenType1> || detail::is_vec_v<GenType1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenType1>::type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = bitselect(a[i], b[i], c[i]);
    return result;
  } else if constexpr (std::is_same_v<GenType1, float>) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(a);
    const std::uint32_t y = std::bit_cast<std::uint32_t>(b);
    const std::uint32_t z = std::bit_cast<std::uint32_t>(c);
    const std::uint32_t result = (x & ~z) | (y & z);
    return std::bit_cast<float>(result);
  } else if constexpr (std::is_same_v<GenType1, double>) {
    const std::uint64_t x = std::bit_cast<std::uint64_t>(a);
    const std::uint64_t y = std::bit_cast<std::uint64_t>(b);
    const std::uint64_t z = std::bit_cast<std::uint64_t>(c);
    const std::uint64_t result = (x & ~z) | (y & z);
    return std::bit_cast<double>(result);
  } else if constexpr (std::is_same_v<GenType1, half>) {
    const std::uint16_t x = std::bit_cast<std::uint16_t>(a);
    const std::uint16_t y = std::bit_cast<std::uint16_t>(b);
    const std::uint16_t z = std::bit_cast<std::uint16_t>(c);
    const std::uint16_t result = (x & ~z) | (y & z);
    return std::bit_cast<half>(result);
  } else {
    return GenType1((a & ~c) | (b & c));
  }
}

template <typename Scalar> Scalar select(Scalar a, Scalar b, bool c) {
  return c ? b : a;
}

template <typename NonScalar1, typename NonScalar2, typename NonScalar3>
auto select(NonScalar1 a, NonScalar2 b, NonScalar3 c) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  if constexpr (detail::is_marray_v<NonScalar1>)
    for (std::size_t i = 0; i < a.size(); i++)
      result[i] = c[i] ? b[i] : a[i];
  else
    for (std::size_t i = 0; i < a.size(); i++)
      result[i] =
          (c[i] & (1ull << (sizeof(typename NonScalar1::value_type) * 8 - 1)))
              ? b[i]
              : a[i];
  return result;
}

} // namespace sycl
