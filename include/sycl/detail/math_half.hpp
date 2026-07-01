#pragma once

#include "math.hpp"

namespace sycl::half_precision {

#define SYCL_UNARY_MATH_FN(NAME, DEFINITION)                                   \
  inline float NAME(float value) { return DEFINITION(value); }                 \
  inline double NAME(double value) { return DEFINITION(value); }               \
  inline half NAME(half value) {                                               \
    return DEFINITION(static_cast<float>(value));                              \
  }                                                                            \
  template <detail::NonScalar NonScalar> auto NAME(NonScalar value) {          \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;  \
    using ElementT = typename ReturnT::value_type;                             \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < value.size(); i++)                             \
      result[i] = DEFINITION(ElementT{value[i]});                              \
    return result;                                                             \
  }

SYCL_UNARY_MATH_FN(cos, std::cos);
SYCL_UNARY_MATH_FN(exp, std::exp)
SYCL_UNARY_MATH_FN(exp2, std::exp2)
SYCL_UNARY_MATH_FN(exp10, [](auto x) {
  return std::pow(static_cast<decltype(x)>(10), x);
})
SYCL_UNARY_MATH_FN(log, std::log)
SYCL_UNARY_MATH_FN(log2, std::log2)
SYCL_UNARY_MATH_FN(log10, std::log10)
SYCL_UNARY_MATH_FN(recip, [](auto x) { return decltype(x)(1) / x; })
SYCL_UNARY_MATH_FN(rsqrt, [](auto x) { return decltype(x)(1) / std::sqrt(x); })
SYCL_UNARY_MATH_FN(sin, std::sin)
SYCL_UNARY_MATH_FN(sqrt, std::sqrt)
SYCL_UNARY_MATH_FN(tan, std::tan)
#undef SYCL_UNARY_MATH_FN

#define SYCL_BINARY_MATH_FN(NAME, DEFINITION)                                  \
  inline float NAME(float lhs, float rhs) { return DEFINITION(lhs, rhs); }     \
  inline double NAME(double lhs, double rhs) { return DEFINITION(lhs, rhs); }  \
  inline half NAME(half lhs, half rhs) {                                       \
    return DEFINITION(static_cast<float>(lhs), static_cast<float>(rhs));       \
  }                                                                            \
  template <detail::NonScalar NonScalar>                                       \
  auto NAME(NonScalar lhs, typename NonScalar::value_type rhs) {               \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;  \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < lhs.size(); i++)                               \
      result[i] = DEFINITION(lhs[i], rhs);                                     \
    return result;                                                             \
  }                                                                            \
  template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>        \
  auto NAME(NonScalar1 lhs, NonScalar2 rhs) {                                  \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type; \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < lhs.size(); i++)                               \
      result[i] = DEFINITION(lhs[i], rhs[i]);                                  \
    return result;                                                             \
  }

SYCL_BINARY_MATH_FN(divide, [](auto x, auto y) { return x / y; })
SYCL_BINARY_MATH_FN(powr, [](auto x, auto y) { return std::pow(x, y); })
#undef SYCL_BINARY_MATH_FN

} // namespace sycl::half_precision
