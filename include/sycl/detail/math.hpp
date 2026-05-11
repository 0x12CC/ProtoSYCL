#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "half.hpp"
#include "marray.hpp"
#include "vec.hpp"

namespace sycl::detail {

template <typename T>
concept GenericIntegerTypeScalar =
    std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
    std::is_same_v<T, short> || std::is_same_v<T, int> ||
    std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned short> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long>;

template <typename T>
concept GenericIntegerType =
    GenericIntegerTypeScalar<T> ||
    (Vec<T> && GenericIntegerTypeScalar<typename T::value_type>) ||
    (MArray<T> && GenericIntegerTypeScalar<typename T::value_type>);

template <typename T>
concept GenericFloatingPointTypeScalar =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, half>;

template <typename T>
concept GenericFloatingPointType =
    GenericFloatingPointTypeScalar<T> ||
    (Vec<T> && GenericFloatingPointTypeScalar<typename T::value_type>) ||
    (MArray<T> && GenericFloatingPointTypeScalar<typename T::value_type>);

template <typename T>
concept GenericScalarType =
    std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
    std::is_same_v<T, short> || std::is_same_v<T, int> ||
    std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned short> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long> || std::is_same_v<T, float> ||
    std::is_same_v<T, double> || std::is_same_v<T, half>;

template <typename T>
concept NonScalar = Vec<T> || MArray<T>;

template <typename T>
concept GenInt = std::is_integral_v<T>;

template <typename T> struct non_scalar_return_type {
  using type = T;
};

template <typename T, int N>
struct non_scalar_return_type<__writeable_swizzle__<vec<T, N>, N>> {
  using type = vec<T, N>;
};

template <typename T, int N> struct non_scalar_return_type<vec<const T, N>> {
  using type = vec<T, N>;
};

} // namespace sycl::detail

namespace sycl {

#define SYCL_UNARY_MATH_FN(NAME, DEFINITION)                                   \
  inline float NAME(float value) { return DEFINITION(value); }                 \
  inline double NAME(double value) { return DEFINITION(value); }               \
  inline half NAME(half value) {                                               \
    return DEFINITION(static_cast<float>(value));                              \
  }                                                                            \
  template <detail::NonScalar NonScalar> auto NAME(NonScalar value) {          \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;  \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < value.size(); i++)                             \
      result[i] = DEFINITION(value[i]);                                        \
    return result;                                                             \
  }

SYCL_UNARY_MATH_FN(acos, std::acos)
SYCL_UNARY_MATH_FN(acosh, std::acosh)
SYCL_UNARY_MATH_FN(acospi, acospi)
SYCL_UNARY_MATH_FN(asin, std::asin)
SYCL_UNARY_MATH_FN(asinh, std::asinh)
SYCL_UNARY_MATH_FN(asinpi, asinpi)
SYCL_UNARY_MATH_FN(atan, std::atan)
SYCL_UNARY_MATH_FN(atanh, std::atanh)
SYCL_UNARY_MATH_FN(atanpi, atanpi)
SYCL_UNARY_MATH_FN(ceil, std::ceil)
SYCL_UNARY_MATH_FN(exp10, exp10)
SYCL_UNARY_MATH_FN(exp2, std::exp2)
SYCL_UNARY_MATH_FN(expm1, std::expm1)
SYCL_UNARY_MATH_FN(fabs, std::fabs)
SYCL_UNARY_MATH_FN(floor, std::floor)
SYCL_UNARY_MATH_FN(log, std::log)
SYCL_UNARY_MATH_FN(log10, std::log10)
SYCL_UNARY_MATH_FN(log1p, std::log1p)
SYCL_UNARY_MATH_FN(log2, std::log2)
SYCL_UNARY_MATH_FN(logb, std::logb)
SYCL_UNARY_MATH_FN(rint, std::rint)
SYCL_UNARY_MATH_FN(sinpi, sinpi)
SYCL_UNARY_MATH_FN(sqrt, std::sqrt)
SYCL_UNARY_MATH_FN(tan, std::tan)
SYCL_UNARY_MATH_FN(tanh, std::tanh)
SYCL_UNARY_MATH_FN(tanpi, tanpi)
SYCL_UNARY_MATH_FN(tgamma, std::tgamma)
SYCL_UNARY_MATH_FN(trunc, std::trunc)
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

SYCL_BINARY_MATH_FN(atan2, std::atan2)
SYCL_BINARY_MATH_FN(atan2pi, atan2pi)
SYCL_BINARY_MATH_FN(fdim, std::fdim)
SYCL_BINARY_MATH_FN(fmax, std::fmax)
SYCL_BINARY_MATH_FN(fmin, std::fmin)
SYCL_BINARY_MATH_FN(fmod, std::fmod)
SYCL_BINARY_MATH_FN(isequal, [](auto x, auto y) { return x == y; });
SYCL_BINARY_MATH_FN(isnotequal, [](auto x, auto y) { return x != y; });
SYCL_BINARY_MATH_FN(nextafter, std::nextafter)
#undef SYCL_BINARY_MATH_FN

#define SYCL_TERNARY_MATH_FN(NAME, DEFINITION)                                 \
  inline float NAME(float a, float b, float c) { return DEFINITION(a, b, c); } \
  inline double NAME(double a, double b, double c) {                           \
    return DEFINITION(a, b, c);                                                \
  }                                                                            \
  inline half NAME(half a, half b, half c) {                                   \
    return DEFINITION(static_cast<float>(a), static_cast<float>(b),            \
                      static_cast<float>(c));                                  \
  }                                                                            \
  template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2,        \
            detail::NonScalar NonScalar3>                                      \
  auto NAME(NonScalar1 a, NonScalar2 b, NonScalar3 c) {                        \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type; \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < a.size(); i++)                                 \
      result[i] = DEFINITION(a[i], b[i], c[i]);                                \
    return result;                                                             \
  }

SYCL_TERNARY_MATH_FN(fma, std::fma)
SYCL_TERNARY_MATH_FN(mad, [](auto a, auto b, auto c) { return a * b + c; })
#undef SYCL_TERNARY_MATH_FN

template <detail::GenericScalarType T1, detail::GenericScalarType T2>
auto max(T1 x, T2 y) {
  return std::max(x, y);
}

template <detail::NonScalar NonScalar>
auto max(NonScalar x, typename NonScalar::value_type y) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = max(x[i], y);
  return result;
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto max(NonScalar1 x, NonScalar2 y) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = max(x[i], y[i]);
  return result;
}

template <detail::GenericScalarType T1, detail::GenericScalarType T2>
auto min(T1 x, T2 y) {
  return std::min(x, y);
}

template <detail::NonScalar NonScalar>
auto min(NonScalar x, typename NonScalar::value_type y) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = min(x[i], y);
  return result;
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto min(NonScalar1 x, NonScalar2 y) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = min(x[i], y[i]);
  return result;
}

template <detail::GenericFloatingPointType GenFloat1,
          detail::GenericFloatingPointType GenFloat2,
          detail::GenericFloatingPointType GenFloat3>
  requires(sizeof(GenFloat1) == sizeof(GenFloat2) &&
           sizeof(GenFloat1) == sizeof(GenFloat3))
auto mix(GenFloat1 x, GenFloat2 y, GenFloat3 a) {
  if constexpr (detail::is_vec_v<GenFloat1> || detail::is_marray_v<GenFloat1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat1>::type;
    using ValueT = typename ReturnT::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = mix(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]),
                      static_cast<ValueT>(a[i]));
    return result;
  } else {
    return GenFloat1{x * (GenFloat1{1} - a) + y * a};
  }
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto mix(NonScalar1 x, NonScalar2 y, typename NonScalar1::value_type a) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = mix(x[i], y[i], a);
  return result;
}

template <detail::GenericFloatingPointType GenFloat1,
          detail::GenericFloatingPointType GenFloat2>
  requires(sizeof(GenFloat1) == sizeof(GenFloat2))
auto step(GenFloat1 edge, GenFloat2 x) {
  if constexpr (detail::is_vec_v<GenFloat1> || detail::is_marray_v<GenFloat1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat1>::type;
    using ValueT = typename ReturnT::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = step(static_cast<ValueT>(edge[i]), static_cast<ValueT>(x[i]));
    return result;
  } else {
    return GenFloat1{x < edge ? GenFloat1{0} : GenFloat1{1}};
  }
}

template <detail::NonScalar NonScalar>
auto step(typename NonScalar::value_type edge, NonScalar x) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = step(edge, x[i]);
  return result;
}

template <detail::GenericFloatingPointType GenFloat1,
          detail::GenericFloatingPointType GenFloat2,
          detail::GenericFloatingPointType GenFloat3>
  requires(sizeof(GenFloat1) == sizeof(GenFloat2) &&
           sizeof(GenFloat1) == sizeof(GenFloat3))
auto smoothstep(GenFloat1 edge0, GenFloat2 edge1, GenFloat3 x) {
  if constexpr (detail::is_vec_v<GenFloat1> || detail::is_marray_v<GenFloat1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat1>::type;
    using ValueT = typename ReturnT::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] =
          smoothstep(static_cast<ValueT>(edge0[i]),
                     static_cast<ValueT>(edge1[i]), static_cast<ValueT>(x[i]));
    return result;
  } else if constexpr (std::is_same_v<GenFloat1, half>) {
    _Float16 t =
        std::clamp((_Float16((float)x) - _Float16((float)edge0)) /
                       (_Float16((float)edge1) - _Float16((float)edge0)),
                   _Float16(0), _Float16(1));
    return half(t * t * (_Float16(3) - _Float16(2) * t));
  } else {
    const auto t =
        std::clamp((x - edge0) / (edge1 - edge0), GenFloat1{0}, GenFloat1{1});
    return t * t * (GenFloat1{3} - GenFloat1{2} * t);
  }
}

template <detail::NonScalar NonScalar>
auto smoothstep(typename NonScalar::value_type edge0,
                typename NonScalar::value_type edge1, NonScalar x) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = smoothstep(edge0, edge1, x[i]);
  return result;
}

template <detail::GenericFloatingPointType GenFloat>
auto degrees(GenFloat radians) {
  if constexpr (detail::is_vec_v<GenFloat> || detail::is_marray_v<GenFloat>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat>::type;
    using ValueT = typename GenFloat::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < radians.size(); i++)
      result[i] = degrees(static_cast<ValueT>(radians[i]));
    return result;
  } else {
    return GenFloat{radians *
                    static_cast<GenFloat>(180.0 / 3.14159265358979323846)};
  }
}

template <detail::GenericFloatingPointType GenFloat>
auto radians(GenFloat degrees) {
  if constexpr (detail::is_vec_v<GenFloat> || detail::is_marray_v<GenFloat>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat>::type;
    using ValueT = typename GenFloat::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < degrees.size(); i++)
      result[i] = radians(static_cast<ValueT>(degrees[i]));
    return result;
  } else {
    return GenFloat{degrees *
                    static_cast<GenFloat>(3.14159265358979323846 / 180.0)};
  }
}

template <detail::GenericFloatingPointType GenFloat> auto sign(GenFloat x) {
  if constexpr (detail::is_vec_v<GenFloat> || detail::is_marray_v<GenFloat>) {
    using ReturnT = typename detail::non_scalar_return_type<GenFloat>::type;
    using ValueT = typename GenFloat::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = sign(static_cast<ValueT>(x[i]));
    return result;
  } else {
    if (std::isnan(static_cast<double>(x)))
      return x;
    if (x == GenFloat{0})
      return GenFloat{0};
    if (x > GenFloat{0})
      return GenFloat{1};
    else
      return GenFloat{-1};
  }
}

template <detail::GenericScalarType T1, detail::GenericScalarType T2,
          detail::GenericScalarType T3>
auto clamp(T1 x, T2 minval, T3 maxval) {
  return std::clamp(x, minval, maxval);
}

template <detail::NonScalar NonScalar>
auto clamp(NonScalar x, typename NonScalar::value_type minval,
           typename NonScalar::value_type maxval) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = std::clamp(x[i], minval, maxval);
  return result;
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2,
          detail::NonScalar NonScalar3>
auto clamp(NonScalar1 x, NonScalar2 minval, NonScalar3 maxval) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = std::clamp(x[i], minval[i], maxval[i]);
  return result;
}

} // namespace sycl
