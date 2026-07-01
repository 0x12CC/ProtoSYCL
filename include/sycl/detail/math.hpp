#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>

#include "half.hpp"
#include "marray.hpp"
#include "vec.hpp"

namespace sycl::detail {

constexpr double pi = std::numbers::pi;

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
concept UInt8Bit =
    std::is_same_v<T, std::uint8_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::uint8_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::uint8_t>);

template <typename T>
concept Int8Bit =
    std::is_same_v<T, std::int8_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::int8_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::int8_t>);

template <typename T>
concept UInt16Bit =
    std::is_same_v<T, std::uint16_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::uint16_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::uint16_t>);

template <typename T>
concept Int16Bit =
    std::is_same_v<T, std::int16_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::int16_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::int16_t>);

template <typename T>
concept UInt32Bit =
    std::is_same_v<T, std::uint32_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::uint32_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::uint32_t>);

template <typename T>
concept Int32Bit =
    std::is_same_v<T, std::int32_t> ||
    (Vec<T> && std::is_same_v<typename T::value_type, std::int32_t>) ||
    (MArray<T> && std::is_same_v<typename T::value_type, std::int32_t>);

template <typename T>
concept GenInt32Bit = UInt32Bit<T> || Int32Bit<T>;

template <typename T> struct non_scalar_return_type {
  using type = T;
};

template <typename T, int N> struct non_scalar_return_type<vec<T, N>> {
  using type = vec<T, N>;
  static constexpr int count = N;
};

template <typename T, int N>
struct non_scalar_return_type<__writeable_swizzle__<vec<T, N>, N>> {
  using type = vec<T, N>;
  static constexpr int count = N;
};

template <typename T, int N> struct non_scalar_return_type<vec<const T, N>> {
  using type = vec<T, N>;
  static constexpr int count = N;
};

template <typename T, size_t N> struct non_scalar_return_type<marray<T, N>> {
  using type = marray<T, N>;
  static constexpr int count = N;
};

template <typename T, size_t N>
struct non_scalar_return_type<marray<const T, N>> {
  using type = marray<T, N>;
  static constexpr int count = N;
};

template <typename NonScalar, typename NewElementType>
struct switch_element_type {
  using type = NonScalar;
};

template <typename T, int N, typename NewElementType>
struct switch_element_type<vec<T, N>, NewElementType> {
  using type = vec<NewElementType, N>;
};

template <typename T, int N, typename NewElementType>
struct switch_element_type<__writeable_swizzle__<vec<T, N>, N>,
                           NewElementType> {
  using type = vec<NewElementType, N>;
};

template <typename T, int N, typename NewElementType>
struct switch_element_type<vec<const T, N>, NewElementType> {
  using type = vec<NewElementType, N>;
};

template <typename T, size_t N, typename NewElementType>
struct switch_element_type<marray<T, N>, NewElementType> {
  using type = marray<NewElementType, N>;
};

template <typename T, size_t N, typename NewElementType>
struct switch_element_type<marray<const T, N>, NewElementType> {
  using type = marray<NewElementType, N>;
};

} // namespace sycl::detail

namespace sycl {

#define SYCL_UNARY_MATH_FN(NAME, DEFINITION)                                   \
  inline float NAME(float value) { return DEFINITION(value); }                 \
  inline double NAME(double value) { return DEFINITION(value); }               \
  inline half NAME(half value) { return DEFINITION(value); }                   \
  template <detail::NonScalar NonScalar> auto NAME(NonScalar value) {          \
    using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;  \
    using ElementT = typename ReturnT::value_type;                             \
    ReturnT result;                                                            \
    for (std::size_t i = 0; i < value.size(); i++)                             \
      result[i] = DEFINITION(ElementT{value[i]});                              \
    return result;                                                             \
  }

SYCL_UNARY_MATH_FN(acos, std::acos)
SYCL_UNARY_MATH_FN(acosh, std::acosh)
SYCL_UNARY_MATH_FN(acospi, [](auto x) {
  return std::acos(x) / static_cast<decltype(x)>(detail::pi);
})
SYCL_UNARY_MATH_FN(asin, std::asin)
SYCL_UNARY_MATH_FN(asinh, std::asinh)
SYCL_UNARY_MATH_FN(asinpi, [](auto x) {
  return std::asin(x) / static_cast<decltype(x)>(detail::pi);
})
SYCL_UNARY_MATH_FN(atan, std::atan)
SYCL_UNARY_MATH_FN(atanh, std::atanh)
SYCL_UNARY_MATH_FN(atanpi, [](auto x) {
  return std::atan(x) / static_cast<decltype(x)>(detail::pi);
})
SYCL_UNARY_MATH_FN(cbrt, std::cbrt)
SYCL_UNARY_MATH_FN(ceil, std::ceil)
SYCL_UNARY_MATH_FN(cos, std::cos)
SYCL_UNARY_MATH_FN(cosh, std::cosh)
SYCL_UNARY_MATH_FN(cospi, [](auto x) {
  using T = decltype(x);
  if (std::isnan(x) || std::isinf(x))
    return T{std::numeric_limits<T>::quiet_NaN()};
  if (std::fabs(x) >= T(1ULL << std::numeric_limits<T>::digits))
    return T{1.0};
  const double rem = std::remainder(double(x), 2.0);
  return T(std::sin((0.5 - std::fabs(rem)) * detail::pi));
})
SYCL_UNARY_MATH_FN(erfc, std::erfc)
SYCL_UNARY_MATH_FN(erf, std::erf)
SYCL_UNARY_MATH_FN(exp, std::exp)
SYCL_UNARY_MATH_FN(exp2, std::exp2)
SYCL_UNARY_MATH_FN(exp10, [](auto x) {
  return std::pow(static_cast<decltype(x)>(10), x);
})
SYCL_UNARY_MATH_FN(expm1, std::expm1)
SYCL_UNARY_MATH_FN(fabs, std::fabs)
SYCL_UNARY_MATH_FN(floor, std::floor)
SYCL_UNARY_MATH_FN(lgamma, std::lgamma)
SYCL_UNARY_MATH_FN(log, std::log)
SYCL_UNARY_MATH_FN(log10, std::log10)
SYCL_UNARY_MATH_FN(log1p, std::log1p)
SYCL_UNARY_MATH_FN(log2, std::log2)
SYCL_UNARY_MATH_FN(logb, std::logb)
SYCL_UNARY_MATH_FN(rint, std::rint)
SYCL_UNARY_MATH_FN(round, std::round)
SYCL_UNARY_MATH_FN(rsqrt, [](auto x) {
  return static_cast<decltype(x)>(1) / std::sqrt(x);
})
SYCL_UNARY_MATH_FN(sin, std::sin)
SYCL_UNARY_MATH_FN(sinh, std::sinh)
SYCL_UNARY_MATH_FN(sinpi, [](auto x) {
  using T = decltype(x);
  if (std::isnan(x) || std::isinf(x))
    return T{std::numeric_limits<T>::quiet_NaN()};
  if (std::fabs(x) >= T(1ULL << std::numeric_limits<T>::digits))
    return T{0.0};
  double rem = std::remainder(double(x), 2.0);
  if (rem > 0.5)
    rem = 1.0 - rem;
  else if (rem < -0.5)
    rem = -1.0 - rem;
  if (rem == 0.0 || rem == -0.0)
    return T(rem);
  return T(std::sin(rem * detail::pi));
})
SYCL_UNARY_MATH_FN(sqrt, std::sqrt)
SYCL_UNARY_MATH_FN(tan, std::tan)
SYCL_UNARY_MATH_FN(tanh, std::tanh)
SYCL_UNARY_MATH_FN(tanpi, [](auto x) {
  using T = decltype(x);
  if (std::isnan(x) || std::isinf(x))
    return T(std::numeric_limits<T>::quiet_NaN());
  if (sycl::fabs(x) >= T(1ULL << std::numeric_limits<T>::digits))
    return T(0.0);
  const double rem = std::remainder(double(x), 2.0);
  double rem_sin = rem;
  if (rem_sin > 0.5)
    rem_sin = 1.0 - rem_sin;
  else if (rem_sin < -0.5)
    rem_sin = -1.0 - rem_sin;
  const double sin_part =
      (rem_sin == 0.0) ? rem_sin : sycl::sin(rem_sin * T(detail::pi));
  const double cos_part = sycl::sin((0.5 - sycl::fabs(rem)) * detail::pi);
  if (cos_part == 0.0)
    return T((sin_part >= 0.0) ? std::numeric_limits<T>::infinity()
                               : -std::numeric_limits<T>::infinity());
  return T(sin_part / cos_part);
})
SYCL_UNARY_MATH_FN(tgamma, std::tgamma)
SYCL_UNARY_MATH_FN(trunc, std::trunc)
#undef SYCL_UNARY_MATH_FN

#define SYCL_BINARY_MATH_FN(NAME, DEFINITION)                                  \
  inline float NAME(float lhs, float rhs) { return DEFINITION(lhs, rhs); }     \
  inline double NAME(double lhs, double rhs) { return DEFINITION(lhs, rhs); }  \
  inline half NAME(half lhs, half rhs) { return DEFINITION(lhs, rhs); }        \
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
SYCL_BINARY_MATH_FN(atan2pi, [](auto y, auto x) {
  return std::atan2(y, x) / static_cast<decltype(x)>(detail::pi);
})
SYCL_BINARY_MATH_FN(copysign, std::copysign)
SYCL_BINARY_MATH_FN(fdim, std::fdim)
SYCL_BINARY_MATH_FN(fmax, std::fmax)
SYCL_BINARY_MATH_FN(fmin, std::fmin)
SYCL_BINARY_MATH_FN(fmod, std::fmod)
SYCL_BINARY_MATH_FN(hypot, std::hypot)
SYCL_BINARY_MATH_FN(maxmag, [](auto x, auto y) {
  using ReturnT = decltype(x + y);
  if (std::abs(x) > std::abs(y)) {
    return static_cast<ReturnT>(x);
  } else if (std::abs(x) < std::abs(y)) {
    return static_cast<ReturnT>(y);
  } else {
    return static_cast<ReturnT>(std::max(x, y));
  }
})
SYCL_BINARY_MATH_FN(minmag, [](auto x, auto y) {
  using ReturnT = decltype(x + y);
  if (std::abs(x) < std::abs(y)) {
    return static_cast<ReturnT>(x);
  } else if (std::abs(x) > std::abs(y)) {
    return static_cast<ReturnT>(y);
  } else {
    return static_cast<ReturnT>(std::min(x, y));
  }
})
SYCL_BINARY_MATH_FN(nextafter, []<typename T>(T x, T y) {
  if constexpr (std::is_same_v<T, half>) {
    if (std::isnan(x))
      return x;
    if (std::isnan(y))
      return y;
    if (x == y)
      return y;
    std::int16_t a = std::bit_cast<std::int16_t>(x);
    std::int16_t b = std::bit_cast<std::int16_t>(y);
    if (a & 0x8000)
      a = 0x8000 - a;
    if (b & 0x8000)
      b = 0x8000 - b;
    a += (a < b) ? 1 : -1;
    a = (a < 0) ? std::int16_t(0x8000) - a : a;
    return std::bit_cast<sycl::half>(a);
  } else {
    return std::nextafter(x, y);
  }
})
SYCL_BINARY_MATH_FN(pow, std::pow)
SYCL_BINARY_MATH_FN(powr, std::pow)
SYCL_BINARY_MATH_FN(remainder, std::remainder)
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
    return GenFloat{radians * static_cast<GenFloat>(180.0 / detail::pi)};
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
    return GenFloat{degrees * static_cast<GenFloat>(detail::pi / 180.0)};
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

inline int ilogb(float x) { return std::ilogb(x); }

inline int ilogb(double x) { return std::ilogb(x); }

inline int ilogb(half x) { return ilogb(static_cast<float>(x)); }

template <detail::NonScalar NonScalar> auto ilogb(NonScalar x) {
  using ReturnT = typename detail::switch_element_type<NonScalar, int>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = ilogb(x[i]);
  return result;
}

template <typename Ptr> float fract(float x, Ptr iptr) {
  *iptr = floor(x);
  return fmin(x - floor(x), nextafter(1.0f, 0.0f));
}

template <typename Ptr> double fract(double x, Ptr iptr) {
  *iptr = floor(x);
  return fmin(x - floor(x), nextafter(1.0, 0.0));
}

template <typename Ptr> half fract(half x, Ptr iptr) {
  *iptr = floor(x);
  return fmin(half(x - floor(x)), nextafter(half{1.0}, half{0.0}));
}

template <detail::NonScalar NonScalar, typename Ptr>
auto fract(NonScalar x, Ptr iptr) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  using ElementT = typename ReturnT::value_type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++) {
    auto floor_val = floor(x[i]);
    result[i] = fmin(ElementT(x[i] - floor_val),
                     nextafter(ElementT{1.0}, ElementT{0.0}));
    (*iptr)[i] = floor_val;
  }
  return result;
}

template <typename Ptr> float frexp(float x, Ptr exp) {
  int exponent;
  const float mantissa = std::frexp(x, &exponent);
  *exp = exponent;
  return mantissa;
}

template <typename Ptr> double frexp(double x, Ptr exp) {
  int exponent;
  const double mantissa = std::frexp(x, &exponent);
  *exp = exponent;
  return mantissa;
}

template <typename Ptr> half frexp(half x, Ptr exp) {
  int exponent;
  const float mantissa = std::frexp(static_cast<float>(x), &exponent);
  *exp = exponent;
  return half(mantissa);
}

template <detail::NonScalar NonScalar, typename Ptr>
auto frexp(NonScalar x, Ptr exp) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = frexp(x[i], &(*exp)[i]);
  return result;
}

inline float ldexp(float x, int k) { return std::ldexp(x, k); }

inline double ldexp(double x, int k) { return std::ldexp(x, k); }

inline half ldexp(half x, int k) {
  return half(std::ldexp(static_cast<float>(x), k));
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto ldexp(NonScalar1 x, NonScalar2 k) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = ldexp(x[i], k[i]);
  return result;
}

template <detail::NonScalar NonScalar> auto ldexp(NonScalar x, int k) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = ldexp(x[i], k);
  return result;
}

template <typename Ptr> inline float lgamma_r(float x, Ptr signp) {
  if (x > 0.0f) {
    *signp = 1;
    return std::lgamma(x);
  }
  const float fx = std::floor(x);
  *signp = (std::fmod(fx, 2.0f) != 0.0f) ? 1 : -1;
  return std::lgamma(x);
}

template <typename Ptr> inline double lgamma_r(double x, Ptr signp) {
  if (x > 0.0f) {
    *signp = 1;
    return std::lgamma(x);
  }
  const double dx = std::floor(x);
  *signp = (std::fmod(dx, 2.0) != 0.0) ? 1 : -1;
  return std::lgamma(x);
}

template <typename Ptr> inline half lgamma_r(half x, Ptr signp) {
  return lgamma_r(static_cast<float>(x), signp);
}

template <typename NonScalar, typename Ptr>
auto lgamma_r(NonScalar x, Ptr signp) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = lgamma_r(x[i], &(*signp)[i]);
  return result;
}

template <typename Ptr> float modf(float x, Ptr iptr) {
  *iptr = floor(x);
  return x - floor(x);
}

template <typename Ptr> double modf(double x, Ptr iptr) {
  *iptr = floor(x);
  return x - floor(x);
}

template <typename Ptr> half modf(half x, Ptr iptr) {
  *iptr = floor(x);
  return x - floor(x);
}

template <detail::NonScalar NonScalar, typename Ptr>
auto modf(NonScalar x, Ptr iptr) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++) {
    auto floor_val = floor(x[i]);
    result[i] = x[i] - floor_val;
    (*iptr)[i] = floor_val;
  }
  return result;
}

inline float nan(unsigned int nancode) {
  return std::numeric_limits<float>::quiet_NaN();
}

inline double nan(unsigned long nancode) {
  return std::numeric_limits<double>::quiet_NaN();
}

inline half nan(unsigned short nancode) {
  return std::numeric_limits<half>::quiet_NaN();
}

template <detail::NonScalar NonScalar> auto nan(NonScalar nancode) {
  using ElementT = typename NonScalar::value_type;
  if constexpr (std::is_same_v<ElementT, unsigned int> ||
                std::is_same_v<ElementT, std::uint32_t>) {
    using ReturnT =
        typename detail::switch_element_type<NonScalar, float>::type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = std::numeric_limits<float>::quiet_NaN();
    return result;
  } else if constexpr (std::is_same_v<ElementT, unsigned long> ||
                       std::is_same_v<ElementT, std::uint64_t>) {
    using ReturnT =
        typename detail::switch_element_type<NonScalar, double>::type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = std::numeric_limits<double>::quiet_NaN();
    return result;
  } else if constexpr (std::is_same_v<ElementT, unsigned short> ||
                       std::is_same_v<ElementT, std::uint16_t>) {
    using ReturnT = typename detail::switch_element_type<NonScalar, half>::type;
    ReturnT result;
    for (std::size_t i = 0; i < result.size(); i++)
      result[i] = std::numeric_limits<half>::quiet_NaN();
    return result;
  } else {
    static_assert(std::is_same_v<ElementT, void>,
                  "nan() is only supported for unsigned int, unsigned long, "
                  "and unsigned short types.");
  }
}

inline float pown(float x, int y) { return std::pow(x, static_cast<float>(y)); }

inline double pown(double x, int y) {
  return std::pow(x, static_cast<double>(y));
}

inline half pown(half x, int y) {
  return half{std::pow(static_cast<float>(x), static_cast<float>(y))};
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto pown(NonScalar1 x, NonScalar2 y) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = pown(x[i], y[i]);
  return result;
}

template <typename Ptr> float remquo(float x, float y, Ptr quo) {
  int q;
  const float r = std::remquo(x, y, &q);
  *quo = q;
  return r;
}

template <typename Ptr> double remquo(double x, double y, Ptr quo) {
  int q;
  const double r = std::remquo(x, y, &q);
  *quo = q;
  return r;
}

template <typename Ptr> half remquo(half x, half y, Ptr quo) {
  half r = std::remainder(static_cast<float>(x), static_cast<float>(y));
  float float_n =
      (static_cast<float>(x) - static_cast<float>(r)) / static_cast<float>(y);
  int q = static_cast<int>(std::round(float_n));
  *quo = q;
  return half(r);
}

template <typename NonScalar1, typename NonScalar2, typename Ptr>
auto remquo(NonScalar1 x, NonScalar2 y, Ptr quo) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = remquo(x[i], y[i], &(*quo)[i]);
  return result;
}

inline float rootn(float x, int n) { return std::pow(x, 1.0f / n); }

inline double rootn(double x, int n) { return std::pow(x, 1.0 / n); }

inline half rootn(half x, int n) {
  return half{std::pow(static_cast<float>(x), 1.0f / static_cast<float>(n))};
}

template <detail::NonScalar NonScalar1, detail::NonScalar NonScalar2>
auto rootn(NonScalar1 x, NonScalar2 n) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar1>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = rootn(x[i], n[i]);
  return result;
}

template <typename Ptr> float sincos(float x, Ptr cosval) {
  *cosval = std::cos(x);
  return std::sin(x);
}

template <typename Ptr> double sincos(double x, Ptr cosval) {
  *cosval = std::cos(x);
  return std::sin(x);
}

template <typename Ptr> half sincos(half x, Ptr cosval) {
  *cosval = half(std::cos(static_cast<float>(x)));
  return std::sin(static_cast<float>(x));
}

template <detail::NonScalar NonScalar, typename Ptr>
auto sincos(NonScalar x, Ptr cosval) {
  using ReturnT = typename detail::non_scalar_return_type<NonScalar>::type;
  ReturnT result;
  for (std::size_t i = 0; i < x.size(); i++)
    result[i] = sincos(x[i], &(*cosval)[i]);
  return result;
}

} // namespace sycl
