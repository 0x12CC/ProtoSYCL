#pragma once

#include "math.hpp"

namespace sycl {

namespace detail {

template <typename T>
concept GenericGeometricType =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, half> ||
    (detail::is_marray_v<T> &&
     (std::is_same_v<typename T::value_type, float> ||
      std::is_same_v<typename T::value_type, double> ||
      std::is_same_v<typename T::value_type, half>)) ||
    (detail::is_vec_v<T> && (std::is_same_v<typename T::value_type, float> ||
                             std::is_same_v<typename T::value_type, double> ||
                             std::is_same_v<typename T::value_type, half>));

template <typename T>
concept FloatGeometricType =
    std::is_same_v<T, float> ||
    (detail::is_marray_v<T> && std::is_same_v<typename T::value_type, float>) ||
    (detail::is_vec_v<T> && std::is_same_v<typename T::value_type, float>);

template <typename T>
concept Geometric3or4Type =
    (detail::is_marray_v<T> && (T::size() == 3 || T::size() == 4) &&
     (std::is_same_v<typename T::value_type, float> ||
      std::is_same_v<typename T::value_type, double> ||
      std::is_same_v<typename T::value_type, half>)) ||
    (detail::is_vec_v<T> && (T::size() == 3 || T::size() == 4) &&
     (std::is_same_v<typename T::value_type, float> ||
      std::is_same_v<typename T::value_type, double> ||
      std::is_same_v<typename T::value_type, half>));

} // namespace detail

template <detail::Geometric3or4Type Geo3or4Float1,
          detail::Geometric3or4Type Geo3or4Float2>
auto cross(Geo3or4Float1 p0, Geo3or4Float2 p1) {
  if constexpr (Geo3or4Float1::size() == 4) {
    return Geo3or4Float1{p0[1] * p1[2] - p0[2] * p1[1],
                         p0[2] * p1[0] - p0[0] * p1[2],
                         p0[0] * p1[1] - p0[1] * p1[0], 0};
  } else {
    return Geo3or4Float1{p0[1] * p1[2] - p0[2] * p1[1],
                         p0[2] * p1[0] - p0[0] * p1[2],
                         p0[0] * p1[1] - p0[1] * p1[0]};
  }
}

template <detail::GenericGeometricType GeoFloat1,
          detail::GenericGeometricType GeoFloat2>
auto dot(GeoFloat1 p0, GeoFloat2 p1) {
  if constexpr (detail::is_marray_v<GeoFloat1> || detail::is_vec_v<GeoFloat1>) {
    using ReturnT = GeoFloat1::value_type;
    ReturnT result = 0;
    for (std::size_t i = 0; i < p0.size(); i++)
      result += p0[i] * p1[i];
    return result;
  } else {
    return p0 * p1;
  }
}

template <detail::GenericGeometricType GeoFloat1,
          detail::GenericGeometricType GeoFloat2>
auto distance(GeoFloat1 p0, GeoFloat2 p1) {
  if constexpr (detail::is_marray_v<GeoFloat1> || detail::is_vec_v<GeoFloat1>) {
    using ReturnT = typename detail::non_scalar_return_type<GeoFloat1>::type;
    ReturnT result;
    for (std::size_t i = 0; i < p0.size(); i++)
      result[i] = p0[i] - p1[i];
    return std::sqrt(dot(result, result));
  } else {
    return std::sqrt(dot(p0 - p1, p0 - p1));
  }
}

template <detail::GenericGeometricType GeoFloat> auto length(GeoFloat p) {
  return std::sqrt(dot(p, p));
}

template <detail::GenericGeometricType GeoFloat> auto normalize(GeoFloat p) {
  if constexpr (detail::is_marray_v<GeoFloat> || detail::is_vec_v<GeoFloat>) {
    using ReturnT = typename detail::non_scalar_return_type<GeoFloat>::type;
    ReturnT result;
    auto len = length(p);
    for (std::size_t i = 0; i < p.size(); i++)
      result[i] = p[i] / len;
    return result;
  } else {
    return p / length(p);
  }
}

template <detail::GenericGeometricType GeoFloat1,
          detail::GenericGeometricType GeoFloat2>
auto fast_distance(GeoFloat1 p0, GeoFloat2 p1) {
  return distance(p0, p1);
}

template <detail::GenericGeometricType GeoFloat> auto fast_length(GeoFloat p) {
  return length(p);
}

template <detail::GenericGeometricType GeoFloat>
auto fast_normalize(GeoFloat p) {
  return normalize(p);
}

} // namespace sycl
