#pragma once

#include "id.hpp"
#include "range.hpp"

namespace sycl::detail {

template <int Dimensions>
std::size_t linearize(id<Dimensions> id, range<Dimensions> range) {
  if constexpr (Dimensions == 1)
    return id[0];
  if constexpr (Dimensions == 2)
    return (id[0] * range[1]) + id[1];
  if constexpr (Dimensions == 3)
    return (id[0] * range[1] * range[2]) + (id[1] * range[2]) + id[2];
}

template <int Dimensions>
id<Dimensions> unlinearize(std::size_t id, range<Dimensions> range) {
  if constexpr (Dimensions == 1)
    return {id};
  else if constexpr (Dimensions == 2) {
    std::size_t id0 = id / range[1];
    std::size_t id1 = id % range[1];
    return {id0, id1};
  } else if constexpr (Dimensions == 3) {
    std::size_t id0 = id / (range[1] * range[2]);
    std::size_t rem = id % (range[1] * range[2]);
    std::size_t id1 = rem / range[2];
    std::size_t id2 = rem % range[2];
    return {id0, id1, id2};
  }
}

} // namespace sycl::detail
