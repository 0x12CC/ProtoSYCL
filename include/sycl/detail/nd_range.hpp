#pragma once

#include "id.hpp"
#include "range.hpp"

namespace sycl {

template <int Dimensions = 1> class nd_range {
public:
  static constexpr int dimensions = Dimensions;

  // The offset is deprecated in SYCL 2020.
  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize,
           id<Dimensions> offset = id<Dimensions>())
      : m_global_range{globalSize}, m_local_range{localSize}, m_offset{offset} {
  }

  friend bool operator==(const nd_range &, const nd_range &) = default;

  range<Dimensions> get_global_range() const { return m_global_range; }
  range<Dimensions> get_local_range() const { return m_local_range; };
  range<Dimensions> get_group_range() const {
    return m_global_range / m_local_range;
  }
  // Deprecated in SYCL 2020. {
  id<Dimensions> get_offset() const { return m_offset; }

private:
  range<Dimensions> m_global_range;
  range<Dimensions> m_local_range;
  id<Dimensions> m_offset;
};

} // namespace sycl
