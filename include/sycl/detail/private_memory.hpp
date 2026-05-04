#pragma once

#include <vector>

#include "group.hpp"
#include "h_item.hpp"

namespace sycl {

/* Deprecated in SYCL 2020 */
template <typename T, int Dimensions = 1> class private_memory {
public:
  // Construct based directly off the number of work-items
  private_memory(const group<Dimensions> &group)
      : m_values(group.get_local_range().size(), T{}) {};

  // Access the instance for the current work-item
  T &operator()(const h_item<Dimensions> &id) {
    return m_values[id.get_local().get_linear_id()];
  }

private:
  std::vector<T> m_values;
};

} // namespace sycl
