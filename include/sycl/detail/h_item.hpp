#pragma once

#include "item.hpp"
#include "range.hpp"

namespace sycl {

template <int Dimensions = 1>
  requires(1 <= Dimensions && Dimensions <= 3)
class h_item;

namespace detail {
template <int Dimensions>
h_item<Dimensions> make_h_item(const item<Dimensions, false> &,
                               const item<Dimensions, false> &,
                               const item<Dimensions, false> &);
}

template <int Dimensions>
  requires(1 <= Dimensions && Dimensions <= 3)
class h_item {
public:
  static constexpr int dimensions = Dimensions;

  h_item() = delete;

  friend bool operator==(const h_item &, const h_item &) = default;

  item<Dimensions, false> get_global() const { return m_global; }

  item<Dimensions, false> get_local() const { return get_logical_local(); }

  item<Dimensions, false> get_logical_local() const { return m_logical; }

  item<Dimensions, false> get_physical_local() const { return m_physical; }

  range<Dimensions> get_global_range() const {
    return get_global().get_range();
  }

  std::size_t get_global_range(int dimension) const {
    return get_global_range()[dimension];
  }

  id<Dimensions> get_global_id() const { return get_global().get_id(); }

  std::size_t get_global_id(int dimension) const {
    return get_global_id()[dimension];
  }

  range<Dimensions> get_local_range() const { return get_local().get_range(); }

  std::size_t get_local_range(int dimension) const {
    return get_local_range()[dimension];
  }

  id<Dimensions> get_local_id() const { return get_local().get_id(); }

  std::size_t get_local_id(int dimension) const {
    return get_local_id()[dimension];
  }

  range<Dimensions> get_logical_local_range() const {
    return get_logical_local().get_range();
  }

  std::size_t get_logical_local_range(int dimension) const {
    return get_logical_local_range()[dimension];
  }

  id<Dimensions> get_logical_local_id() const {
    return get_logical_local().get_id();
  }

  std::size_t get_logical_local_id(int dimension) const {
    return get_logical_local_id()[dimension];
  }

  range<Dimensions> get_physical_local_range() const {
    return get_physical_local().get_range();
  }

  std::size_t get_physical_local_range(int dimension) const {
    return get_physical_local_range()[dimension];
  }

  id<Dimensions> get_physical_local_id() const {
    return get_physical_local().get_id();
  }

  std::size_t get_physical_local_id(int dimension) const {
    return get_physical_local_id()[dimension];
  }

private:
  h_item(const item<Dimensions, false> &global,
         const item<Dimensions, false> &logical,
         const item<Dimensions, false> &physical)
      : m_global{global}, m_logical{logical}, m_physical{physical} {}

  friend h_item<Dimensions>
  detail::make_h_item<Dimensions>(const item<Dimensions, false> &,
                                  const item<Dimensions, false> &,
                                  const item<Dimensions, false> &);

  item<Dimensions, false> m_global;
  item<Dimensions, false> m_logical;
  item<Dimensions, false> m_physical;
};

template <int Dimensions>
h_item<Dimensions>
detail::make_h_item(const item<Dimensions, false> &global,
                    const item<Dimensions, false> &logical,
                    const item<Dimensions, false> &physical) {
  return h_item<Dimensions>{global, logical, physical};
}

} // namespace sycl
