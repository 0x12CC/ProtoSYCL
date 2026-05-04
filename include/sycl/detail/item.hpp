#pragma once

#include "id.hpp"
#include "range.hpp"
#include "util.linearize.hpp"

namespace sycl {

template <int Dimensions = 1, bool WithOffset = true> class item;

namespace detail {
template <int Dimensions>
item<Dimensions, false> make_item(const id<Dimensions> &,
                                  const range<Dimensions> &);
template <int Dimensions>
item<Dimensions, true> make_item(const id<Dimensions> &,
                                 const range<Dimensions> &,
                                 const id<Dimensions> &);
} // namespace detail

template <int Dimensions, bool WithOffset> class item {
public:
  static constexpr int dimensions = Dimensions;

  item() = delete;

  friend bool operator==(const item &, const item &) = default;

  id<Dimensions> get_id() const { return m_id; }

  std::size_t get_id(int dimension) const { return m_id[dimension]; }

  std::size_t operator[](int dimension) const { return m_id[dimension]; }

  range<Dimensions> get_range() const { return m_range; }

  std::size_t get_range(int dimension) const { return m_range[dimension]; }

  // Deprecated in SYCL 2020.
  id<Dimensions> get_offset() const
    requires WithOffset
  {
    return m_offset;
  }

  // Deprecated in SYCL 2020.
  operator item<Dimensions, true>() const
    requires(WithOffset == false)
  {
    return detail::make_item(m_id, m_range, id<Dimensions>{});
  }

  operator std::size_t() const
    requires(Dimensions == 1)
  {
    return m_id[0];
  }

  std::size_t get_linear_id() const { return detail::linearize(m_id, m_range); }

private:
  item(id<Dimensions> ident, range<Dimensions> range)
      : m_id{ident}, m_range{range} {}
  item(id<Dimensions> ident, range<Dimensions> range, id<Dimensions> offset)
      : m_id{ident}, m_range{range}, m_offset{offset} {}

  friend item<Dimensions, false>
  detail::make_item<Dimensions>(const id<Dimensions> &,
                                const range<Dimensions> &);
  friend item<Dimensions, true>
  detail::make_item<Dimensions>(const id<Dimensions> &,
                                const range<Dimensions> &,
                                const id<Dimensions> &);

  id<Dimensions> m_id;
  range<Dimensions> m_range;
  id<Dimensions> m_offset;
};

template <int Dimensions>
item<Dimensions, false> detail::make_item(const id<Dimensions> &ident,
                                          const range<Dimensions> &range) {
  return item<Dimensions, false>{ident, range};
}

template <int Dimensions>
item<Dimensions, true> detail::make_item(const id<Dimensions> &ident,
                                         const range<Dimensions> &range,
                                         const id<Dimensions> &offset) {
  return item<Dimensions, true>{ident, range, offset};
}

}; // namespace sycl
