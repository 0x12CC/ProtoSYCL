#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace sycl::detail {

template <typename T, typename Accessor> class accessor_iterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = value_type *;
  using reference = value_type &;

  accessor_iterator(const Accessor *accessor = nullptr, std::size_t value = 0)
      : m_accessor{accessor}, m_value{value} {}
  accessor_iterator(const accessor_iterator &other) = default;

  accessor_iterator(
      const accessor_iterator<std::remove_const_t<T>, Accessor> &other)
    requires(std::is_const_v<T>)
      : m_accessor{other.m_accessor}, m_value{other.m_value} {}

  reference operator*() const { return m_accessor->get_nth_value(m_value); }
  reference operator[](difference_type n) const {
    return m_accessor->get_nth_value(m_value + n);
  }

  friend auto operator<=>(const accessor_iterator &,
                          const accessor_iterator &) = default;

  accessor_iterator &operator++() {
    m_value++;
    return *this;
  }
  accessor_iterator operator++(int) {
    accessor_iterator tmp = *this;
    ++(*this);
    return tmp;
  }
  accessor_iterator &operator--() {
    m_value--;
    return *this;
  }
  accessor_iterator operator--(int) {
    accessor_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  friend accessor_iterator operator+(accessor_iterator it, difference_type n) {
    return {it.m_accessor, it.m_value + n};
  }
  friend accessor_iterator operator+(difference_type n, accessor_iterator it) {
    return it + n;
  }

  friend accessor_iterator operator-(accessor_iterator it, difference_type n) {
    return {it.m_accessor, it.m_value - n};
  }
  friend difference_type operator-(accessor_iterator a, accessor_iterator b) {
    return a.m_value - b.m_value;
  }

  friend accessor_iterator &operator+=(accessor_iterator &it,
                                       difference_type n) {
    it.m_value += n;
    return it;
  }
  friend accessor_iterator &operator-=(accessor_iterator &it,
                                       difference_type n) {
    it.m_value -= n;
    return it;
  }

private:
  friend accessor_iterator<const T, Accessor>;

  const Accessor *m_accessor = nullptr;
  std::size_t m_value = 0;
};

} // namespace sycl::detail
