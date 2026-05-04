#pragma once

#include <type_traits>

namespace sycl {

template <typename T> class specialization_id;

namespace detail {
template <typename T>
T get_specialization_default_value(const specialization_id<T> &);
}

template <typename T> class specialization_id {
public:
  using value_type = T;

  template <class... Args>
  explicit constexpr specialization_id(Args &&...args)
    requires std::is_constructible_v<T, Args...>
      : m_value{args...} {}

  specialization_id(const specialization_id &rhs) = delete;
  specialization_id(specialization_id &&rhs) = delete;
  specialization_id &operator=(const specialization_id &rhs) = delete;
  specialization_id &operator=(specialization_id &&rhs) = delete;

private:
  friend T detail::get_specialization_default_value(const specialization_id &);
  const T m_value;
};

template <typename T>
T detail::get_specialization_default_value(const specialization_id<T> &id) {
  return id.m_value;
}

} // namespace sycl
