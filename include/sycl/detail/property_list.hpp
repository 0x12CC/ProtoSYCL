#pragma once

#include <any>
#include <type_traits>
#include <vector>

namespace sycl {

class property_list;

namespace detail {
template <typename Property> bool has_property(const property_list &) noexcept;
template <typename Property> Property get_property(const property_list &);
} // namespace detail

template <typename Property> struct is_property : std::false_type {};

template <typename Property>
inline constexpr bool is_property_v = is_property<Property>::value;

template <typename Property, typename SyclObject> struct is_property_of;

template <typename Property, typename SyclObject>
inline constexpr bool is_property_of_v =
    is_property_of<Property, SyclObject>::value;

class property_list {
public:
  template <typename... Properties>
  property_list(Properties... props)
    requires(is_property_v<Properties> && ...)
      : m_elements{props...} {}

private:
  template <typename Property>
  friend bool detail::has_property(const property_list &) noexcept;
  template <typename Property>
  friend Property detail::get_property(const property_list &);

  std::vector<std::any> m_elements;
};

} // namespace sycl
