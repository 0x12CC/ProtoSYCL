#pragma once

#include "exception.hpp"
#include "property_list.hpp"

namespace sycl {

namespace property {

struct no_init {};

namespace queue {
struct enable_profiling {};
struct in_order {};
}; // namespace queue

} // namespace property

inline constexpr property::no_init no_init;

template <> struct is_property<property::no_init> : std::true_type {};

template <>
struct is_property<property::queue::enable_profiling> : std::true_type {};
template <> struct is_property<property::queue::in_order> : std::true_type {};

template <typename Property>
bool detail::has_property(const property_list &list) noexcept {
  for (const auto &prop : list.m_elements)
    if (prop.type() == typeid(Property))
      return true;
  return false;
}

template <typename Property>
Property detail::get_property(const property_list &list) {
  for (const auto &prop : list.m_elements)
    if (prop.type() == typeid(Property))
      return std::any_cast<Property>(prop);
  throw sycl::exception{sycl::errc::invalid};
}

} // namespace sycl
