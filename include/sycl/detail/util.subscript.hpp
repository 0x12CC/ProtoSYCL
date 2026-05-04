#pragma once

#include "id.hpp"

namespace sycl::detail {

template <typename Target, int TargetDimensions, int SubscriptDimension>
class subscript_proxy;

template <typename Target, int TargetDimensions, int SubscriptDimension>
inline decltype(auto) subscript(Target &target, id<TargetDimensions> id,
                                const std::size_t index) {
  id[SubscriptDimension] = index;
  if constexpr (SubscriptDimension == TargetDimensions - 1)
    return target[id];
  else
    return subscript_proxy<Target, TargetDimensions, SubscriptDimension + 1>{
        target, id};
}

template <typename Target, int TargetDimensions, int SubscriptDimension>
class subscript_proxy {
public:
  subscript_proxy(Target &target, const id<TargetDimensions> id)
      : m_target{target}, m_ident{id} {}

  inline decltype(auto) operator[](std::size_t index) const {
    return subscript<Target, TargetDimensions, SubscriptDimension>(
        m_target, m_ident, index);
  }

private:
  Target &m_target;
  id<TargetDimensions> m_ident;
};

} // namespace sycl::detail
