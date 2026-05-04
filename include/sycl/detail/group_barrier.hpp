#pragma once

#include "group.hpp"

namespace sycl {

template <typename Group>
void group_barrier(Group g, memory_scope scope = Group::fence_scope) {
  std::barrier<> *barrier = detail::get_barrier(g);
  barrier->arrive_and_wait();
}

} // namespace sycl
