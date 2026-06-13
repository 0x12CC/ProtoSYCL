#pragma once

#include <algorithm>
#include <atomic>

#include "access.hpp"
#include "memory_order.hpp"
#include "memory_scope.hpp"
#include "multi_ptr.hpp"

namespace sycl {

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace> class atomic {
public:
  template <typename PointerT, access::decorated IsDecorated>
  atomic(multi_ptr<PointerT, AddressSpace, IsDecorated> ptr)
      : m_value{*ptr.get_raw()} {}

  atomic(const atomic &other) : m_value{other.m_value} {}

  void store(T operand, memory_order memoryOrder = memory_order::relaxed) {
    m_value.store(operand, detail::get_std_memory_order(memoryOrder));
  }

  T load(memory_order memoryOrder = memory_order::relaxed) const {
    return m_value.load(detail::get_std_memory_order(memoryOrder));
  }

  T exchange(T operand, memory_order memoryOrder = memory_order::relaxed) {
    return m_value.exchange(operand, detail::get_std_memory_order(memoryOrder));
  }

  bool compare_exchange_strong(
      T &expected, T desired,
      memory_order successMemoryOrder = memory_order::relaxed,
      memory_order failMemoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.compare_exchange_strong(
        expected, desired, detail::get_std_memory_order(successMemoryOrder),
        detail::get_std_memory_order(failMemoryOrder));
  }

  T fetch_add(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.fetch_add(operand,
                             detail::get_std_memory_order(memoryOrder));
  }

  T fetch_sub(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.fetch_sub(operand,
                             detail::get_std_memory_order(memoryOrder));
  }

  T fetch_and(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.fetch_and(operand,
                             detail::get_std_memory_order(memoryOrder));
  }

  T fetch_or(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.fetch_or(operand, detail::get_std_memory_order(memoryOrder));
  }

  T fetch_xor(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.fetch_xor(operand,
                             detail::get_std_memory_order(memoryOrder));
  }

  T fetch_min(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    auto value = load(memoryOrder);
    while (!compare_exchange_weak(value, std::min(operand, value), memoryOrder,
                                  memoryOrder))
      ;
    return value;
  }

  T fetch_max(T operand, memory_order memoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    auto value = load(memoryOrder);
    while (!compare_exchange_weak(value, std::max(operand, value), memoryOrder,
                                  memoryOrder))
      ;
    return value;
  }

private:
  std::atomic_ref<T> m_value;

  bool
  compare_exchange_weak(T &expected, T desired,
                        memory_order successMemoryOrder = memory_order::relaxed,
                        memory_order failMemoryOrder = memory_order::relaxed)
    requires(!std::is_same_v<T, float>)
  {
    return m_value.compare_exchange_weak(
        expected, desired, detail::get_std_memory_order(successMemoryOrder),
        detail::get_std_memory_order(failMemoryOrder));
  }
};

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
void atomic_store(atomic<T, AddressSpace> object, T operand,
                  memory_order memoryOrder = memory_order::relaxed) {
  object.store(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_load(atomic<T, AddressSpace> object,
              memory_order memoryOrder = memory_order::relaxed) {
  return object.load(memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_exchange(atomic<T, AddressSpace> object, T operand,
                  memory_order memoryOrder = memory_order::relaxed) {
  return object.exchange(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
bool atomic_compare_exchange_strong(
    atomic<T, AddressSpace> object, T &expected, T desired,
    memory_order successMemoryOrder = memory_order::relaxed,
    memory_order failMemoryOrder = memory_order::relaxed) {
  return object.compare_exchange_strong(expected, desired, successMemoryOrder,
                                        failMemoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_add(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_add(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_sub(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_sub(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_and(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_and(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_or(atomic<T, AddressSpace> object, T operand,
                  memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_or(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_xor(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_xor(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_min(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_min(operand, memoryOrder);
}

/* Deprecated in SYCL 2020 */
template <typename T, access::address_space AddressSpace>
T atomic_fetch_max(atomic<T, AddressSpace> object, T operand,
                   memory_order memoryOrder = memory_order::relaxed) {
  return object.fetch_max(operand, memoryOrder);
}

inline void atomic_fence(memory_order order, memory_scope scope) {
  std::ignore = scope;
  std::atomic_thread_fence(detail::get_std_memory_order(order));
}

} // namespace sycl
