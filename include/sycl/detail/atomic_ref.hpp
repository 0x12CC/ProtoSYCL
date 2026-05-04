#pragma once

#include <algorithm>
#include <type_traits>
#include <utility>

#include "access.hpp"
#include "memory_order.hpp"
#include "memory_scope.hpp"

namespace sycl {

namespace detail {

template <typename T>
concept Integral = std::is_integral_v<T>;

template <typename T>
concept Floating = std::is_floating_point_v<T>;

} // namespace detail

template <memory_order ReadModifyWriteOrder> struct memory_order_traits;

template <> struct memory_order_traits<memory_order::relaxed> {
  static constexpr memory_order read_order = memory_order::relaxed;
  static constexpr memory_order write_order = memory_order::relaxed;
};

template <> struct memory_order_traits<memory_order::acq_rel> {
  static constexpr memory_order read_order = memory_order::acquire;
  static constexpr memory_order write_order = memory_order::release;
};

template <> struct memory_order_traits<memory_order::seq_cst> {
  static constexpr memory_order read_order = memory_order::seq_cst;
  static constexpr memory_order write_order = memory_order::seq_cst;
};

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace =
              access::address_space::generic_space>
class atomic_ref {
public:
  using value_type = T;
  static constexpr std::size_t required_alignment = alignof(value_type);
  static constexpr bool is_always_lock_free =
      std::atomic<value_type>::is_always_lock_free;
  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept { return m_value.is_lock_free(); }

  explicit atomic_ref(T &value) : m_value{value} {}
  atomic_ref(const atomic_ref &) noexcept = default;
  atomic_ref &operator=(const atomic_ref &) = delete;

  void store(T operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    m_value.store(operand, detail::get_std_memory_order(order));
  }

  T operator=(T desired) const noexcept {
    store(desired);
    return desired;
  }

  T load(memory_order order = default_read_order,
         memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.load(detail::get_std_memory_order(order));
  }

  operator T() const noexcept { return load(); }

  T exchange(T operand, memory_order order = default_read_modify_write_order,
             memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.exchange(operand, detail::get_std_memory_order(order));
  }

  bool
  compare_exchange_weak(T &expected, T desired, memory_order success,
                        memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_weak(expected, desired,
                                         detail::get_std_memory_order(success),
                                         detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_weak(T &expected, T desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, scope);
  }

  bool
  compare_exchange_strong(T &expected, T desired, memory_order success,
                          memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_strong(
        expected, desired, detail::get_std_memory_order(success),
        detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_strong(T &expected, T desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

private:
  std::atomic_ref<value_type> m_value;
};

// Partial specialization for integral types
template <detail::Integral Integral, memory_order DefaultOrder,
          memory_scope DefaultScope, access::address_space AddressSpace>
class atomic_ref<Integral, DefaultOrder, DefaultScope, AddressSpace> {
public:
  using value_type = Integral;
  static constexpr std::size_t required_alignment = alignof(value_type);
  static constexpr bool is_always_lock_free =
      std::atomic<value_type>::is_always_lock_free;
  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept { return m_value.is_lock_free(); }

  explicit atomic_ref(Integral &value) : m_value{value} {}
  atomic_ref(const atomic_ref &) noexcept = default;
  atomic_ref &operator=(const atomic_ref &) = delete;

  void store(Integral operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    m_value.store(operand, detail::get_std_memory_order(order));
  }

  Integral operator=(Integral desired) const noexcept {
    store(desired);
    return desired;
  }

  Integral load(memory_order order = default_read_order,
                memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.load(detail::get_std_memory_order(order));
  }

  operator Integral() const noexcept { return load(); }

  Integral exchange(Integral operand,
                    memory_order order = default_read_modify_write_order,
                    memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.exchange(operand, detail::get_std_memory_order(order));
  }

  bool
  compare_exchange_weak(Integral &expected, Integral desired,
                        memory_order success, memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_weak(expected, desired,
                                         detail::get_std_memory_order(success),
                                         detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_weak(Integral &expected, Integral desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool
  compare_exchange_strong(Integral &expected, Integral desired,
                          memory_order success, memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_strong(
        expected, desired, detail::get_std_memory_order(success),
        detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_strong(Integral &expected, Integral desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  using difference_type = value_type;

  Integral fetch_add(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_add(operand, detail::get_std_memory_order(order));
  }

  Integral fetch_sub(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_sub(operand, detail::get_std_memory_order(order));
  }

  Integral fetch_and(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_and(operand, detail::get_std_memory_order(order));
  }

  Integral fetch_or(Integral operand,
                    memory_order order = default_read_modify_write_order,
                    memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_or(operand, detail::get_std_memory_order(order));
  }

  Integral fetch_xor(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_xor(operand, detail::get_std_memory_order(order));
  }

  Integral fetch_min(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    auto value = load(order);
    while (
        !compare_exchange_weak(value, std::min(operand, value), order, order))
      ;
    return value;
  }

  Integral fetch_max(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    auto value = load(order);
    while (
        !compare_exchange_weak(value, std::max(operand, value), order, order))
      ;
    return value;
  }

  Integral operator++(int) const noexcept { return fetch_add(1); }
  Integral operator--(int) const noexcept { return fetch_sub(1); }
  Integral operator++() const noexcept { return fetch_add(1) + 1; }
  Integral operator--() const noexcept { return fetch_sub(1) - 1; }
  Integral operator+=(Integral operand) const noexcept {
    return fetch_add(operand) + operand;
  }
  Integral operator-=(Integral operand) const noexcept {
    return fetch_sub(operand) - operand;
  }
  Integral operator&=(Integral operand) const noexcept {
    return fetch_and(operand) & operand;
  }
  Integral operator|=(Integral operand) const noexcept {
    return fetch_or(operand) | operand;
  }
  Integral operator^=(Integral operand) const noexcept {
    return fetch_xor(operand) ^ operand;
  }

private:
  std::atomic_ref<value_type> m_value;
};

// Partial specialization for floating-point types
template <detail::Floating Floating, memory_order DefaultOrder,
          memory_scope DefaultScope, access::address_space AddressSpace>
class atomic_ref<Floating, DefaultOrder, DefaultScope, AddressSpace> {
public:
  using value_type = Floating;
  static constexpr std::size_t required_alignment = alignof(value_type);
  static constexpr bool is_always_lock_free =
      std::atomic<value_type>::is_always_lock_free;
  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept { return m_value.is_lock_free(); }

  explicit atomic_ref(Floating &value) : m_value{value} {}
  atomic_ref(const atomic_ref &) noexcept = default;
  atomic_ref &operator=(const atomic_ref &) = delete;

  void store(Floating operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    m_value.store(operand, detail::get_std_memory_order(order));
  }

  Floating operator=(Floating desired) const noexcept {
    store(desired);
    return desired;
  }

  Floating load(memory_order order = default_read_order,
                memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.load(detail::get_std_memory_order(order));
  }

  operator Floating() const noexcept { return load(); }

  Floating exchange(Floating operand,
                    memory_order order = default_read_modify_write_order,
                    memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.exchange(operand, detail::get_std_memory_order(order));
  }

  bool
  compare_exchange_weak(Floating &expected, Floating desired,
                        memory_order success, memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_weak(expected, desired,
                                         detail::get_std_memory_order(success),
                                         detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_weak(Floating &expected, Floating desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool
  compare_exchange_strong(Floating &expected, Floating desired,
                          memory_order success, memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_strong(
        expected, desired, detail::get_std_memory_order(success),
        detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_strong(Floating &expected, Floating desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  using difference_type = value_type;

  Floating fetch_add(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_add(operand, detail::get_std_memory_order(order));
  }

  Floating fetch_sub(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_sub(operand, detail::get_std_memory_order(order));
  }

  Floating fetch_min(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    auto value = load(order);
    while (
        !compare_exchange_weak(value, std::min(operand, value), order, order))
      ;
    return value;
  }

  Floating fetch_max(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    auto value = load(order);
    while (
        !compare_exchange_weak(value, std::max(operand, value), order, order))
      ;
    return value;
  }

  Floating operator+=(Floating operand) const noexcept {
    return fetch_add(operand) + operand;
  }
  Floating operator-=(Floating operand) const noexcept {
    return fetch_sub(operand) - operand;
  }

private:
  std::atomic_ref<value_type> m_value;
};

// Partial specialization for pointers
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space AddressSpace>
class atomic_ref<T *, DefaultOrder, DefaultScope, AddressSpace> {
public:
  using value_type = T *;
  using difference_type = std::ptrdiff_t;
  static constexpr std::size_t required_alignment = alignof(value_type);
  static constexpr bool is_always_lock_free =
      std::atomic<value_type>::is_always_lock_free;
  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept { return m_value.is_lock_free(); }

  explicit atomic_ref(T *&value) : m_value{value} {}
  atomic_ref(const atomic_ref &) noexcept = default;
  atomic_ref &operator=(const atomic_ref &) = delete;

  void store(T *operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    m_value.store(operand, detail::get_std_memory_order(order));
  }

  T *operator=(T *desired) const noexcept {
    store(desired);
    return desired;
  }

  T *load(memory_order order = default_read_order,
          memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.load(detail::get_std_memory_order(order));
  }

  operator T *() const noexcept { return load(); }

  T *exchange(T *operand, memory_order order = default_read_modify_write_order,
              memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.exchange(operand, detail::get_std_memory_order(order));
  }

  bool
  compare_exchange_weak(T *&expected, T *desired, memory_order success,
                        memory_order failure,
                        memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_weak(expected, desired,
                                         detail::get_std_memory_order(success),
                                         detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_weak(T *&expected, T *desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool
  compare_exchange_strong(T *&expected, T *desired, memory_order success,
                          memory_order failure,
                          memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    if (failure == memory_order::acq_rel)
      failure = memory_order::acquire;
    return m_value.compare_exchange_strong(
        expected, desired, detail::get_std_memory_order(success),
        detail::get_std_memory_order(failure));
  }

  bool
  compare_exchange_strong(T *&expected, T *desired,
                          memory_order order = default_read_modify_write_order,
                          memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  T *fetch_add(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_add(operand, detail::get_std_memory_order(order));
  }

  T *fetch_sub(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    std::ignore = scope;
    return m_value.fetch_sub(operand, detail::get_std_memory_order(order));
  }

  T *operator++(int) const noexcept { return fetch_add(1); }
  T *operator--(int) const noexcept { return fetch_sub(1); }
  T *operator++() const noexcept { return fetch_add(1) + 1; }
  T *operator--() const noexcept { return fetch_sub(1) - 1; }
  T *operator+=(difference_type operand) const noexcept {
    return fetch_add(operand) + operand;
  }
  T *operator-=(difference_type operand) const noexcept {
    return fetch_sub(operand) - operand;
  }

private:
  std::atomic_ref<value_type> m_value;
};

} // namespace sycl
