#pragma once

#include <atomic>

namespace sycl {

enum class memory_order { relaxed, acquire, release, acq_rel, seq_cst };

inline constexpr auto memory_order_relaxed = memory_order::relaxed;
inline constexpr auto memory_order_acquire = memory_order::acquire;
inline constexpr auto memory_order_release = memory_order::release;
inline constexpr auto memory_order_acq_rel = memory_order::acq_rel;
inline constexpr auto memory_order_seq_cst = memory_order::seq_cst;

namespace detail {

inline std::memory_order get_std_memory_order(const memory_order &order) {
  if (order == memory_order::relaxed)
    return std::memory_order::relaxed;
  else if (order == memory_order::acquire)
    return std::memory_order::acquire;
  else if (order == memory_order::release)
    return std::memory_order::release;
  else if (order == memory_order::acq_rel)
    return std::memory_order::acq_rel;
  else
    return std::memory_order::seq_cst;
}

} // namespace detail

} // namespace sycl
