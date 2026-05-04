#pragma once

#include "access.hpp"

namespace sycl {

namespace detail {

template <access_mode AccessMode, target Target> struct accessor_tag {
  inline static constexpr ::sycl::access_mode access_mode = AccessMode;
  inline static constexpr ::sycl::target target = Target;
};

} // namespace detail

inline constexpr detail::accessor_tag<access_mode::read, target::device>
    read_only;
inline constexpr detail::accessor_tag<access_mode::read_write, target::device>
    read_write;
inline constexpr detail::accessor_tag<access_mode::write, target::device>
    write_only;
inline constexpr detail::accessor_tag<access_mode::read, target::host_task>
    read_only_host_task;
inline constexpr detail::accessor_tag<access_mode::read_write,
                                      target::host_task>
    read_write_host_task;
inline constexpr detail::accessor_tag<access_mode::write, target::host_task>
    write_only_host_task;

} // namespace sycl
