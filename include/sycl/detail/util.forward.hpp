#pragma once

#include <type_traits>

#include "access.hpp"

namespace sycl {

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write),
          target AccessTarget = target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write)>
class host_accessor;

template <typename DataT, int Dimensions = 1> class local_accessor;

class handler;

} // namespace sycl
