#include "sycl/detail/stream.hpp"

namespace sycl {

const stream &operator<<(const stream &os, const stream_manipulator &rhs) {
  using CharT = std::ostream::char_type;
  using Traits = std::ostream::traits_type;
  if (rhs == stream_manipulator::flush)
    os << std::flush<CharT, Traits>;
  else if (rhs == stream_manipulator::dec)
    os << std::dec;
  else if (rhs == stream_manipulator::hex)
    os << std::hex;
  else if (rhs == stream_manipulator::oct)
    os << std::oct;
  else if (rhs == stream_manipulator::noshowbase)
    os << std::noshowbase;
  else if (rhs == stream_manipulator::showbase)
    os << std::showbase;
  else if (rhs == stream_manipulator::noshowpos)
    os << std::noshowpos;
  else if (rhs == stream_manipulator::showpos)
    os << std::showpos;
  else if (rhs == stream_manipulator::endl)
    os << std::endl<CharT, Traits>;
  else if (rhs == stream_manipulator::fixed)
    os << std::fixed;
  else if (rhs == stream_manipulator::scientific)
    os << std::scientific;
  else if (rhs == stream_manipulator::hexfloat)
    os << std::hexfloat;
  else if (rhs == stream_manipulator::defaultfloat)
    os << std::defaultfloat;
  return os;
}

} // namespace sycl
