#pragma once

#include <limits>

namespace sycl {

class half {
public:
  constexpr half() : m_value{static_cast<_Float16>(0.0f)} {}
  constexpr half(float value) : m_value{static_cast<_Float16>(value)} {}

  operator float() const { return m_value; }

  half &operator+=(const half &rhs) {
    m_value += rhs.m_value;
    return *this;
  }

  half &operator-=(const half &rhs) {
    m_value -= rhs.m_value;
    return *this;
  }

  half &operator*=(const half &rhs) {
    m_value *= rhs.m_value;
    return *this;
  }

  half &operator/=(const half &rhs) {
    m_value /= rhs.m_value;
    return *this;
  }

  half &operator++() {
    ++m_value;
    return *this;
  }

  half &operator--() {
    --m_value;
    return *this;
  }

  half operator++(int) {
    half temp = *this;
    ++m_value;
    return temp;
  }

  half operator--(int) {
    half temp = *this;
    --m_value;
    return temp;
  }

  half operator+() { return *this; }

  half operator-() { return half{-m_value}; }

private:
  _Float16 m_value;
};

} // namespace sycl

template <>
struct std::numeric_limits<sycl::half> : std::numeric_limits<float> {};
