#pragma once

#include <array>
#include <cstddef>

namespace sycl {

template <int Dimensions, bool WithOffset> class item;
template <int Dimensions> class range;

template <int Dimensions = 1> class id {
public:
  static constexpr int dimensions = Dimensions;

  id() : m_elements{} {}

  id(std::size_t dim0)
    requires(Dimensions == 1)
      : m_elements{dim0} {}

  id(std::size_t dim0, std::size_t dim1)
    requires(Dimensions == 2)
      : m_elements{dim0, dim1} {}

  id(std::size_t dim0, std::size_t dim1, std::size_t dim2)
    requires(Dimensions == 3)
      : m_elements{dim0, dim1, dim2} {}

  friend bool operator==(const id &lhs, const id &rhs) {
    return (lhs.m_elements == rhs.m_elements);
  }

  friend bool operator!=(const id &lhs, const id &rhs) { return !(lhs == rhs); }

  id(const range<Dimensions> &range) {
    if constexpr (Dimensions == 1)
      m_elements = {range[0]};
    if constexpr (Dimensions == 2)
      m_elements = {range[0], range[1]};
    if constexpr (Dimensions == 3)
      m_elements = {range[0], range[1], range[2]};
  }

  template <bool WithOffset>
  id(const item<Dimensions, WithOffset> &item) : id{item.get_id()} {}

  std::size_t get(int dimension) const { return m_elements[dimension]; }
  std::size_t &operator[](int dimension) { return m_elements[dimension]; }
  std::size_t operator[](int dimension) const { return m_elements[dimension]; }

  operator std::size_t() const
    requires(Dimensions == 1)
  {
    return m_elements[0];
  }

#define SYCL_OP(OP)                                                            \
  friend id operator OP(const id &lhs, const id &rhs) {                        \
    id result;                                                                 \
    result.m_elements[0] = lhs.m_elements[0] OP rhs.m_elements[0];             \
    if constexpr (Dimensions >= 2)                                             \
      result.m_elements[1] = lhs.m_elements[1] OP rhs.m_elements[1];           \
    if constexpr (Dimensions == 3)                                             \
      result.m_elements[2] = lhs.m_elements[2] OP rhs.m_elements[2];           \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
    requires std::is_integral_v<T>                                             \
  friend id operator OP(const id &lhs, const T &rhs) {                         \
    id result;                                                                 \
    result.m_elements[0] = lhs.m_elements[0] OP rhs;                           \
    if constexpr (Dimensions >= 2)                                             \
      result.m_elements[1] = lhs.m_elements[1] OP rhs;                         \
    if constexpr (Dimensions == 3)                                             \
      result.m_elements[2] = lhs.m_elements[2] OP rhs;                         \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
    requires std::is_integral_v<T>                                             \
  friend id operator OP(const T &lhs, const id &rhs) {                         \
    id result;                                                                 \
    result.m_elements[0] = lhs OP rhs.m_elements[0];                           \
    if constexpr (Dimensions >= 2)                                             \
      result.m_elements[1] = lhs OP rhs.m_elements[1];                         \
    if constexpr (Dimensions == 3)                                             \
      result.m_elements[2] = lhs OP rhs.m_elements[2];                         \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
  SYCL_OP(*)
  SYCL_OP(/)
  SYCL_OP(%)
  SYCL_OP(<<)
  SYCL_OP(>>)
  SYCL_OP(&)
  SYCL_OP(|)
  SYCL_OP(^)
  SYCL_OP(&&)
  SYCL_OP(||)
  SYCL_OP(<)
  SYCL_OP(>)
  SYCL_OP(<=)
  SYCL_OP(>=)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend id &operator OP(id & lhs, const id & rhs) {                           \
    lhs.m_elements[0] OP rhs.m_elements[0];                                    \
    if constexpr (Dimensions >= 2)                                             \
      lhs.m_elements[1] OP rhs.m_elements[1];                                  \
    if constexpr (Dimensions == 3)                                             \
      lhs.m_elements[2] OP rhs.m_elements[2];                                  \
    return lhs;                                                                \
  }                                                                            \
  friend id &operator OP(id & lhs, const std::size_t & rhs) {                  \
    lhs.m_elements[0] OP rhs;                                                  \
    if constexpr (Dimensions >= 2)                                             \
      lhs.m_elements[1] OP rhs;                                                \
    if constexpr (Dimensions == 3)                                             \
      lhs.m_elements[2] OP rhs;                                                \
    return lhs;                                                                \
  }

  SYCL_OP(+=)
  SYCL_OP(-=)
  SYCL_OP(*=)
  SYCL_OP(/=)
  SYCL_OP(%=)
  SYCL_OP(<<=)
  SYCL_OP(>>=)
  SYCL_OP(&=)
  SYCL_OP(|=)
  SYCL_OP(^=)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend id operator OP(const id &rhs) {                                       \
    id result;                                                                 \
    result.m_elements[0] = OP rhs.m_elements[0];                               \
    if constexpr (Dimensions >= 2)                                             \
      result.m_elements[1] = OP rhs.m_elements[1];                             \
    if constexpr (Dimensions == 3)                                             \
      result.m_elements[2] = OP rhs.m_elements[2];                             \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend id &operator OP(id & rhs) {                                           \
    rhs.m_elements[0] OP;                                                      \
    if constexpr (Dimensions >= 2)                                             \
      rhs.m_elements[1] OP;                                                    \
    if constexpr (Dimensions == 3)                                             \
      rhs.m_elements[2] OP;                                                    \
    return rhs;                                                                \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend id operator OP(id &rhs, int) {                                        \
    id result = rhs;                                                           \
    OP rhs;                                                                    \
    return result;                                                             \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

private:
  std::array<std::size_t, Dimensions> m_elements;
};

id(std::size_t) -> id<1>;
id(std::size_t, std::size_t) -> id<2>;
id(std::size_t, std::size_t, std::size_t) -> id<3>;

} // namespace sycl
