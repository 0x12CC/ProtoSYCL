#pragma once

#include <array>
#include <cstddef>

namespace sycl {

template <int Dimensions> class range {
public:
  static constexpr int dimensions = Dimensions;

  range()
    requires(1 <= Dimensions && Dimensions <= 3)
      : dims{} {}

  range(std::size_t dim0)
    requires(Dimensions == 1)
      : dims{dim0} {}

  range(std::size_t dim0, std::size_t dim1)
    requires(Dimensions == 2)
      : dims{dim0, dim1} {}

  range(std::size_t dim0, std::size_t dim1, std::size_t dim2)
    requires(Dimensions == 3)
      : dims{dim0, dim1, dim2} {}

  friend bool operator==(const range &, const range &) = default;

  std::size_t get(int dimension) const { return dims[dimension]; }
  std::size_t &operator[](int dimension) { return dims[dimension]; }
  std::size_t operator[](int dimension) const { return dims[dimension]; }

  std::size_t size() const {
    if constexpr (Dimensions == 1)
      return dims[0];
    if constexpr (Dimensions == 2)
      return dims[0] * dims[1];
    if constexpr (Dimensions == 3)
      return dims[0] * dims[1] * dims[2];
  }

#define SYCL_OP(OP)                                                            \
  friend range operator OP(const range &lhs, const range &rhs) {               \
    range result;                                                              \
    result.dims[0] = lhs.dims[0] OP rhs.dims[0];                               \
    if constexpr (Dimensions >= 2)                                             \
      result.dims[1] = lhs.dims[1] OP rhs.dims[1];                             \
    if constexpr (Dimensions == 3)                                             \
      result.dims[2] = lhs.dims[2] OP rhs.dims[2];                             \
    return result;                                                             \
  }                                                                            \
  friend range operator OP(const range &lhs, const std::size_t &rhs) {         \
    range result;                                                              \
    result.dims[0] = lhs.dims[0] OP rhs;                                       \
    if constexpr (Dimensions >= 2)                                             \
      result.dims[1] = lhs.dims[1] OP rhs;                                     \
    if constexpr (Dimensions == 3)                                             \
      result.dims[2] = lhs.dims[2] OP rhs;                                     \
    return result;                                                             \
  }                                                                            \
  friend range operator OP(const std::size_t &lhs, const range &rhs) {         \
    range result;                                                              \
    result.dims[0] = lhs OP rhs.dims[0];                                       \
    if constexpr (Dimensions >= 2)                                             \
      result.dims[1] = lhs OP rhs.dims[1];                                     \
    if constexpr (Dimensions == 3)                                             \
      result.dims[2] = lhs OP rhs.dims[2];                                     \
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
  friend range &operator OP(range & lhs, const range & rhs) {                  \
    lhs.dims[0] OP rhs.dims[0];                                                \
    if constexpr (Dimensions >= 2)                                             \
      lhs.dims[1] OP rhs.dims[1];                                              \
    if constexpr (Dimensions == 3)                                             \
      lhs.dims[2] OP rhs.dims[2];                                              \
    return lhs;                                                                \
  }                                                                            \
  friend range &operator OP(range & lhs, const std::size_t & rhs) {            \
    lhs.dims[0] OP rhs;                                                        \
    if constexpr (Dimensions >= 2)                                             \
      lhs.dims[1] OP rhs;                                                      \
    if constexpr (Dimensions == 3)                                             \
      lhs.dims[2] OP rhs;                                                      \
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
  friend range operator OP(const range &rhs) {                                 \
    range result;                                                              \
    result.dims[0] = OP rhs.dims[0];                                           \
    if constexpr (Dimensions >= 2)                                             \
      result.dims[1] = OP rhs.dims[1];                                         \
    if constexpr (Dimensions == 3)                                             \
      result.dims[2] = OP rhs.dims[2];                                         \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend range &operator OP(range & rhs) {                                     \
    rhs.dims[0] OP;                                                            \
    if constexpr (Dimensions >= 2)                                             \
      rhs.dims[1] OP;                                                          \
    if constexpr (Dimensions == 3)                                             \
      rhs.dims[2] OP;                                                          \
    return rhs;                                                                \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend range operator OP(range &rhs, int) {                                  \
    range result = rhs;                                                        \
    OP rhs;                                                                    \
    return result;                                                             \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

private:
  std::array<std::size_t, Dimensions> dims;
};

range(std::size_t) -> range<1>;
range(std::size_t, std::size_t) -> range<2>;
range(std::size_t, std::size_t, std::size_t) -> range<3>;

} // namespace sycl
