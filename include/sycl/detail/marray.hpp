#pragma once

#include <array>
#include <cstdint>

#include "half.hpp"

namespace sycl {

template <typename DataT, std::size_t NumElements> class marray;

namespace detail {

template <typename T> constexpr bool is_marray_v = false;

template <typename DataT, std::size_t NumElements>
constexpr bool is_marray_v<marray<DataT, NumElements>> = true;

template <typename T>
concept MArray = is_marray_v<T>;

} // namespace detail

template <typename DataT, std::size_t NumElements> class marray {
public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;
  using iterator = DataT *;
  using const_iterator = const DataT *;

  marray() = default;

  explicit constexpr marray(const DataT &arg) {
    for (std::size_t i = 0; i < NumElements; i++)
      m_elements[i] = arg;
  }

  template <typename... ArgTN> constexpr marray(const ArgTN &...args) {
    init_with_offset<0>(args...);
  }

  constexpr marray(const marray<DataT, NumElements> &rhs) = default;
  constexpr marray(marray<DataT, NumElements> &&rhs) = default;

  operator DataT() const
    requires(NumElements == 1)
  {
    return m_elements[0];
  };

  static constexpr std::size_t size() noexcept { return NumElements; }

  // subscript operator
  reference operator[](std::size_t index) { return m_elements[index]; }
  const_reference operator[](std::size_t index) const {
    return m_elements[index];
  }

  marray &operator=(const marray<DataT, NumElements> &rhs) = default;
  marray &operator=(const DataT &rhs) {
    *this = marray{rhs};
    return *this;
  }

  // iterator functions
  iterator begin() { return m_elements.begin(); }
  const_iterator begin() const { return m_elements.begin(); }

  iterator end() { return m_elements.end(); }
  const_iterator end() const { return m_elements.end(); }

#define SYCL_OP(OP)                                                            \
  friend marray operator OP(const marray &lhs, const marray &rhs) {            \
    marray result;                                                             \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  friend marray operator OP(const marray &lhs, const DataT &rhs) {             \
    marray result;                                                             \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs[i] OP rhs;                                               \
    return result;                                                             \
  }                                                                            \
  friend marray operator OP(const DataT &lhs, const marray &rhs) {             \
    marray result;                                                             \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs OP rhs[i];                                               \
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
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend auto operator OP(const marray &lhs, const marray &rhs) {              \
    marray<bool, NumElements> result;                                          \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  friend auto operator OP(const marray &lhs, const DataT &rhs) {               \
    marray<bool, NumElements> result;                                          \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs[i] OP rhs;                                               \
    return result;                                                             \
  }                                                                            \
  friend auto operator OP(const DataT &lhs, const marray &rhs) {               \
    marray<bool, NumElements> result;                                          \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = lhs OP rhs[i];                                               \
    return result;                                                             \
  }

  SYCL_OP(==)
  SYCL_OP(!=)
  SYCL_OP(&&)
  SYCL_OP(||)
  SYCL_OP(<)
  SYCL_OP(>)
  SYCL_OP(<=)
  SYCL_OP(>=)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend marray &operator OP(marray & lhs, const marray & rhs) {               \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      lhs[i] OP rhs[i];                                                        \
    return lhs;                                                                \
  }                                                                            \
  friend marray &operator OP(marray & lhs, const DataT & rhs) {                \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      lhs[i] OP rhs;                                                           \
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
  friend marray operator OP(const marray &rhs) {                               \
    marray result;                                                             \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      result[i] = OP rhs[i];                                                   \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
  SYCL_OP(~)
#undef SYCL_OP

  friend marray<bool, NumElements> operator!(const marray &rhs) {
    marray<bool, NumElements> result;
    for (std::size_t i = 0; i < NumElements; i++)
      result[i] = !rhs[i];
    return result;
  }

#define SYCL_OP(OP)                                                            \
  friend marray &operator OP(marray & rhs) {                                   \
    for (std::size_t i = 0; i < NumElements; i++)                              \
      rhs[i] OP;                                                               \
    return rhs;                                                                \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend marray operator OP(marray &rhs, int) {                                \
    marray result = rhs;                                                       \
    OP rhs;                                                                    \
    return result;                                                             \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

private:
  template <typename, std::size_t> friend class marray;

  template <std::size_t Offset, typename... ArgTN>
    requires(Offset + 1 <= NumElements)
  constexpr void init_with_offset(const DataT &arg, const ArgTN &...args) {
    m_elements[Offset] = arg;
    if constexpr (Offset + 1 < NumElements)
      init_with_offset<Offset + 1>(args...);
  }

  template <std::size_t Offset, std::size_t ArgNumElements, typename... ArgTN>
    requires(Offset + ArgNumElements <= NumElements)
  constexpr void init_with_offset(const marray<DataT, ArgNumElements> &arg,
                                  const ArgTN &...args) {
    for (std::size_t i = 0; i < ArgNumElements; ++i) {
      m_elements[Offset + i] = arg.m_elements[i];
    }
    if constexpr (Offset + ArgNumElements < NumElements)
      init_with_offset<Offset + ArgNumElements>(args...);
  }

  std::array<DataT, NumElements> m_elements{};
};

using mbool2 = marray<bool, 2>;
using mbool3 = marray<bool, 3>;
using mbool4 = marray<bool, 4>;
using mbool8 = marray<bool, 8>;
using mbool16 = marray<bool, 16>;

using mchar2 = marray<std::int8_t, 2>;
using mchar3 = marray<std::int8_t, 3>;
using mchar4 = marray<std::int8_t, 4>;
using mchar8 = marray<std::int8_t, 8>;
using mchar16 = marray<std::int8_t, 16>;

using muchar2 = marray<std::uint8_t, 2>;
using muchar3 = marray<std::uint8_t, 3>;
using muchar4 = marray<std::uint8_t, 4>;
using muchar8 = marray<std::uint8_t, 8>;
using muchar16 = marray<std::uint8_t, 16>;

using mshort2 = marray<std::int16_t, 2>;
using mshort3 = marray<std::int16_t, 3>;
using mshort4 = marray<std::int16_t, 4>;
using mshort8 = marray<std::int16_t, 8>;
using mshort16 = marray<std::int16_t, 16>;

using mushort2 = marray<std::uint16_t, 2>;
using mushort3 = marray<std::uint16_t, 3>;
using mushort4 = marray<std::uint16_t, 4>;
using mushort8 = marray<std::uint16_t, 8>;
using mushort16 = marray<std::uint16_t, 16>;

using mint2 = marray<std::int32_t, 2>;
using mint3 = marray<std::int32_t, 3>;
using mint4 = marray<std::int32_t, 4>;
using mint8 = marray<std::int32_t, 8>;
using mint16 = marray<std::int32_t, 16>;

using muint2 = marray<std::uint32_t, 2>;
using muint3 = marray<std::uint32_t, 3>;
using muint4 = marray<std::uint32_t, 4>;
using muint8 = marray<std::uint32_t, 8>;
using muint16 = marray<std::uint32_t, 16>;

using mlong2 = marray<std::int64_t, 2>;
using mlong3 = marray<std::int64_t, 3>;
using mlong4 = marray<std::int64_t, 4>;
using mlong8 = marray<std::int64_t, 8>;
using mlong16 = marray<std::int64_t, 16>;

using mulong2 = marray<std::uint64_t, 2>;
using mulong3 = marray<std::uint64_t, 3>;
using mulong4 = marray<std::uint64_t, 4>;
using mulong8 = marray<std::uint64_t, 8>;
using mulong16 = marray<std::uint64_t, 16>;

using mhalf2 = marray<half, 2>;
using mhalf3 = marray<half, 3>;
using mhalf4 = marray<half, 4>;
using mhalf8 = marray<half, 8>;
using mhalf16 = marray<half, 16>;

using mfloat2 = marray<float, 2>;
using mfloat3 = marray<float, 3>;
using mfloat4 = marray<float, 4>;
using mfloat8 = marray<float, 8>;
using mfloat16 = marray<float, 16>;

using mdouble2 = marray<double, 2>;
using mdouble3 = marray<double, 3>;
using mdouble4 = marray<double, 4>;
using mdouble8 = marray<double, 8>;
using mdouble16 = marray<double, 16>;

} // namespace sycl
