#pragma once

#include <type_traits>

#include "marray.hpp"
#include "math.hpp"
#include "vec.hpp"

namespace sycl {

template <detail::GenericIntegerType GenInt> auto abs(GenInt x) {
  if constexpr (detail::is_vec_v<GenInt> || detail::is_marray_v<GenInt>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt>::type;
    using ValueT = typename GenInt::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = sycl::abs(static_cast<ValueT>(x[i]));
    return result;
  } else {
    return GenInt(x < 0 ? -x : x);
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto abs_diff(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "abs_diff requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] =
          abs_diff(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    return static_cast<GenInt1>(x > y ? x - y : y - x);
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto add_sat(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "add_sat requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = add_sat(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    if (y > 0 && x > std::numeric_limits<GenInt1>::max() - y)
      return std::numeric_limits<GenInt1>::max();
    if (y < 0 && x < std::numeric_limits<GenInt1>::min() - y)
      return std::numeric_limits<GenInt1>::min();
    return GenInt1(x + y);
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto hadd(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "hadd requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = hadd(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    return GenInt1((x & y) + ((x ^ y) >> 1));
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto rhadd(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "hadd requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = rhadd(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    return GenInt1((x | y) - ((x ^ y) >> 1));
  }
}

template <detail::GenericIntegerType GenInt> auto clz(GenInt x) {
  if constexpr (detail::is_vec_v<GenInt> || detail::is_marray_v<GenInt>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt>::type;
    using ValueT = typename GenInt::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = clz(static_cast<ValueT>(x[i]));
    return result;
  } else {
    using UnsignedT = std::make_unsigned_t<GenInt>;
    return static_cast<GenInt>(std::countl_zero(static_cast<UnsignedT>(x)));
  }
}

template <detail::GenericIntegerType GenInt> auto ctz(GenInt x) {
  if constexpr (detail::is_vec_v<GenInt> || detail::is_marray_v<GenInt>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt>::type;
    using ValueT = typename GenInt::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = ctz(static_cast<ValueT>(x[i]));
    return result;
  } else {
    using UnsignedT = std::make_unsigned_t<GenInt>;
    return static_cast<GenInt>(std::countr_zero(static_cast<UnsignedT>(x)));
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto mul_hi(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "mul_hi requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = mul_hi(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    using WideInt = std::conditional_t<
        std::is_signed_v<GenInt1>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(int32_t)), int64_t,
                           __int128>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(uint32_t)), uint64_t,
                           unsigned __int128>>;
    return static_cast<GenInt1>(
        (static_cast<WideInt>(x) * static_cast<WideInt>(y)) >>
        (sizeof(GenInt1) * 8));
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2,
          detail::GenericIntegerType GenInt3>
auto mad_hi(GenInt1 a, GenInt2 b, GenInt3 c) {
  static_assert(std::is_same_v<GenInt1, GenInt2> &&
                    std::is_same_v<GenInt1, GenInt3>,
                "mad_hi requires all arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < a.size(); i++)
      result[i] = mad_hi(static_cast<ValueT>(a[i]), static_cast<ValueT>(b[i]),
                         static_cast<ValueT>(c[i]));
    return result;
  } else {
    using WideInt = std::conditional_t<
        std::is_signed_v<GenInt1>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(int32_t)), int64_t,
                           __int128>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(uint32_t)), uint64_t,
                           unsigned __int128>>;
    return static_cast<GenInt1>(
        ((static_cast<WideInt>(a) * static_cast<WideInt>(b)) >>
         (sizeof(GenInt1) * 8)) +
        static_cast<WideInt>(c));
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2,
          detail::GenericIntegerType GenInt3>
auto mad_sat(GenInt1 a, GenInt2 b, GenInt3 c) {
  static_assert(std::is_same_v<GenInt1, GenInt2> &&
                    std::is_same_v<GenInt1, GenInt3>,
                "mad_sat requires all arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < a.size(); i++)
      result[i] = mad_sat(static_cast<ValueT>(a[i]), static_cast<ValueT>(b[i]),
                          static_cast<ValueT>(c[i]));
    return result;
  } else {
    using WideInt = std::conditional_t<
        std::is_signed_v<GenInt1>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(int32_t)), int64_t,
                           __int128>,
        std::conditional_t<(sizeof(GenInt1) <= sizeof(uint32_t)), uint64_t,
                           unsigned __int128>>;
    const WideInt result = static_cast<WideInt>(a) * static_cast<WideInt>(b) +
                           static_cast<WideInt>(c);
    if (result > std::numeric_limits<GenInt1>::max())
      return std::numeric_limits<GenInt1>::max();
    if (result < std::numeric_limits<GenInt1>::min())
      return std::numeric_limits<GenInt1>::min();
    return static_cast<GenInt1>(result);
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto rotate(GenInt1 v, GenInt2 count) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "rotate requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < v.size(); i++)
      result[i] =
          rotate(static_cast<ValueT>(v[i]), static_cast<ValueT>(count[i]));
    return result;
  } else {
    constexpr int num_bits = sizeof(GenInt1) * 8;
    int i = static_cast<int>(count) % num_bits;
    if (i < 0)
      i += num_bits;
    if (i == 0)
      return static_cast<GenInt1>(v);
    using UnsignedT = std::make_unsigned_t<GenInt1>;
    const UnsignedT uv = static_cast<UnsignedT>(v);
    const UnsignedT result = (uv << i) | (uv >> (num_bits - i));
    return static_cast<GenInt1>(result);
  }
}

template <detail::GenericIntegerType GenInt1,
          detail::GenericIntegerType GenInt2>
auto sub_sat(GenInt1 x, GenInt2 y) {
  static_assert(std::is_same_v<GenInt1, GenInt2>,
                "sub_sat requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<GenInt1> || detail::is_marray_v<GenInt1>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt1>::type;
    using ValueT = typename GenInt1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = sub_sat(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    if (y > 0 && x < std::numeric_limits<GenInt1>::min() + y)
      return std::numeric_limits<GenInt1>::min();
    if (y < 0 && x > std::numeric_limits<GenInt1>::max() + y)
      return std::numeric_limits<GenInt1>::max();
    return GenInt1(x - y);
  }
}

template <detail::UInt8Bit UInt8Bit1, detail::UInt8Bit UInt8Bit2>
auto upsample(UInt8Bit1 hi, UInt8Bit2 lo) {
  static_assert(std::is_same_v<UInt8Bit1, UInt8Bit2>,
                "upsample requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<UInt8Bit1> || detail::is_marray_v<UInt8Bit1>) {
    constexpr int N = detail::non_scalar_return_type<UInt8Bit1>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<UInt8Bit1>, vec<std::uint16_t, N>,
                           marray<std::uint16_t, N>>;
    using ValueT = typename UInt8Bit1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<ValueT>(lo[i]));
    return result;
  } else {
    const std::uint16_t hi_u = static_cast<std::uint16_t>(hi);
    const std::uint16_t lo_u = static_cast<std::uint16_t>(lo);
    return static_cast<std::uint16_t>((hi_u << 8) | lo_u);
  }
}

template <detail::Int8Bit Int8Bit, detail::UInt8Bit UInt8Bit>
auto upsample(Int8Bit hi, UInt8Bit lo) {
  if constexpr (detail::is_vec_v<Int8Bit> || detail::is_marray_v<Int8Bit>) {
    constexpr int N = detail::non_scalar_return_type<Int8Bit>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<Int8Bit>, vec<std::int16_t, N>,
                           marray<std::int16_t, N>>;
    using ValueT = typename Int8Bit::value_type;
    using UValueT = typename UInt8Bit::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<UValueT>(lo[i]));
    return result;
  } else {
    const std::uint16_t hi_u = static_cast<std::int16_t>(hi);
    const std::uint16_t lo_u = static_cast<std::uint16_t>(lo);
    return static_cast<std::int16_t>((hi_u << 8) | lo_u);
  }
}

template <detail::UInt16Bit UInt16Bit1, detail::UInt16Bit UInt16Bit2>
auto upsample(UInt16Bit1 hi, UInt16Bit2 lo) {
  static_assert(std::is_same_v<UInt16Bit1, UInt16Bit2>,
                "upsample requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<UInt16Bit1> ||
                detail::is_marray_v<UInt16Bit1>) {
    constexpr int N = detail::non_scalar_return_type<UInt16Bit1>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<UInt16Bit1>, vec<std::uint32_t, N>,
                           marray<std::uint32_t, N>>;
    using ValueT = typename UInt16Bit1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<ValueT>(lo[i]));
    return result;
  } else {
    const std::uint32_t hi_u = static_cast<std::uint32_t>(hi);
    const std::uint32_t lo_u = static_cast<std::uint32_t>(lo);
    return static_cast<std::uint32_t>((hi_u << 16) | lo_u);
  }
}

template <detail::Int16Bit Int16Bit, detail::UInt16Bit UInt16Bit>
auto upsample(Int16Bit hi, UInt16Bit lo) {
  if constexpr (detail::is_vec_v<Int16Bit> || detail::is_marray_v<Int16Bit>) {
    constexpr int N = detail::non_scalar_return_type<Int16Bit>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<Int16Bit>, vec<std::int32_t, N>,
                           marray<std::int32_t, N>>;
    using ValueT = typename Int16Bit::value_type;
    using UValueT = typename UInt16Bit::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<UValueT>(lo[i]));
    return result;
  } else {
    const std::uint32_t hi_u = static_cast<std::int32_t>(hi);
    const std::uint32_t lo_u = static_cast<std::uint32_t>(lo);
    return static_cast<std::int32_t>((hi_u << 16) | lo_u);
  }
}

template <detail::UInt32Bit UInt32Bit1, detail::UInt32Bit UInt32Bit2>
auto upsample(UInt32Bit1 hi, UInt32Bit2 lo) {
  static_assert(std::is_same_v<UInt32Bit1, UInt32Bit2>,
                "upsample requires both arguments to be of the same type");
  if constexpr (detail::is_vec_v<UInt32Bit1> ||
                detail::is_marray_v<UInt32Bit1>) {
    constexpr int N = detail::non_scalar_return_type<UInt32Bit1>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<UInt32Bit1>, vec<std::uint64_t, N>,
                           marray<std::uint64_t, N>>;
    using ValueT = typename UInt32Bit1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<ValueT>(lo[i]));
    return result;
  } else {
    const std::uint64_t hi_u = static_cast<std::uint64_t>(hi);
    const std::uint64_t lo_u = static_cast<std::uint64_t>(lo);
    return static_cast<std::uint64_t>((hi_u << 32) | lo_u);
  }
}

template <detail::Int32Bit Int32Bit, detail::UInt32Bit UInt32Bit>
auto upsample(Int32Bit hi, UInt32Bit lo) {
  if constexpr (detail::is_vec_v<Int32Bit> || detail::is_marray_v<Int32Bit>) {
    constexpr int N = detail::non_scalar_return_type<Int32Bit>::count;
    using ReturnT =
        std::conditional_t<detail::is_vec_v<Int32Bit>, vec<std::int64_t, N>,
                           marray<std::int64_t, N>>;
    using ValueT = typename Int32Bit::value_type;
    using UValueT = typename UInt32Bit::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < hi.size(); i++)
      result[i] =
          upsample(static_cast<ValueT>(hi[i]), static_cast<UValueT>(lo[i]));
    return result;
  } else {
    const std::uint64_t hi_u = static_cast<std::int64_t>(hi);
    const std::uint64_t lo_u = static_cast<std::uint64_t>(lo);
    return static_cast<std::int64_t>((hi_u << 32) | lo_u);
  }
}

template <detail::GenericIntegerType GenInt> auto popcount(GenInt x) {
  if constexpr (detail::is_vec_v<GenInt> || detail::is_marray_v<GenInt>) {
    using ReturnT = typename detail::non_scalar_return_type<GenInt>::type;
    using ValueT = typename GenInt::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = popcount(static_cast<ValueT>(x[i]));
    return result;
  } else {
    GenInt count = 0;
    for (GenInt bit = 1; bit != 0; bit <<= 1)
      if (x & bit)
        count++;
    return count;
  }
}

template <detail::GenInt32Bit Int32Bit1, detail::GenInt32Bit Int32Bit2,
          detail::GenInt32Bit Int32Bit3>
auto mad24(Int32Bit1 x, Int32Bit2 y, Int32Bit3 z) {
  if constexpr (detail::is_vec_v<Int32Bit1> || detail::is_marray_v<Int32Bit1>) {
    using ReturnT = typename detail::non_scalar_return_type<Int32Bit1>::type;
    using ValueT = typename Int32Bit1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = mad24(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]),
                        static_cast<ValueT>(z[i]));
    return result;
  } else {
    return Int32Bit1{x * y + z};
  }
}

template <detail::GenInt32Bit Int32Bit1, detail::GenInt32Bit Int32Bit2>
auto mul24(Int32Bit1 x, Int32Bit2 y) {
  if constexpr (detail::is_vec_v<Int32Bit1> || detail::is_marray_v<Int32Bit1>) {
    using ReturnT = typename detail::non_scalar_return_type<Int32Bit1>::type;
    using ValueT = typename Int32Bit1::value_type;
    ReturnT result;
    for (std::size_t i = 0; i < x.size(); i++)
      result[i] = mul24(static_cast<ValueT>(x[i]), static_cast<ValueT>(y[i]));
    return result;
  } else {
    return Int32Bit1{x * y};
  }
}

} // namespace sycl
