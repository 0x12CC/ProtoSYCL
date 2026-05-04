#pragma once

#include <cstdint>

namespace sycl::opencl {

using cl_bool = bool;

using cl_char = std::int8_t;
using cl_uchar = std::uint8_t;

using cl_short = std::int16_t;
using cl_ushort = std::uint16_t;

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;

using cl_long = std::int64_t;
using cl_ulong = std::uint64_t;

using cl_half = _Float16;
using cl_float = float;
using cl_double = double;

}; // namespace sycl::opencl

namespace sycl {

template <typename DataT, int NumElements> class vec;

using cl_int2 = vec<opencl::cl_int, 2>;
using cl_int4 = vec<opencl::cl_int, 4>;

using cl_float2 = vec<opencl::cl_float, 2>;
using cl_float4 = vec<opencl::cl_float, 4>;

using cl_half2 = vec<opencl::cl_half, 2>;
using cl_half4 = vec<opencl::cl_half, 4>;

} // namespace sycl
