#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "access.hpp"
#include "half.hpp"
#include "multi_ptr.hpp"

namespace sycl {

template <typename DataT, int NumElements>
class alignas((NumElements == 3 ? 4 : NumElements) * sizeof(DataT)) vec;

namespace detail {

template <typename DataT>
using vec_logical_ret_t = std::conditional_t<
    sizeof(DataT) == 1, std::int8_t,
    std::conditional_t<
        sizeof(DataT) == 2, std::int16_t,
        std::conditional_t<sizeof(DataT) == 4, std::int32_t, std::int64_t>>>;

template <typename ArgT> struct num_elements_t {
  static constexpr int value = 1;
};

template <typename DataT, int NumElements>
struct num_elements_t<vec<DataT, NumElements>> {
  static constexpr int value = NumElements;
};

template <typename ArgT>
inline constexpr int num_elements_v = num_elements_t<ArgT>::value;

template <typename T>
constexpr bool is_sycl_floating_point =
    (std::is_same_v<T, sycl::half> || std::is_same_v<T, float> ||
     std::is_same_v<T, double>);

template <typename VecT, int NumElements> class __writeable_swizzle__;

template <typename T> constexpr bool is_vec_v = false;

template <typename DataT, std::size_t NumElements>
constexpr bool is_vec_v<vec<DataT, NumElements>> = true;

template <typename DataT, std::size_t NumElements>
constexpr bool
    is_vec_v<__writeable_swizzle__<vec<DataT, NumElements>, NumElements>> =
        true;

template <typename T>
concept Vec = is_vec_v<T>;

} // namespace detail

enum class rounding_mode { automatic, rte, rtz, rtp, rtn };

struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

template <typename DataT, int NumElements>
class alignas((NumElements == 3 ? 4 : NumElements) * sizeof(DataT)) vec {
public:
  using element_type = DataT;
  using value_type = DataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = __unspecified__;
#endif

  vec() = default;

  explicit constexpr vec(const DataT &arg) {
    std::fill(m_elements.begin(), m_elements.end(), arg);
  }

  template <typename... ArgTN>
  constexpr vec(const ArgTN &...args)
    requires((detail::num_elements_v<ArgTN> + ...) == NumElements)
  {
    init_with_offset<0>(args...);
  }

  constexpr vec(const vec &) = default;

#ifdef __SYCL_DEVICE_ONLY__
  vec(vector_t nativeVector);

  operator vector_t() const;
#endif

  operator DataT() const
    requires(NumElements == 1)
  {
    return m_elements[0];
  }

  static constexpr std::size_t byte_size() noexcept {
    return (NumElements == 3 ? 4 : NumElements) * sizeof(DataT);
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  // Deprecated
  std::size_t get_size() const { return byte_size(); }

  // Deprecated
  std::size_t get_count() const { return size(); }

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    vec<ConvertT, NumElements> result{};
    for (int i = 0; i < NumElements; i++)
      if constexpr (detail::is_sycl_floating_point<DataT> &&
                    !detail::is_sycl_floating_point<ConvertT>)
        result[i] = static_cast<ConvertT>(std::rint(m_elements[i]));
      else
        result[i] = static_cast<ConvertT>(m_elements[i]);
    return result;
  }

  template <typename AsT> AsT as() const {
    AsT result{};
    std::memcpy(&result[0], &m_elements[0], byte_size());
    return result;
  }

  template <int N>
  using __writeable_swizzle__ = detail::__writeable_swizzle__<vec, N>;
  template <typename T = DataT, int N = NumElements>
  using __const_swizzle__ = vec<const T, N>;

  // Available on when the number of swizzleIndexes template parameters is
  // 1, 2, 3, 4, 8, or 16.
  // Available only when each of the swizzleIndexes template parameters is
  // greater or equal to 0 and less than NumElements.
  template <int... swizzleIndexes>
  __writeable_swizzle__<sizeof...(swizzleIndexes)> swizzle() {
    constexpr int n = sizeof...(swizzleIndexes);
    return __writeable_swizzle__<n>{*this, std::array{swizzleIndexes...}};
  }

  template <int... swizzleIndexes>
  vec<DataT, sizeof...(swizzleIndexes)> swizzle() const {
    constexpr int n = sizeof...(swizzleIndexes);
    vec<DataT, n> result{};
    std::array<int, n> indices{swizzleIndexes...};
    for (int i = 0; i < n; i++)
      result[i] = m_elements[indices[i]];
    return result;
  }

#include "util.vec_swizzles.inc"

  // load and store member functions
  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(std::size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) {
    load(offset, static_cast<const DataT *>(ptr));
  }

  void load(std::size_t offset, const DataT *ptr) {
    for (int i = 0; i < NumElements; i++)
      m_elements[i] = ptr[offset + i];
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(std::size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
    store(offset, static_cast<DataT *>(ptr));
  }

  void store(std::size_t offset, DataT *ptr) const {
    for (int i = 0; i < NumElements; i++)
      ptr[offset + i] = m_elements[i];
  }

  // subscript operator
  DataT &operator[](int index) { return m_elements[index]; }
  const DataT &operator[](int index) const { return m_elements[index]; }

  vec &operator=(const DataT &rhs)
    requires(NumElements >= 2)
  {
    for (int i = 0; i < NumElements; i++)
      (*this)[i] = rhs;
    return *this;
  }

#define SYCL_OP(OP)                                                            \
  friend vec operator OP(const vec &lhs, const vec &rhs) {                     \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec operator OP(const vec &lhs, const T &rhs)                         \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs;                                               \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec operator OP(const T &lhs, const vec &rhs)                         \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
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
  friend vec &operator OP(vec & lhs, const vec & rhs) {                        \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs[i] OP rhs[i];                                                        \
    return lhs;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend vec &operator OP(vec & lhs, const T & rhs)                            \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    for (int i = 0; i < NumElements; i++)                                      \
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
  friend vec &operator OP(vec & rhs) {                                         \
    for (int i = 0; i < NumElements; i++)                                      \
      OP rhs[i];                                                               \
    return rhs;                                                                \
  }                                                                            \
  friend vec operator OP(vec &lhs, int) {                                      \
    vec result = lhs;                                                          \
    OP lhs;                                                                    \
    return result;                                                             \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend vec operator OP(const vec &rhs) {                                     \
    vec result{};                                                              \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = OP rhs[i];                                                   \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
  SYCL_OP(~)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      const vec & lhs, const vec & rhs) {                                      \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs[i]) ? -1 : 0);                               \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      const vec & lhs, const T & rhs)                                          \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs) ? -1 : 0);                                  \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      const T & lhs, const vec & rhs)                                          \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs OP rhs[i]) ? -1 : 0);                                  \
    return result;                                                             \
  }

  SYCL_OP(==)
  SYCL_OP(!=)
  SYCL_OP(<)
  SYCL_OP(>)
  SYCL_OP(<=)
  SYCL_OP(>=)
  SYCL_OP(&&)
  SYCL_OP(||)
#undef SYCL_OP

  friend vec<detail::vec_logical_ret_t<DataT>, NumElements>
  operator!(const vec &v) {
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;
    for (int i = 0; i < NumElements; i++)
      result[i] = (!v[i] ? -1 : 0);
    return result;
  }

private:
  template <typename, int> friend class vec;

  template <std::size_t Offset, typename... ArgTN>
    requires(Offset + 1 <= NumElements)
  constexpr void init_with_offset(const DataT &arg, const ArgTN &...args) {
    m_elements[Offset] = arg;
    init_with_offset<Offset + 1>(args...);
  }

  template <std::size_t Offset, int ArgNumElements, typename... ArgTN>
    requires(Offset + ArgNumElements <= NumElements)
  constexpr void init_with_offset(const vec<DataT, ArgNumElements> &arg,
                                  const ArgTN &...args) {
    for (std::size_t i = 0; i < ArgNumElements; i++)
      m_elements[Offset + i] = arg[i];
    init_with_offset<Offset + ArgNumElements>(args...);
  }

  // Not in spec
  template <std::size_t Offset, typename VecT, int ArgNumElements,
            typename... ArgTN>
    requires(Offset + ArgNumElements <= NumElements)
  constexpr void init_with_offset(
      const detail::__writeable_swizzle__<VecT, ArgNumElements> &arg,
      const ArgTN &...args) {
    for (std::size_t i = 0; i < ArgNumElements; i++)
      m_elements[Offset + i] = arg[i];
    init_with_offset<Offset + ArgNumElements>(args...);
  }

  template <std::size_t Offset, typename... ArgTN>
  constexpr void init_with_offset() {}

  std::array<std::remove_const_t<DataT>, NumElements == 3 ? 4 : NumElements>
      m_elements{};
};

namespace detail {

template <typename VecT, int NumElements> class __writeable_swizzle__ {
private:
  using DataT = VecT::element_type;

public:
  using element_type = DataT;
  using value_type = DataT;

#ifdef __SYCL_DEVICE_ONLY__
  operator vector_t() const
#endif

  operator DataT() const
    requires(NumElements == 1)
  {
    return (*this)[0];
  }

  static constexpr std::size_t byte_size() noexcept {
    return VecT::byte_size();
  }

  static constexpr std::size_t size() noexcept { return VecT::size(); }

  // Deprecated
  std::size_t get_size() const { return byte_size(); }

  // Deprecated
  std::size_t get_count() const { return size(); }

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    return static_cast<vec<DataT, NumElements>>(*this)
        .template convert<ConvertT, RoundingMode>();
  }

  template <typename asT> asT as() const {
    return static_cast<vec<DataT, NumElements>>(*this).template as<asT>();
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(std::size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) const;

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(std::size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const;

  operator vec<DataT, NumElements>() const {
    auto make_vec = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return vec<DataT, NumElements>{m_vec[m_indices[I]]...};
    };
    return make_vec(std::make_index_sequence<NumElements>{});
  }

  template <int... swizzleIndexes>
  __writeable_swizzle__<__writeable_swizzle__, sizeof...(swizzleIndexes)>
  swizzle() {
    return __writeable_swizzle__<__writeable_swizzle__,
                                 sizeof...(swizzleIndexes)>{
        *this, {swizzleIndexes...}};
  }

#include "util.vec_swizzles.inc"

  DataT &operator[](int index) const { return m_vec[m_indices[index]]; }

  template <typename OtherDataT>
  const __writeable_swizzle__ &
  operator=(const __writeable_swizzle__<OtherDataT, NumElements> &rhs) const {
    for (int i = 0; i < NumElements; i++)
      (*this)[i] = rhs[i];
    return *this;
  }

  const __writeable_swizzle__ &
  operator=(const __writeable_swizzle__ &rhs) const {
    return this->operator= <VecT>(rhs);
  }

  const __writeable_swizzle__ &operator=(const DataT &rhs) const {
    for (int i = 0; i < NumElements; i++)
      (*this)[i] = rhs;
    return *this;
  }

  const __writeable_swizzle__ &
  operator=(const vec<DataT, NumElements> &rhs) const {
    for (int i = 0; i < NumElements; i++)
      (*this)[i] = rhs[i];
    return *this;
  }

#define SYCL_OP(OP)                                                            \
  friend vec<DataT, NumElements> operator OP(                                  \
      const __writeable_swizzle__ & lhs, const __writeable_swizzle__ & rhs) {  \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<DataT, NumElements> operator OP(                                  \
      const __writeable_swizzle__ & lhs, const T & rhs)                        \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs;                                               \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<DataT, NumElements> operator OP(                                  \
      const T & lhs, const __writeable_swizzle__ & rhs)                        \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs OP rhs[i];                                               \
    return result;                                                             \
  }                                                                            \
  friend vec<DataT, NumElements> operator OP(                                  \
      const __writeable_swizzle__ & lhs, const vec<DataT, NumElements> &rhs) { \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs[i];                                            \
    return result;                                                             \
  }                                                                            \
  friend vec<DataT, NumElements> operator OP(                                  \
      const vec<DataT, NumElements> &lhs, const __writeable_swizzle__ & rhs) { \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = lhs[i] OP rhs[i];                                            \
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
  friend __writeable_swizzle__ operator OP(__writeable_swizzle__ lhs,          \
                                           const __writeable_swizzle__ &rhs) { \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs[i] OP rhs[i];                                                        \
    return lhs;                                                                \
  }                                                                            \
  friend __writeable_swizzle__ operator OP(                                    \
      __writeable_swizzle__ lhs, const vec<DataT, NumElements> &rhs) {         \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs[i] OP rhs[i];                                                        \
    return lhs;                                                                \
  }                                                                            \
  friend vec<DataT, NumElements> &operator OP(                                 \
      vec<DataT, NumElements> &lhs, const __writeable_swizzle__ & rhs) {       \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs[i] OP rhs[i];                                                        \
    return lhs;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend __writeable_swizzle__ operator OP(__writeable_swizzle__ lhs,          \
                                           const T &rhs)                       \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs[i] OP rhs;                                                           \
    return lhs;                                                                \
  }                                                                            \
  template <typename T>                                                        \
  friend T &operator OP(T & lhs, const __writeable_swizzle__ & rhs)            \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    for (int i = 0; i < NumElements; i++)                                      \
      lhs OP rhs[i];                                                           \
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
  friend __writeable_swizzle__ operator OP(__writeable_swizzle__ rhs) {        \
    for (int i = 0; i < NumElements; i++)                                      \
      OP rhs[i];                                                               \
    return rhs;                                                                \
  }                                                                            \
  friend vec<DataT, NumElements> operator OP(__writeable_swizzle__ lhs, int) { \
    vec<DataT, NumElements> result = lhs;                                      \
    OP lhs;                                                                    \
    return result;                                                             \
  }

  SYCL_OP(++)
  SYCL_OP(--)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend vec<DataT, NumElements> operator OP(                                  \
      const __writeable_swizzle__ & rhs) {                                     \
    vec<DataT, NumElements> result{};                                          \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = OP rhs[i];                                                   \
    return result;                                                             \
  }

  SYCL_OP(+)
  SYCL_OP(-)
  SYCL_OP(~)
#undef SYCL_OP

#define SYCL_OP(OP)                                                            \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      __writeable_swizzle__ lhs, __writeable_swizzle__ rhs) {                  \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs[i]) ? -1 : 0);                               \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      __writeable_swizzle__ lhs, const T & rhs)                                \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs) ? -1 : 0);                                  \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      const T & lhs, __writeable_swizzle__ rhs)                                \
    requires std::is_convertible_v<DataT, T>                                   \
  {                                                                            \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs OP rhs[i]) ? -1 : 0);                                  \
    return result;                                                             \
  }                                                                            \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      __writeable_swizzle__ lhs, const vec<DataT, NumElements> &rhs) {         \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs[i]) ? -1 : 0);                               \
    return result;                                                             \
  }                                                                            \
  friend vec<detail::vec_logical_ret_t<DataT>, NumElements> operator OP(       \
      const vec<DataT, NumElements> &lhs, __writeable_swizzle__ rhs) {         \
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;                 \
    for (int i = 0; i < NumElements; i++)                                      \
      result[i] = ((lhs[i] OP rhs[i]) ? -1 : 0);                               \
    return result;                                                             \
  }

  SYCL_OP(==)
  SYCL_OP(!=)
  SYCL_OP(<)
  SYCL_OP(>)
  SYCL_OP(<=)
  SYCL_OP(>=)
  SYCL_OP(&&)
  SYCL_OP(||)
#undef SYCL_OP

  friend vec<detail::vec_logical_ret_t<DataT>, NumElements>
  operator!(__writeable_swizzle__ v) {
    vec<detail::vec_logical_ret_t<DataT>, NumElements> result;
    for (int i = 0; i < NumElements; i++)
      result[i] = (!v[i] ? -1 : 0);
    return result;
  }

private:
  friend VecT;

  __writeable_swizzle__(VecT &vec, std::array<int, NumElements> indices)
      : m_vec{vec}, m_indices{indices} {}

  VecT &m_vec;
  std::array<int, NumElements> m_indices;
};

} // namespace detail

// Deduction guides
template <class T, class... U>
  requires(std::is_same_v<T, U> && ...)
vec(T, U...) -> vec<T, sizeof...(U) + 1>;

using char2 = vec<std::int8_t, 2>;
using char3 = vec<std::int8_t, 3>;
using char4 = vec<std::int8_t, 4>;
using char8 = vec<std::int8_t, 8>;
using char16 = vec<std::int8_t, 16>;

using uchar2 = vec<std::uint8_t, 2>;
using uchar3 = vec<std::uint8_t, 3>;
using uchar4 = vec<std::uint8_t, 4>;
using uchar8 = vec<std::uint8_t, 8>;
using uchar16 = vec<std::uint8_t, 16>;

using short2 = vec<std::int16_t, 2>;
using short3 = vec<std::int16_t, 3>;
using short4 = vec<std::int16_t, 4>;
using short8 = vec<std::int16_t, 8>;
using short16 = vec<std::int16_t, 16>;

using ushort2 = vec<std::uint16_t, 2>;
using ushort3 = vec<std::uint16_t, 3>;
using ushort4 = vec<std::uint16_t, 4>;
using ushort8 = vec<std::uint16_t, 8>;
using ushort16 = vec<std::uint16_t, 16>;

using int2 = vec<std::int32_t, 2>;
using int3 = vec<std::int32_t, 3>;
using int4 = vec<std::int32_t, 4>;
using int8 = vec<std::int32_t, 8>;
using int16 = vec<std::int32_t, 16>;

using uint2 = vec<std::uint32_t, 2>;
using uint3 = vec<std::uint32_t, 3>;
using uint4 = vec<std::uint32_t, 4>;
using uint8 = vec<std::uint32_t, 8>;
using uint16 = vec<std::uint32_t, 16>;

using long2 = vec<std::int64_t, 2>;
using long3 = vec<std::int64_t, 3>;
using long4 = vec<std::int64_t, 4>;
using long8 = vec<std::int64_t, 8>;
using long16 = vec<std::int64_t, 16>;

using ulong2 = vec<std::uint64_t, 2>;
using ulong3 = vec<std::uint64_t, 3>;
using ulong4 = vec<std::uint64_t, 4>;
using ulong8 = vec<std::uint64_t, 8>;
using ulong16 = vec<std::uint64_t, 16>;

using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using half8 = vec<half, 8>;
using half16 = vec<half, 16>;

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

} // namespace sycl
