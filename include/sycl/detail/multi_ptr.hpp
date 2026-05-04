#pragma once

#include <iterator>
#include <utility>

#include "access.hpp"
#include "util.forward.hpp"

namespace sycl {

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
class multi_ptr;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using global_ptr =
    multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using local_ptr =
    multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

// Deprecated in SYCL 2020
template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space,
              access::decorated::legacy>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

template <typename ElementType>
using raw_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::no>;

template <typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space,
                                access::decorated::no>;

template <typename ElementType>
using raw_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::no>;

template <typename ElementType>
using raw_generic_ptr =
    sycl::multi_ptr<ElementType, sycl::access::address_space::generic_space,
                    sycl::access::decorated::no>;

template <typename ElementType>
using decorated_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_local_ptr =
    multi_ptr<ElementType, access::address_space::local_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_generic_ptr =
    sycl::multi_ptr<ElementType, sycl::access::address_space::generic_space,
                    sycl::access::decorated::yes>;

template <typename T> struct remove_decoration {
  using type = T;
};

template <typename T> using remove_decoration_t = remove_decoration<T>::type;

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr {
public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;
  using pointer = std::add_pointer_t<value_type>;
  using reference = std::add_lvalue_reference_t<value_type>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                               std::add_lvalue_reference_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  explicit multi_ptr(typename multi_ptr<ElementType, Space,
                                        access::decorated::yes>::pointer ptr)
      : m_value{ptr} {}
  multi_ptr(std::nullptr_t) : m_value{nullptr} {}

  template <typename AccDataT, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder> acc)
    requires(Space == access::address_space::global_space ||
             Space == access::address_space::generic_space) &&
            (std::is_same_v<std::remove_const_t<ElementType>,
                            std::remove_const_t<AccDataT>>) &&
            (std::is_const_v<ElementType> ||
             !std::is_const_v<
                 typename accessor<AccDataT, Dimensions, Mode, target::device,
                                   IsPlaceholder>::value_type>)
      : m_value{acc.get_pointer()} {}

  template <typename AccDataT, int Dimensions>
  multi_ptr(local_accessor<AccDataT, Dimensions> acc)
    requires(Space == access::address_space::local_space ||
             Space == access::address_space::generic_space) &&
            (std::is_same_v<std::remove_const_t<ElementType>,
                            std::remove_const_t<AccDataT>>) &&
            (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>)
      : m_value{acc.get_pointer()} {}

  // Deprecated
  template <typename AccDataT, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::local, IsPlaceholder> acc)
    requires(Space == access::address_space::local_space ||
             Space == access::address_space::generic_space) &&
            (std::is_same_v<std::remove_const_t<ElementType>,
                            std::remove_const_t<AccDataT>>) &&
            (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>)
      : m_value{acc.get_pointer()} {}

  // Deprecated
  template <typename AccDataT, int Dimensions,
            access::placeholder IsPlaceholder>
  multi_ptr(accessor<AccDataT, Dimensions, access_mode::read,
                     target::constant_buffer, IsPlaceholder>
                acc)
    requires(Space == access::address_space::constant_space &&
             (std::is_same_v<std::remove_const_t<ElementType>,
                             std::remove_const_t<AccDataT>>) &&
             (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>))
      : m_value{acc.get_pointer()} {}

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated>
  multi_ptr &operator=(const multi_ptr<value_type, AS, IsDecorated> &rhs)
    requires((Space == access::address_space::generic_space &&
              AS != access::address_space::constant_space))
  {
    m_value = static_cast<value_type *>(rhs);
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated>
  multi_ptr &operator=(multi_ptr<value_type, AS, IsDecorated> &&rhs)
    requires((Space == access::address_space::generic_space &&
              AS != access::address_space::constant_space))
  {
    m_value = static_cast<value_type *>(rhs);
    return *this;
  }

  reference operator[](std::ptrdiff_t n) const { return *(m_value + n); }

  reference operator*() const { return *m_value; }
  pointer operator->() const { return m_value; }

  pointer get() const { return m_value; }
  pointer get_raw() const { return m_value; }
  pointer *get_decorated() const { return m_value; }

  // Conversion to the underlying pointer type
  // Deprecated, get() should be used instead.
  operator pointer() const { return m_value; }

  // Cast to private_ptr
  template <access::decorated IsDecorated>
  explicit operator multi_ptr<value_type, access::address_space::private_space,
                              IsDecorated>()
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<value_type, access::address_space::private_space,
                     IsDecorated>{m_value};
  }

  // Cast to private_ptr of const data
  template <access::decorated IsDecorated>
  explicit
  operator multi_ptr<const value_type, access::address_space::private_space,
                     IsDecorated>() const
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<const value_type, access::address_space::private_space,
                     IsDecorated>{m_value};
  }

  // Cast to global_ptr
  template <access::decorated IsDecorated>
  explicit operator multi_ptr<value_type, access::address_space::global_space,
                              IsDecorated>()
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<value_type, access::address_space::global_space,
                     IsDecorated>{m_value};
  }

  // Cast to global_ptr of const data
  template <access::decorated IsDecorated>
  explicit
  operator multi_ptr<const value_type, access::address_space::global_space,
                     IsDecorated>() const
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<const value_type, access::address_space::global_space,
                     IsDecorated>{m_value};
  }

  // Cast to local_ptr
  template <access::decorated IsDecorated>
  explicit operator multi_ptr<value_type, access::address_space::local_space,
                              IsDecorated>()
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<value_type, access::address_space::local_space,
                     IsDecorated>{m_value};
  }

  // Cast to local_ptr of const data
  template <access::decorated IsDecorated>
  explicit
  operator multi_ptr<const value_type, access::address_space::local_space,
                     IsDecorated>() const
    requires(Space == access::address_space::generic_space)
  {
    return multi_ptr<const value_type, access::address_space::local_space,
                     IsDecorated>{m_value};
  }

  // Implicit conversion to a multi_ptr<void>.
  template <access::decorated IsDecorated>
  operator multi_ptr<void, Space, IsDecorated>() const
    requires(!std::is_const_v<value_type>)
  {
    return multi_ptr<void, Space, IsDecorated>{m_value};
  }

  // Implicit conversion to a multi_ptr<const void>.
  template <access::decorated IsDecorated>
  operator multi_ptr<const void, Space, IsDecorated>() const
    requires(std::is_const_v<value_type>)
  {
    return multi_ptr<const void, Space, IsDecorated>{m_value};
  }

  // Implicit conversion to multi_ptr<const value_type, Space>.
  template <access::decorated IsDecorated>
  operator multi_ptr<const value_type, Space, IsDecorated>() const {
    return multi_ptr<const value_type, Space, IsDecorated>{m_value};
  }

  // Implicit conversion to the non-decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::no>() const
    requires(is_decorated == true)
  {
    return multi_ptr<value_type, Space, access::decorated::no>{m_value};
  }

  // Implicit conversion to the decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::yes>() const
    requires(is_decorated == false)
  {
    return multi_ptr<value_type, Space, access::decorated::yes>{m_value};
  }

  void prefetch(std::size_t numElements) const
    requires(Space == access::address_space::global_space)
  {
    std::ignore = numElements;
  }

  // Arithmetic operators
  friend multi_ptr &operator++(multi_ptr &mp) {
    ++mp.m_value;
    return mp;
  }
  friend multi_ptr operator++(multi_ptr &mp, int) {
    return multi_ptr{mp.m_value++};
  }
  friend multi_ptr &operator--(multi_ptr &mp) {
    --mp.m_value;
    return mp;
  }
  friend multi_ptr operator--(multi_ptr &mp, int) {
    return multi_ptr{mp.m_value--};
  }
  friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) {
    lhs.m_value += r;
    return lhs;
  }
  friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) {
    lhs.m_value -= r;
    return lhs;
  }
  friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) {
    return multi_ptr{lhs.m_value + r};
  }
  friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) {
    return multi_ptr{lhs.m_value - r};
  }
  // friend reference operator*(const multi_ptr &lhs) { /* ... */ }

  friend auto operator<=>(const multi_ptr &, const multi_ptr &) = default;

  friend bool operator==(const multi_ptr &lhs, std::nullptr_t) {
    return lhs == multi_ptr{nullptr};
  }
  friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs != multi_ptr{nullptr};
  }
  friend bool operator<(const multi_ptr &lhs, std::nullptr_t) {
    return lhs < multi_ptr{nullptr};
  }
  friend bool operator>(const multi_ptr &lhs, std::nullptr_t) {
    return lhs > multi_ptr{nullptr};
  }
  friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs <= multi_ptr{nullptr};
  }
  friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs >= multi_ptr{nullptr};
  }

  friend bool operator==(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} == rhs;
  }
  friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} != rhs;
  }
  friend bool operator<(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} < rhs;
  }
  friend bool operator>(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} > rhs;
  }
  friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} <= rhs;
  }
  friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} >= rhs;
  }

private:
  pointer m_value;
};

namespace detail {
template <typename T>
concept VoidType = std::is_void_v<T> && !std::is_volatile_v<T>;
}

// Specialization of multi_ptr for void and const void
// VoidType can be either void or const void
template <detail::VoidType VoidType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr<VoidType, Space, DecorateAddress> {
public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = VoidType;
  using pointer = std::add_pointer_t<value_type>;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  explicit multi_ptr(
      typename multi_ptr<VoidType, Space, access::decorated::yes>::pointer ptr)
      : m_value{ptr} {}
  multi_ptr(std::nullptr_t) : m_value{nullptr} {}

  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder>
          acc)
    requires((Space == access::address_space::global_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> ||
              !std::is_const_v<typename accessor<ElementType, Dimensions, Mode,
                                                 target::device,
                                                 IsPlaceholder>::value_type>))
      : m_value{acc.get_pointer} {}

  template <typename ElementType, int Dimensions>
  multi_ptr(local_accessor<ElementType, Dimensions> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> || !std::is_const_v<ElementType>))
      : m_value{acc.get_pointer} {}

  // Deprecated
  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> || !std::is_const_v<ElementType>))
      : m_value{acc.get_pointer} {}

  // Deprecated
  template <typename ElementType, int Dimensions,
            access::placeholder IsPlaceholder>
  multi_ptr(accessor<ElementType, Dimensions, access_mode::read,
                     target::constant_buffer, IsPlaceholder>
                acc)
    requires(Space == access::address_space::constant_space &&
             (std::is_const_v<VoidType> || !std::is_const_v<ElementType>))
      : m_value{acc.get_pointer()} {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }

  pointer get() const { return m_value; }

  // Conversion to the underlying pointer type
  operator pointer() const { return m_value; }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const
    requires(std::is_const_v<ElementType> || !std::is_const_v<VoidType>)
  {
    return multi_ptr<ElementType, Space, DecorateAddress>{m_value};
  }

  // Implicit conversion to the non-decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::no>() const
    requires(is_decorated == true)
  {
    return multi_ptr<value_type, Space, access::decorated::no>{m_value};
  }

  // Implicit conversion to the decorated version of multi_ptr.
  operator multi_ptr<value_type, Space, access::decorated::yes>() const
    requires(is_decorated == false)
  {
    return multi_ptr<value_type, Space, access::decorated::yes>{m_value};
  }

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space, DecorateAddress>() const {
    return multi_ptr<const void, Space, DecorateAddress>{m_value};
  }

  friend bool operator<=>(const multi_ptr &, const multi_ptr &) = default;

  friend bool operator==(const multi_ptr &lhs, std::nullptr_t) {
    return lhs == multi_ptr{nullptr};
  }
  friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs != multi_ptr{nullptr};
  }
  friend bool operator<(const multi_ptr &lhs, std::nullptr_t) {
    return lhs < multi_ptr{nullptr};
  }
  friend bool operator>(const multi_ptr &lhs, std::nullptr_t) {
    return lhs > multi_ptr{nullptr};
  }
  friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs <= multi_ptr{nullptr};
  }
  friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs >= multi_ptr{nullptr};
  }

  friend bool operator==(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} == rhs;
  }
  friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} != rhs;
  }
  friend bool operator<(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} < rhs;
  }
  friend bool operator>(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} > rhs;
  }
  friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} <= rhs;
  }
  friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} >= rhs;
  }

private:
  pointer m_value;
};

// Deprecated, address_space_cast should be used instead.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType *ptr) {
  return multi_ptr<ElementType, Space, DecorateAddress>{ptr};
}

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *ptr) {
  return multi_ptr<ElementType, Space, DecorateAddress>{ptr};
}

// Deduction guides
template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(
    accessor<T, Dimensions, access_mode::read, target::device, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::global_space,
                 access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(
    accessor<T, Dimensions, access_mode::write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read_write, target::device,
                   IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::constant_buffer,
                   IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::constant_space,
                 access::decorated::no>;

template <typename T, int Dimensions, access_mode Mode,
          access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::local, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

template <typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

// Legacy interface, inherited from 1.2.1.
// Deprecated.
template <typename ElementType, access::address_space Space>
class [[deprecated]] multi_ptr<ElementType, Space, access::decorated::legacy> {
public:
  using value_type = ElementType;
  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions.
  using pointer_t =
      multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
      multi_ptr<const ElementType, Space, access::decorated::yes>::pointer;
  using reference_t =
      multi_ptr<ElementType, Space, access::decorated::yes>::reference;
  using const_reference_t =
      multi_ptr<const ElementType, Space, access::decorated::yes>::reference;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t ptr) : m_value{ptr} {}
  multi_ptr(std::nullptr_t) : m_value{nullptr} {}
  ~multi_ptr() = default;

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t ptr) {
    m_value = ptr;
    return *this;
  }
  multi_ptr &operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }
  // What could this possibly mean?
  // friend ElementType &operator*(const multi_ptr &mp) { /* ... */ }
  ElementType *operator->() const { return m_value; }

  template <typename AccDataT, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder> acc)
    requires((Space == access::address_space::global_space ||
              Space == access::address_space::generic_space) &&
             (std::is_same_v<std::remove_const_t<ElementType>,
                             std::remove_const_t<AccDataT>>) &&
             (std::is_const_v<ElementType> ||
              !std::is_const_v<
                  typename accessor<AccDataT, Dimensions, Mode, target::device,
                                    IsPlaceholder>::value_type>))
      : m_value{acc.get_pointer()} {}

  template <typename AccDataT, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::local, IsPlaceholder> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_same_v<std::remove_const_t<ElementType>,
                             std::remove_const_t<AccDataT>>) &&
             (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>))
      : m_value{acc.get_pointer()} {}

  template <typename AccDataT, int Dimensions>
  multi_ptr(local_accessor<AccDataT, Dimensions> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_same_v<std::remove_const_t<ElementType>,
                             std::remove_const_t<AccDataT>>) &&
             (std::is_const_v<ElementType> || !std::is_const_v<AccDataT>))
      : m_value{acc.get_pointer()} {}

  template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::constant_buffer,
                     IsPlaceholder>
                acc)
    requires(Space == access::address_space::constant_space)
      : m_value{acc.get_pointer()} {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_value; }

  std::add_pointer_t<value_type> get_raw() const { return m_value; }

  pointer_t get_decorated() const { return m_value; }

  // Implicit conversion to the underlying pointer type
  operator ElementType *() const { return m_value; }

  // Implicit conversion to a multi_ptr<void>
  // Available only when ElementType is not const-qualified
  operator multi_ptr<void, Space, access::decorated::legacy>() const {
    return m_value;
  }

  // Implicit conversion to a multi_ptr<const void>
  // Available only when ElementType is const-qualified
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    return m_value;
  }

  // Implicit conversion to multi_ptr<const ElementType, Space>
  operator multi_ptr<const ElementType, Space, access::decorated::legacy>()
      const {
    return m_value;
  }

  // Arithmetic operators
  friend multi_ptr &operator++(multi_ptr &mp) {
    ++mp.m_value;
    return mp;
  }
  friend multi_ptr operator++(multi_ptr &mp, int) {
    return multi_ptr{mp.m_value++};
  }
  friend multi_ptr &operator--(multi_ptr &mp) {
    --mp.m_value;
    return mp;
  }
  friend multi_ptr operator--(multi_ptr &mp, int) {
    return multi_ptr{mp.m_value--};
  }
  friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) {
    lhs.m_value += r;
    return lhs;
  }
  friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) {
    lhs.m_value -= r;
    return lhs;
  }
  friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) {
    return multi_ptr{lhs.m_value + r};
  }
  friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) {
    return multi_ptr{lhs.m_value - r};
  }

  void prefetch(std::size_t numElements) const { std::ignore = numElements; }

  friend auto operator<=>(const multi_ptr &, const multi_ptr &) = default;

  friend bool operator==(const multi_ptr &lhs, std::nullptr_t) {
    return lhs == multi_ptr{nullptr};
  }
  friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs != multi_ptr{nullptr};
  }
  friend bool operator<(const multi_ptr &lhs, std::nullptr_t) {
    return lhs < multi_ptr{nullptr};
  }
  friend bool operator>(const multi_ptr &lhs, std::nullptr_t) {
    return lhs > multi_ptr{nullptr};
  }
  friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs <= multi_ptr{nullptr};
  }
  friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs >= multi_ptr{nullptr};
  }

  friend bool operator==(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} == rhs;
  }
  friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} != rhs;
  }
  friend bool operator<(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} < rhs;
  }
  friend bool operator>(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} > rhs;
  }
  friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} <= rhs;
  }
  friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} >= rhs;
  }

private:
  pointer_t m_value;
};

// Legacy interface, inherited from 1.2.1.
// Deprecated.
// Specialization of multi_ptr for void and const void
// VoidType can be either void or const void
template <detail::VoidType VoidType, access::address_space Space>
class [[deprecated]] multi_ptr<VoidType, Space, access::decorated::legacy> {
public:
  using value_type = VoidType;
  using element_type = VoidType;
  using difference_type = std::ptrdiff_t;

  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = multi_ptr<VoidType, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
      multi_ptr<const VoidType, Space, access::decorated::yes>::pointer;

  static constexpr access::address_space address_space = Space;

  // Constructors
  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  multi_ptr(pointer_t ptr) : m_value{ptr} {}
  multi_ptr(std::nullptr_t) : m_value{nullptr} {}
  ~multi_ptr() = default;

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  multi_ptr &operator=(pointer_t ptr) {
    m_value = ptr;
    return *this;
  }
  multi_ptr &operator=(std::nullptr_t) {
    m_value = nullptr;
    return *this;
  }

  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder>
  multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder>
          acc)
    requires((Space == access::address_space::global_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> ||
              !std::is_const_v<typename accessor<ElementType, Dimensions, Mode,
                                                 target::device,
                                                 IsPlaceholder>::value_type>))
      : m_value{acc.get_pointer()} {}

  template <typename ElementType, int Dimensions, access_mode Mode>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::local> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> || !std::is_const_v<ElementType>))
      : m_value{acc.get_pointer()} {}

  template <typename AccDataT, int Dimensions>
  multi_ptr(local_accessor<AccDataT, Dimensions> acc)
    requires((Space == access::address_space::local_space ||
              Space == access::address_space::generic_space) &&
             (std::is_const_v<VoidType> || !std::is_const_v<element_type>))
      : m_value{acc.get_pointer()} {}

  template <typename ElementType, int Dimensions, access_mode Mode>
  multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::constant_buffer> acc)
    requires(Space == access::address_space::constant_space)
      : m_value{acc.get_pointer()} {}

  // Returns the underlying OpenCL C pointer
  pointer_t get() const { return m_value; }

  std::add_pointer_t<value_type> get_raw() const { return m_value; }

  pointer_t get_decorated() const { return m_value; }

  // Implicit conversion to the underlying pointer type
  operator VoidType *() const { return m_value; }

  // Explicit conversion to a multi_ptr<ElementType>
  // If VoidType is const, ElementType must be as well
  template <typename ElementType>
  explicit
  operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {
    return static_cast<ElementType *>(m_value);
  }

  // Implicit conversion to multi_ptr<const void, Space>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    return m_value;
  }

  friend auto operator<=>(const multi_ptr &, const multi_ptr &) = default;

  friend bool operator==(const multi_ptr &lhs, std::nullptr_t) {
    return lhs == multi_ptr{nullptr};
  }
  friend bool operator!=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs != multi_ptr{nullptr};
  }
  friend bool operator<(const multi_ptr &lhs, std::nullptr_t) {
    return lhs < multi_ptr{nullptr};
  }
  friend bool operator>(const multi_ptr &lhs, std::nullptr_t) {
    return lhs > multi_ptr{nullptr};
  }
  friend bool operator<=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs <= multi_ptr{nullptr};
  }
  friend bool operator>=(const multi_ptr &lhs, std::nullptr_t) {
    return lhs >= multi_ptr{nullptr};
  }

  friend bool operator==(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} == rhs;
  }
  friend bool operator!=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} != rhs;
  }
  friend bool operator<(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} < rhs;
  }
  friend bool operator>(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} > rhs;
  }
  friend bool operator<=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} <= rhs;
  }
  friend bool operator>=(std::nullptr_t, const multi_ptr &rhs) {
    return multi_ptr{nullptr} >= rhs;
  }

private:
  pointer_t m_value;
};

} // namespace sycl
