#pragma once

#include "access.hpp"
#include "accessor_tag.hpp"
#include "atomic.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "multi_ptr.hpp"
#include "property.hpp"
#include "util.accessor_iterator.hpp"
#include "util.linearize.hpp"
#include "util.subscript.hpp"

namespace sycl::detail {

template <typename DataT, int Dimensions> struct accessor_impl {
  DataT *m_data = nullptr;
  range<Dimensions == 0 ? 1 : Dimensions> m_bufferRange;
  range<Dimensions == 0 ? 1 : Dimensions> m_accessRange;
  id<Dimensions == 0 ? 1 : Dimensions> m_offset;
  handler *m_handler = nullptr;
  property_list m_props;
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder IsPlaceholder>
const void *
get_accessor_data_pointer(const accessor<DataT, Dimensions, AccessMode,
                                         AccessTarget, IsPlaceholder> &);

} // namespace sycl::detail

namespace sycl {

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder IsPlaceholder>
void register_accessor(handler &,
                       const accessor<DataT, Dimensions, AccessMode,
                                      AccessTarget, IsPlaceholder> &);

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder IsPlaceholder>
class accessor {
public:
  using value_type =
      std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;
  template <access::decorated IsDecorated>
  using accessor_ptr =
      multi_ptr<value_type, access::address_space::global_space, IsDecorated>;
  using iterator = detail::accessor_iterator<value_type, accessor>;
  using const_iterator = detail::accessor_iterator<const value_type, accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  accessor() = default;
  accessor(const accessor &) = default;

  template <typename AllocatorT>
  accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef, TagT tag,
           const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, propList} {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, TagT tag,
           const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, commandGroupHandlerRef, propList} {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange);
    check_props();
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, TagT tag,
           const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, accessRange, propList} {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, id<Dimensions> accessOffset,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange, accessOffset);
    check_props();
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, id<Dimensions> accessOffset, TagT tag,
           const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, accessRange, accessOffset, propList} {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_accessRange = accessRange;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange);
    check_props();
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           TagT tag, const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, propList} {}

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           id<Dimensions> accessOffset, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_accessRange = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange, accessOffset);
    check_props();
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT, typename TagT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           id<Dimensions> accessOffset, TagT tag,
           const property_list &propList = {})
    requires(Dimensions > 0)
      : accessor{bufferRef, commandGroupHandlerRef, accessRange, accessOffset,
                 propList} {}

  accessor(
      const accessor<std::remove_const_t<DataT>, Dimensions, access_mode::read,
                     AccessTarget, IsPlaceholder> &other)
    requires(AccessMode == access_mode::read && std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  accessor(const accessor<const DataT, Dimensions, access_mode::read,
                          AccessTarget, IsPlaceholder> &other)
    requires(AccessMode == access_mode::read && !std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  accessor(const accessor<std::remove_const_t<DataT>, Dimensions,
                          access_mode::read_write, AccessTarget, IsPlaceholder>
               &other)
    requires(AccessMode == access_mode::read && std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  accessor(const accessor<const DataT, Dimensions, access_mode::read_write,
                          AccessTarget, IsPlaceholder> &other)
    requires(AccessMode == access_mode::read && !std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  accessor(const accessor<std::remove_const_t<DataT>, Dimensions,
                          access_mode::read_write, AccessTarget, IsPlaceholder>
               &other)
    requires(AccessMode == access_mode::write && !std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  friend bool operator==(const accessor &, const accessor &) = default;

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  void swap(accessor &other) { std::swap(m_impl, other.m_impl); }

  bool is_placeholder() const {
    return m_impl->m_handler == nullptr && m_impl->m_data != nullptr;
  }

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type size() const noexcept { return m_impl->m_accessRange.size(); }

  size_type max_size() const noexcept {
    return m_impl->m_data ? m_impl->m_bufferRange.size() : 0;
  }

  // Deprecated
  std::size_t get_size() const { return byte_size(); }

  // Deprecated
  std::size_t get_count() const { return size(); }

  bool empty() const noexcept { return size() == 0; }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return m_impl->m_accessRange;
  }

  id<Dimensions> get_offset() const
    requires(Dimensions > 0)
  {
    return m_impl->m_offset;
  }

  operator reference() const
    requires(AccessMode != access_mode::atomic && Dimensions == 0)
  {
    return m_impl->m_data[0];
  }

  const accessor &operator=(const value_type &other) const
    requires(AccessMode != access_mode::atomic &&
             AccessMode != access_mode::read && Dimensions == 0)
  {
    m_impl->m_data[0] = other;
    return *this;
  }

  const accessor &operator=(value_type &&other) const
    requires(AccessMode != access_mode::atomic &&
             AccessMode != access_mode::read && Dimensions == 0)
  {
    m_impl->m_data[0] = other;
    return *this;
  }

  reference operator[](id<Dimensions> index) const
    requires(Dimensions > 1)
  {
    const auto offset = m_impl->m_offset;
    const auto buffer_range = m_impl->m_bufferRange;
    const auto linear_index = detail::linearize(offset + index, buffer_range);
    return m_impl->m_data[linear_index];
  }

  decltype(auto) operator[](std::size_t index) const
    requires(Dimensions > 1)
  {
    return detail::subscript<decltype(*this), Dimensions, 0>(
        *this, id<Dimensions>{}, index);
  }

  reference operator[](std::size_t index) const
    requires(AccessMode != access_mode::atomic && Dimensions == 1)
  {
    return m_impl->m_data[m_impl->m_offset[0] + index];
  }

  /* Deprecated */
  operator atomic<DataT, access::address_space::global_space>() const
    requires(AccessMode == access_mode::atomic && Dimensions == 0)
  {
    return atomic<DataT, access::address_space::global_space>{
        multi_ptr<DataT, access::address_space::global_space>(m_impl->m_data)};
  }

  /* Deprecated */
  atomic<DataT, access::address_space::global_space>
  operator[](id<Dimensions> index) const
    requires(AccessMode == access_mode::atomic && Dimensions == 1)
  {
    return atomic<DataT, access::address_space::global_space>{
        multi_ptr<DataT, access::address_space::global_space>(
            m_impl->m_data + index + m_impl->m_offset)};
  }

  /* Deprecated in SYCL 2020 */
  global_ptr<value_type> get_pointer() const noexcept
    requires(AccessTarget == target::device)
  {
    return m_impl->m_data;
  }

  std::add_pointer_t<value_type> get_pointer() const noexcept
    requires(AccessTarget == target::host_task)
  {
    return m_impl->m_data;
  }

  template <access::decorated IsDecorated>
  accessor_ptr<IsDecorated> get_multi_ptr() const noexcept
    requires(AccessTarget == target::device)
  {
    return accessor_ptr<IsDecorated>{m_impl->m_data};
  }

  iterator begin() const noexcept { return iterator{this, 0}; }

  iterator end() const noexcept {
    return iterator{this, m_impl->m_accessRange.size()};
  }

  const_iterator cbegin() const noexcept { return begin(); }

  const_iterator cend() const noexcept { return end(); }

  reverse_iterator rbegin() const noexcept { return reverse_iterator{end()}; }

  reverse_iterator rend() const noexcept { return reverse_iterator{begin()}; }

  const_reverse_iterator crbegin() const noexcept { return rbegin(); }

  const_reverse_iterator crend() const noexcept { return rend(); }

private:
  friend accessor<std::remove_const_t<DataT>, Dimensions, access_mode::read,
                  AccessTarget, IsPlaceholder>;
  friend accessor<const DataT, Dimensions, access_mode::read, AccessTarget,
                  IsPlaceholder>;
  friend accessor<std::remove_const_t<DataT>, Dimensions, access_mode::write,
                  AccessTarget, IsPlaceholder>;
  friend accessor<const DataT, Dimensions, access_mode::write, AccessTarget,
                  IsPlaceholder>;
  friend iterator;
  friend const_iterator;
  friend std::hash<accessor>;
  friend const void *detail::get_accessor_data_pointer<>(const accessor &);

  template <typename AllocatorT>
  void check_range(const buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                   range<Dimensions> accessRange,
                   id<Dimensions> accessOffset = {}) {
    for (int i = 0; i < Dimensions; i++)
      if ((accessOffset[i] + accessRange[i]) > bufferRef.get_range()[i])
        throw exception(errc::invalid);
  }

  void check_props() {
    if (has_property<property::no_init>() && AccessMode == access_mode::read)
      throw exception(errc::invalid);
  }

  reference get_nth_value(std::size_t n) const {
    if constexpr (Dimensions == 0)
      return (*this);
    else {
      return (*this)[detail::unlinearize(n, m_impl->m_accessRange)];
    }
  }

  std::shared_ptr<detail::accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<
          detail::accessor_impl<std::remove_const_t<DataT>, Dimensions>>()};
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::constant_buffer,
               IsPlaceholder> {
public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = range<1>{1};
    m_impl->m_props = propList;
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = range<1>{1};
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_props = propList;
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_props = propList;
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = bufferRef.get_range();
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_props = propList;
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_props = propList;
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, id<Dimensions> accessOffset,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_props = propList;
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_props = propList;
    register_accessor(commandGroupHandlerRef, *this);
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           handler &commandGroupHandlerRef, range<Dimensions> accessRange,
           id<Dimensions> accessOffset, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_bufferRange = bufferRef.get_range();
    m_impl->m_accessRange = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_handler = &commandGroupHandlerRef;
    m_impl->m_props = propList;
    register_accessor(commandGroupHandlerRef, *this);
  }

  friend bool operator==(const accessor &, const accessor &) = default;

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  bool is_placeholder() const { return m_impl->m_handler == nullptr; }

  // Not in spec
  std::size_t size() const noexcept { return m_impl->m_accessRange.size(); }

  std::size_t get_size() const noexcept { return get_count() * sizeof(DataT); }

  std::size_t get_count() const noexcept {
    return m_impl->m_accessRange.size();
  }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return m_impl->m_accessRange;
  }

  id<Dimensions> get_offset() const
    requires(Dimensions > 0)
  {
    return m_impl->m_offset;
  }

  operator reference() const
    requires(Dimensions == 0)
  {
    return m_impl->m_data[0];
  }

  reference operator[](id<Dimensions> index) const
    requires(Dimensions > 0)
  {
    const auto offset = m_impl->m_offset;
    const auto buffer_range = m_impl->m_bufferRange;
    const auto linear_index = detail::linearize(offset + index, buffer_range);
    return m_impl->m_data[linear_index];
  }

  auto operator[](std::size_t index) const
    requires(Dimensions > 1)
  {
    return detail::subscript<decltype(*this), Dimensions, 0>(
        *this, id<Dimensions>{}, index);
  }

  reference operator[](std::size_t index) const
    requires(Dimensions == 1)
  {
    return m_impl->m_data[index];
  }

  constant_ptr<DataT> get_pointer() const noexcept { return m_impl->m_data; }

private:
  friend std::hash<accessor>;
  friend const void *detail::get_accessor_data_pointer<>(const accessor &);

  std::shared_ptr<detail::accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<
          detail::accessor_impl<std::remove_const_t<DataT>, Dimensions>>()};
};

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &,
         detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &,
         detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
         detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
         id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
         detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
         id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &,
         detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &,
         detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
         detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
         id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>,
         const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
         detail::accessor_tag<AccessMode, AccessTarget>, const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
accessor(buffer<DataT, Dimensions, AllocatorT> &, handler &, range<Dimensions>,
         id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>,
         const property_list &)
    -> accessor<DataT, Dimensions, AccessMode, AccessTarget,
                access::placeholder::false_t>;

} // namespace sycl

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::target AccessTarget, sycl::access::placeholder IsPlaceholder>
struct std::hash<sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                                IsPlaceholder>> {
  std::size_t
  operator()(const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                                  IsPlaceholder> &acc) const noexcept {
    return std::hash<decltype(acc.m_impl)>{}(acc.m_impl);
  }
};

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::target AccessTarget, sycl::access::placeholder IsPlaceholder>
const void *sycl::detail::get_accessor_data_pointer(
    const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                         IsPlaceholder> &acc) {
  return static_cast<const void *>(acc.m_impl->m_data);
}
