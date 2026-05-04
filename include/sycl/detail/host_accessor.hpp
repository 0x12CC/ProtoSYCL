#pragma once

#include "access.hpp"
#include "accessor_tag.hpp"
#include "buffer.hpp"
#include "exception.hpp"
#include "property.hpp"
#include "util.accessor_iterator.hpp"
#include "util.linearize.hpp"
#include "util.subscript.hpp"
#include "util.task.hpp"
#include "util.task_graph.hpp"

namespace sycl {

namespace detail {

template <typename DataT, int Dimensions> struct host_accessor_impl {
  DataT *m_data = nullptr;
  range<Dimensions == 0 ? 1 : Dimensions> m_buffer_range;
  range<Dimensions == 0 ? 1 : Dimensions> m_access_range;
  id<Dimensions == 0 ? 1 : Dimensions> m_offset;
  property_list m_props;
};

} // namespace detail

template <typename DataT, int Dimensions, access_mode AccessMode>
class host_accessor {
public:
  using value_type =
      std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;
  using iterator = detail::accessor_iterator<value_type, host_accessor>;
  using const_iterator =
      detail::accessor_iterator<const value_type, host_accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  host_accessor() = default;
  host_accessor(const host_accessor &) = default;

  template <typename AllocatorT>
  host_accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
                const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
    wait_for_data();
  }

  template <typename AllocatorT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = bufferRef.get_range();
    m_impl->m_props = propList;
    check_props();
    wait_for_data();
  }

  template <typename AllocatorT, typename TagT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef, TagT tag,
                const property_list &propList = {})
    requires(Dimensions > 0)
      : host_accessor{bufferRef, propList} {}

  template <typename AllocatorT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                range<Dimensions> accessRange,
                const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = accessRange;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange);
    check_props();
    wait_for_data();
  }

  template <typename AllocatorT, typename TagT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                range<Dimensions> accessRange, TagT tag,
                const property_list &propList = {})
    requires(Dimensions > 0)
      : host_accessor{bufferRef, accessRange, propList} {}

  template <typename AllocatorT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                range<Dimensions> accessRange, id<Dimensions> accessOffset,
                const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_props = propList;
    check_range(bufferRef, accessRange, accessOffset);
    check_props();
    wait_for_data();
  }

  template <typename AllocatorT, typename TagT>
  host_accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
                range<Dimensions> accessRange, id<Dimensions> accessOffset,
                TagT tag, const property_list &propList = {})
    requires(Dimensions > 0)
      : host_accessor{bufferRef, accessRange, accessOffset, propList} {}

  /* -- common interface members -- */

  host_accessor(const host_accessor<std::remove_const_t<DataT>, Dimensions,
                                    access_mode::read> &other)
    requires(AccessMode == access_mode::read && std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  host_accessor(
      const host_accessor<const DataT, Dimensions, access_mode::read> &other)
    requires(AccessMode == access_mode::read && !std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  host_accessor(const host_accessor<std::remove_const_t<DataT>, Dimensions,
                                    access_mode::read_write> &other)
    requires(AccessMode == access_mode::read && std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  host_accessor(const host_accessor<const DataT, Dimensions,
                                    access_mode::read_write> &other)
    requires(AccessMode == access_mode::read && !std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  friend bool operator==(const host_accessor &,
                         const host_accessor &) = default;

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  void swap(host_accessor &other) { std::swap(m_impl, other.m_impl); }

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type size() const noexcept { return m_impl->m_access_range.size(); }

  size_type max_size() const noexcept {
    return m_impl->m_data ? m_impl->m_buffer_range.size() : 0;
  }

  bool empty() const noexcept { return size() == 0; }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return m_impl->m_access_range;
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

  const host_accessor &operator=(const value_type &other) const
    requires(AccessMode != access_mode::read && Dimensions == 0)
  {
    m_impl->m_data[0] = other;
    return *this;
  }

  const host_accessor &operator=(value_type &&other) const
    requires(AccessMode != access_mode::read && Dimensions == 0)
  {
    m_impl->m_data[0] = other;
    return *this;
  }

  reference operator[](id<Dimensions> index) const
    requires(Dimensions > 0)
  {
    const auto offset = m_impl->m_offset;
    const auto buffer_range = m_impl->m_buffer_range;
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
    requires(Dimensions == 1)
  {
    return m_impl->m_data[m_impl->m_offset[0] + index];
  }

  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return m_impl->m_data;
  }

  iterator begin() const noexcept { return iterator{this, 0}; }

  iterator end() const noexcept {
    return iterator{this, m_impl->m_access_range.size()};
  }

  const_iterator cbegin() const noexcept { return begin(); }

  const_iterator cend() const noexcept { return end(); }

  reverse_iterator rbegin() const noexcept { return reverse_iterator{end()}; }

  reverse_iterator rend() const noexcept { return reverse_iterator{begin()}; }

  const_reverse_iterator crbegin() const noexcept { return rbegin(); }

  const_reverse_iterator crend() const noexcept { return rend(); }

private:
  friend host_accessor<std::remove_const_t<DataT>, Dimensions,
                       access_mode::read>;
  friend host_accessor<const DataT, Dimensions, access_mode::read>;
  friend iterator;
  friend const_iterator;
  friend std::hash<host_accessor>;

  reference get_nth_value(std::size_t n) const {
    if constexpr (Dimensions == 0)
      return (*this);
    else
      return (*this)[detail::unlinearize(n, m_impl->m_access_range)];
  }

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

  void wait_for_data() {
    detail::task t{};
    t.m_action = [] {};
    t.add_requisite(detail::requisite{
        m_impl->m_data, m_impl->m_buffer_range.size() * sizeof(DataT),
        detail::requisite::access_mode::read_write});
    std::vector<std::exception_ptr> async_exceptions;
    auto event = detail::get_task_graph().add_task(t, async_exceptions);
    event.wait();
  }

  std::shared_ptr<
      detail::host_accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<detail::host_accessor_impl<
          std::remove_const_t<DataT>, Dimensions>>()};
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::host_buffer,
               IsPlaceholder> {
public:
  using value_type =
      std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  accessor(buffer<DataT, 1, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = range<1>{1};
    m_impl->m_props = propList;
  }

  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = bufferRef.get_range();
    m_impl->m_props = propList;
  }

  /* Available only when: (Dimensions > 0) */
  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = accessRange;
    m_impl->m_props = propList;
  }

  /* Available only when: (Dimensions > 0) */
  template <typename AllocatorT>
  accessor(buffer<DataT, Dimensions, AllocatorT> &bufferRef,
           range<Dimensions> accessRange, id<Dimensions> accessOffset,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = detail::get_buffer_data(bufferRef);
    m_impl->m_buffer_range = bufferRef.get_range();
    m_impl->m_access_range = accessRange;
    m_impl->m_offset = accessOffset;
    m_impl->m_props = propList;
  }

  friend bool operator==(const accessor &, const accessor &) = default;

  bool is_placeholder() const { return false; }

  // Not in spec
  std::size_t size() const noexcept { return get_count(); }

  std::size_t get_size() const { return get_count() * sizeof(DataT); }

  std::size_t get_count() const { return m_impl->m_access_range.size(); }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return m_impl->m_access_range;
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
    const auto buffer_range = m_impl->m_buffer_range;
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
    requires(Dimensions == 1)
  {
    return m_impl->m_data[m_impl->m_offset[0] + index];
  }

  std::add_pointer_t<value_type> get_pointer() const noexcept {
    return m_impl->m_data;
  }

private:
  friend std::hash<accessor>;

  std::shared_ptr<
      detail::host_accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<detail::host_accessor_impl<
          std::remove_const_t<DataT>, Dimensions>>()};
};

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &,
              detail::accessor_tag<AccessMode, AccessTarget>)
    -> host_accessor<DataT, Dimensions, AccessMode>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
              detail::accessor_tag<AccessMode, AccessTarget>)
    -> host_accessor<DataT, Dimensions, AccessMode>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
              id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>)
    -> host_accessor<DataT, Dimensions, AccessMode>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &,
              detail::accessor_tag<AccessMode, AccessTarget>,
              const property_list &)
    -> host_accessor<DataT, Dimensions, AccessMode>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
              detail::accessor_tag<AccessMode, AccessTarget>,
              const property_list &)
    -> host_accessor<DataT, Dimensions, AccessMode>;

template <typename DataT, int Dimensions, typename AllocatorT,
          access_mode AccessMode, target AccessTarget>
host_accessor(buffer<DataT, Dimensions, AllocatorT> &, range<Dimensions>,
              id<Dimensions>, detail::accessor_tag<AccessMode, AccessTarget>,
              const property_list &)
    -> host_accessor<DataT, Dimensions, AccessMode>;

} // namespace sycl

template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
struct std::hash<sycl::host_accessor<DataT, Dimensions, AccessMode>> {
  std::size_t
  operator()(const sycl::host_accessor<DataT, Dimensions, AccessMode> &acc)
      const noexcept {
    return std::hash<decltype(acc.m_impl)>{}(acc.m_impl);
  }
};
