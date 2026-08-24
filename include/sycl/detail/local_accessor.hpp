#pragma once

#include "access.hpp"
#include "handler.hpp"
#include "multi_ptr.hpp"
#include "util.accessor_iterator.hpp"
#include "util.linearize.hpp"
#include "util.subscript.hpp"

namespace sycl::detail {

template <typename DataT, int Dimensions> struct local_accessor_impl {

  ~local_accessor_impl() {
    std::allocator<std::remove_const_t<DataT>>{}.deallocate(m_data,
                                                            m_range.size());
  }

  DataT *m_data;
  range<Dimensions == 0 ? 1 : Dimensions> m_range;
  property_list m_props;
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          target AccessTarget, access::placeholder IsPlaceholder>
const void *
get_accessor_data_pointer(const accessor<DataT, Dimensions, AccessMode,
                                         AccessTarget, IsPlaceholder> &);

} // namespace sycl::detail

namespace sycl {

template <typename DataT, int Dimensions> class local_accessor {
public:
  using value_type = DataT;
  using reference = value_type &;
  using const_reference = const DataT &;
  template <access::decorated IsDecorated>
  using accessor_ptr =
      multi_ptr<value_type, access::address_space::local_space, IsDecorated>;
  using iterator = detail::accessor_iterator<value_type, local_accessor>;
  using const_iterator =
      detail::accessor_iterator<const value_type, local_accessor>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  local_accessor() = default;
  local_accessor(const local_accessor &) = default;

  local_accessor(handler &commandGroupHandlerRef,
                 const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = std::allocator<std::remove_const_t<DataT>>{}.allocate(1);
    m_impl->m_props = propList;
    detail::registerAccessor(commandGroupHandlerRef, *this);
  }

  local_accessor(range<Dimensions> allocationSize,
                 handler &commandGroupHandlerRef,
                 const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = std::allocator<std::remove_const_t<DataT>>{}.allocate(
        allocationSize.size());
    m_impl->m_range = allocationSize;
    m_impl->m_props = propList;
    detail::registerAccessor(commandGroupHandlerRef, *this);
  }

  local_accessor(
      const local_accessor<std::remove_const_t<DataT>, Dimensions> &other)
    requires(std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  local_accessor(const local_accessor<const DataT, Dimensions> &other)
    requires(!std::is_const_v<DataT>)
      : m_impl{other.m_impl} {}

  friend bool operator==(const local_accessor &,
                         const local_accessor &) = default;

  void swap(local_accessor &other) { std::swap(m_impl, other.m_impl); }

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type size() const noexcept {
    return empty() ? 0 : m_impl->m_range.size();
  }

  size_type max_size() const noexcept { return size(); }

  bool empty() const noexcept { return m_impl->m_data == nullptr; }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return empty() ? range<Dimensions>{} : m_impl->m_range;
  }

  operator reference() const
    requires(Dimensions == 0)
  {
    return m_impl->m_data[0];
  }

  /* Available only when: (!std::is_const_v<DataT> && Dimensions == 0) */
  const local_accessor &operator=(const value_type &other) const;

  const local_accessor &operator=(value_type &&other) const
    requires(!std::is_const_v<DataT> && Dimensions == 0)
  {
    m_impl->m_data[0] = other;
    return *this;
  }

  reference operator[](id<Dimensions> index) const
    requires(Dimensions > 0)
  {
    return m_impl->m_data[detail::linearize(index, m_impl->m_range)];
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
    return m_impl->m_data[index];
  }

  /* Deprecated in SYCL 2020 */
  local_ptr<value_type> get_pointer() const noexcept { return m_impl->m_data; }

  template <access::decorated IsDecorated>
  accessor_ptr<IsDecorated> get_multi_ptr() const noexcept {
    return accessor_ptr<IsDecorated>{m_impl->m_data};
  }

  iterator begin() const noexcept { return iterator{this, 0}; }

  iterator end() const noexcept {
    return iterator{this, m_impl->m_range.size()};
  }

  const_iterator cbegin() const noexcept { return begin(); }

  const_iterator cend() const noexcept { return end(); }

  reverse_iterator rbegin() const noexcept { return reverse_iterator{end()}; }

  reverse_iterator rend() const noexcept { return reverse_iterator{begin()}; }

  const_reverse_iterator crbegin() const noexcept { return rbegin(); }

  const_reverse_iterator crend() const noexcept { return rend(); }

private:
  friend local_accessor<std::remove_const_t<DataT>, Dimensions>;
  friend local_accessor<const DataT, Dimensions>;
  friend iterator;
  friend const_iterator;
  friend std::hash<local_accessor>;

  reference get_nth_value(std::size_t n) const {
    if constexpr (Dimensions == 0)
      return (*this);
    else
      return (*this)[detail::unlinearize(n, m_impl->m_range)];
  }

  std::shared_ptr<
      detail::local_accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<detail::local_accessor_impl<
          std::remove_const_t<DataT>, Dimensions>>()};
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder>
class accessor<DataT, Dimensions, AccessMode, target::local, IsPlaceholder> {
public:
  using value_type = DataT;
  using reference = DataT &;
  using const_reference = const DataT &;

  accessor(handler &commandGroupHandlerRef, const property_list &propList = {})
    requires(Dimensions == 0)
  {
    m_impl->m_data = std::allocator<std::remove_const_t<DataT>>{}.allocate(1);
    m_impl->m_range = range<1>{1};
    m_impl->m_props = propList;
    register_accessor(commandGroupHandlerRef, *this);
  }

  accessor(range<Dimensions> allocationSize, handler &commandGroupHandlerRef,
           const property_list &propList = {})
    requires(Dimensions > 0)
  {
    m_impl->m_data = std::allocator<std::remove_const_t<DataT>>{}.allocate(
        allocationSize.size());
    m_impl->m_range = allocationSize;
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

  // Not in spec
  std::size_t size() const noexcept { return get_count(); }

  std::size_t get_size() const { return get_count() * sizeof(DataT); }

  std::size_t get_count() const { return m_impl->m_range.size(); }

  range<Dimensions> get_range() const
    requires(Dimensions > 0)
  {
    return m_impl->m_range;
  }

  operator reference() const
    requires(AccessMode == access_mode::read_write && Dimensions == 0)
  {
    return m_impl->m_data[0];
  }

  reference operator[](id<Dimensions> index) const
    requires(AccessMode == access_mode::read_write && Dimensions > 0)
  {
    return m_impl->m_data[detail::linearize(index, m_impl->m_range)];
  }

  decltype(auto) operator[](std::size_t index) const
    requires(Dimensions > 1)
  {
    return detail::subscript<decltype(*this), Dimensions, 0>(
        *this, id<Dimensions>{}, index);
  }

  reference operator[](std::size_t index) const
    requires(AccessMode == access_mode::read_write && Dimensions == 1)
  {
    return m_impl->m_data[index];
  }

  operator atomic<DataT, access::address_space::local_space>() const
    requires(AccessMode == access_mode::atomic && Dimensions == 0)
  {
    return atomic<DataT, access::address_space::local_space>{
        multi_ptr<DataT, access::address_space::local_space>{
            &m_impl->m_data[0]}};
  }

  atomic<DataT, access::address_space::local_space>
  operator[](id<Dimensions> index) const
    requires(AccessMode == access_mode::atomic && Dimensions > 0)
  {
    return atomic<DataT, access::address_space::local_space>{
        multi_ptr<DataT, access::address_space::local_space>{
            &m_impl->m_data[detail::linearize(index, m_impl->m_range)]}};
  }

  atomic<DataT, access::address_space::local_space>
  operator[](std::size_t index) const
    requires(AccessMode == access_mode::atomic && Dimensions == 1)
  {
    return atomic<DataT, access::address_space::local_space>{
        multi_ptr<DataT, access::address_space::local_space>{
            &m_impl->m_data[index]}};
  }

  local_ptr<DataT> get_pointer() const noexcept { return m_impl->m_data; }

private:
  friend std::hash<accessor>;
  friend const void *detail::get_accessor_data_pointer<>(const accessor &);

  std::shared_ptr<
      detail::local_accessor_impl<std::remove_const_t<DataT>, Dimensions>>
      m_impl{std::make_shared<detail::local_accessor_impl<
          std::remove_const_t<DataT>, Dimensions>>()};
};

} // namespace sycl

template <typename DataT, int Dimensions>
struct std::hash<sycl::local_accessor<DataT, Dimensions>> {
  std::size_t operator()(
      const sycl::local_accessor<DataT, Dimensions> &acc) const noexcept {
    return std::hash<decltype(acc.m_impl)>{}(acc.m_impl);
  }
};
