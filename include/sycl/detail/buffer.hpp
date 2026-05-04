#pragma once

#include <any>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>

#include "access.hpp"
#include "context.hpp"
#include "id.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "util.forward.hpp"
#include "util.linearize.hpp"
#include "util.task.hpp"
#include "util.task_graph.hpp"

namespace sycl {

template <class T> using buffer_allocator = std::allocator<T>;

template <typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
class buffer;

namespace detail {

struct buffer_impl;

template <typename T, int Dimensions, typename AllocatorT>
std::remove_const_t<T> *
get_buffer_data(const buffer<T, Dimensions, AllocatorT> &);

struct buffer_impl {

  ~buffer_impl() {
    detail::task t{};
    t.m_action = [] {};
    t.add_requisite(detail::requisite{
        m_data, m_dataSize, detail::requisite::access_mode::read_write});
    std::vector<std::exception_ptr> async_exceptions;
    auto event = detail::get_task_graph().add_task(t, async_exceptions);
    event.wait();

    m_deallocate();
  }

  std::shared_ptr<buffer_impl> parent = nullptr;
  void *m_data = nullptr;
  std::size_t m_dataSize = 0;
  std::function<void()> m_deallocate = [] {};
  std::any m_hostData;
  std::any m_allocator;
  bool m_writeBack = true;
  property_list m_props;
};

template <typename Container, typename T>
concept ContiguousContainer = requires(Container container) {
  { std::data(container) } -> std::convertible_to<T *>;
  { std::size(container) } -> std::convertible_to<std::size_t>;
};

} // namespace detail

namespace property::buffer {

class use_host_ptr {
public:
  use_host_ptr() = default;
};

class use_mutex {
public:
  use_mutex(std::mutex &mutexRef) : m_value{&mutexRef} {}

  std::mutex *get_mutex_ptr() const { return m_value; }

private:
  std::mutex *m_value;
};

class context_bound {
public:
  context_bound(context boundContext) : m_boundContext{boundContext} {}

  context get_context() const { return m_boundContext; }

private:
  context m_boundContext;
};

} // namespace property::buffer

template <>
struct is_property<property::buffer::use_host_ptr> : std::true_type {};
template <> struct is_property<property::buffer::use_mutex> : std::true_type {};
template <>
struct is_property<property::buffer::context_bound> : std::true_type {};

template <typename T, int Dimensions, typename AllocatorT> class buffer {
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  buffer(const range<Dimensions> &bufferRange,
         const property_list &propList = {})
      : buffer{bufferRange, AllocatorT{}, propList} {}

  buffer(const range<Dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {}) {
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = bufferRange;
    allocate_memory();
  }

  buffer(T *hostData, const range<Dimensions> &bufferRange,
         const property_list &propList = {})
    requires(!std::is_const_v<T>)
      : buffer{hostData, bufferRange, AllocatorT{}, propList} {}

  buffer(T *hostData, const range<Dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {})
    requires(!std::is_const_v<T>)
  {
    m_impl->m_hostData = hostData;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = bufferRange;
    allocate_memory();
  }

  buffer(const T *hostData, const range<Dimensions> &bufferRange,
         const property_list &propList = {})
      : buffer{hostData, bufferRange, AllocatorT{}, propList} {}

  buffer(const T *hostData, const range<Dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {}) {
    m_impl->m_hostData = hostData;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = bufferRange;
    allocate_memory();
  }

  template <detail::ContiguousContainer<T> Container>
  buffer(Container &container, AllocatorT allocator,
         const property_list &propList = {})
    requires(Dimensions == 1)
  {
    m_impl->m_hostData = std::data(container);
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = range<1>{std::size(container)};
    allocate_memory();
  }

  template <detail::ContiguousContainer<T> Container>
  buffer(Container &container, const property_list &propList = {})
    requires(Dimensions == 1)
      : buffer{container, AllocatorT{}, propList} {}

  buffer(const std::shared_ptr<T> &hostData,
         const range<Dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {}) {
    m_impl->m_hostData = hostData;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = bufferRange;
    allocate_memory();
  }

  buffer(const std::shared_ptr<T> &hostData,
         const range<Dimensions> &bufferRange,
         const property_list &propList = {})
      : buffer{hostData, bufferRange, AllocatorT{}, propList} {}

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<Dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {}) {
    m_impl->m_hostData = hostData;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = bufferRange;
    allocate_memory();
  }

  buffer(const std::shared_ptr<T[]> &hostData,
         const range<Dimensions> &bufferRange,
         const property_list &propList = {})
      : buffer{hostData, bufferRange, AllocatorT{}, propList} {}

  template <class InputIterator>
  buffer(InputIterator first, InputIterator last, AllocatorT allocator,
         const property_list &propList = {})
    requires(Dimensions == 1)
  {
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    m_range = range<1>{static_cast<std::size_t>(std::distance(first, last))};
    allocate_memory();
    std::copy(first, last,
              static_cast<std::remove_const_t<T> *>(m_impl->m_data));
  }

  template <class InputIterator>
  buffer(InputIterator first, InputIterator last,
         const property_list &propList = {})
    requires(Dimensions == 1)
      : buffer{first, last, AllocatorT{}, propList} {}

  buffer(buffer &b, const id<Dimensions> &baseIndex,
         const range<Dimensions> &subRange) {
    m_impl->parent = b.m_impl;
    m_impl->m_allocator = b.m_impl->m_allocator;
    m_impl->m_data = static_cast<T *>(b.m_impl->m_data) +
                     detail::linearize(baseIndex, b.get_range());
    m_impl->m_props = b.m_impl->m_props;
    m_range = subRange;
  }

  friend bool operator==(const buffer &, const buffer &) = default;

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  range<Dimensions> get_range() const { return m_range; }

  std::size_t byte_size() const noexcept {
    return get_range().size() * sizeof(T);
  }

  std::size_t size() const noexcept { return get_range().size(); }

  // Deprecated
  std::size_t get_count() const { return size(); }

  // Deprecated
  std::size_t get_size() const { return byte_size(); }

  AllocatorT get_allocator() const {
    return std::any_cast<AllocatorT>(m_impl->m_allocator);
  }

  template <access_mode Mode = access_mode::read_write,
            target Targ = target::device>
  accessor<T, Dimensions, Mode, Targ> get_access(handler &commandGroupHandler) {
    return accessor<T, Dimensions, Mode, Targ>{*this, commandGroupHandler};
  }

  // Deprecated
  template <access_mode Mode>
  accessor<T, Dimensions, Mode, target::host_buffer> get_access() {
    detail::task t{};
    t.m_action = [] {};
    t.add_requisite(
        detail::requisite{m_impl->m_data, m_impl->m_dataSize,
                          detail::requisite::access_mode::read_write});
    std::vector<std::exception_ptr> async_exceptions;
    auto event = detail::get_task_graph().add_task(t, async_exceptions);
    event.wait();

    return accessor<T, Dimensions, Mode, target::host_buffer>{*this};
  }

  template <access_mode Mode = access_mode::read_write,
            target Targ = target::device>
  accessor<T, Dimensions, Mode, Targ>
  get_access(handler &commandGroupHandler, range<Dimensions> accessRange,
             id<Dimensions> accessOffset = {}) {
    return accessor<T, Dimensions, Mode, Targ>{*this, commandGroupHandler,
                                               accessRange, accessOffset};
  }

  // Deprecated
  template <access_mode Mode>
  accessor<T, Dimensions, Mode, target::host_buffer>
  get_access(range<Dimensions> accessRange, id<Dimensions> accessOffset = {}) {
    return accessor<T, Dimensions, Mode, target::host_buffer>{
        *this, accessRange, accessOffset};
  }

  template <typename... Ts> auto get_access(Ts &&...args) {
    return accessor{*this, std::forward<Ts>(args)...};
  }

  template <typename... Ts> auto get_host_access(Ts &&...args) {
    return host_accessor{*this, std::forward<Ts>(args)...};
  }

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    if constexpr (std::is_same_v<Destination, std::nullptr_t>)
      m_impl->m_writeBack = false;
    else
      m_impl->m_hostData = finalData;
  }

  void set_write_back(bool flag = true) { m_impl->m_writeBack = flag; }

  bool is_sub_buffer() const { return m_impl->parent.get() != nullptr; }

  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim,
         typename std::allocator_traits<AllocatorT>::template rebind_alloc<
             ReinterpretT>>
  reinterpret(range<ReinterpretDim> reinterpretRange) const {
    buffer<ReinterpretT, ReinterpretDim,
           typename std::allocator_traits<AllocatorT>::template rebind_alloc<
               ReinterpretT>>
        reinterpreted_buffer;
    reinterpreted_buffer.m_impl = m_impl;
    reinterpreted_buffer.m_range = reinterpretRange;
    return reinterpreted_buffer;
  }

  template <typename ReinterpretT, int ReinterpretDim = Dimensions>
  buffer<ReinterpretT, ReinterpretDim,
         typename std::allocator_traits<AllocatorT>::template rebind_alloc<
             ReinterpretT>>
  reinterpret() const
    requires(ReinterpretDim == 1 || (ReinterpretDim == Dimensions &&
                                     sizeof(ReinterpretT) == sizeof(T)))
  {
    range<ReinterpretDim> reinterpret_range;
    if constexpr (ReinterpretDim == 1)
      reinterpret_range = range<1>{byte_size() / sizeof(ReinterpretT)};
    else
      reinterpret_range = m_range;
    return reinterpret<ReinterpretT, ReinterpretDim>(reinterpret_range);
  }

private:
  friend std::hash<buffer>;
  template <typename T_, int Dimensions_, typename AllocatorT_>
  friend class buffer;

  buffer() = default;

  friend std::remove_const_t<T> *
  detail::get_buffer_data<T, Dimensions, AllocatorT>(
      const buffer<T, Dimensions, AllocatorT> &);

  void allocate_memory() {
    using DataT = std::remove_const_t<T>;
    auto host_data = m_impl->m_hostData;

    if (DataT **ptr = std::any_cast<DataT *>(&host_data)) {
      m_impl->m_data = *ptr;
      m_impl->m_dataSize = size() * sizeof(T);
      return;
    }

    if (std::shared_ptr<DataT> *ptr =
            std::any_cast<std::shared_ptr<DataT>>(&host_data)) {
      m_impl->m_data = ptr->get();
      m_impl->m_dataSize = size() * sizeof(T);
      return;
    }

    if (std::shared_ptr<DataT[]> *ptr =
            std::any_cast<std::shared_ptr<DataT[]>>(&host_data)) {
      m_impl->m_data = ptr->get();
      m_impl->m_dataSize = size() * sizeof(T);
      return;
    }

    AllocatorT allocator = get_allocator();
    m_impl->m_data = allocator.allocate(size());
    m_impl->m_dataSize = byte_size();
    m_impl->m_deallocate = [allocator, data_ptr = m_impl->m_data,
                            size = size()] mutable {
      allocator.deallocate(static_cast<std::remove_const_t<T> *>(data_ptr),
                           size);
    };

    if (const T **ptr = std::any_cast<const T *>(&host_data)) {
      std::memcpy(m_impl->m_data, *ptr, byte_size());
      host_data = std::any{};
    }
  }

  std::shared_ptr<detail::buffer_impl> m_impl{
      std::make_shared<detail::buffer_impl>()};
  range<Dimensions> m_range;
};

template <class InputIterator, class AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1,
              AllocatorT>;

template <class InputIterator>
buffer(InputIterator, InputIterator, const property_list & = {})
    -> buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template <class T, int Dimensions, class AllocatorT>
buffer(const T *, const range<Dimensions> &, AllocatorT,
       const property_list & = {}) -> buffer<T, Dimensions, AllocatorT>;

template <class T, int Dimensions>
buffer(const T *, const range<Dimensions> &, const property_list & = {})
    -> buffer<T, Dimensions>;

template <class Container, class AllocatorT>
buffer(Container &, AllocatorT, const property_list & = {})
    -> buffer<typename Container::value_type, 1, AllocatorT>;

template <class Container>
buffer(Container &, const property_list & = {})
    -> buffer<typename Container::value_type, 1>;

} // namespace sycl

namespace sycl::detail {

template <typename T, int Dimensions, typename AllocatorT>
std::remove_const_t<T> *
get_buffer_data(const buffer<T, Dimensions, AllocatorT> &bufferRef) {
  return static_cast<std::remove_const_t<T> *>(bufferRef.m_impl->m_data);
}

} // namespace sycl::detail

template <typename T, int Dimensions, typename AllocatorT>
struct std::hash<sycl::buffer<T, Dimensions, AllocatorT>> {
  std::size_t operator()(
      const sycl::buffer<T, Dimensions, AllocatorT> &bufferRef) const noexcept {
    return std::hash<decltype(bufferRef.m_impl)>{}(bufferRef.m_impl);
  }
};
