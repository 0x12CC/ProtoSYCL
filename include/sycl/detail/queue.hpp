#pragma once

#include "context.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "property.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "util.descriptor.hpp"
#include "util.task.hpp"
#include "util.task_graph.hpp"

namespace sycl::detail {

struct unnamed_kernel;

struct queue_impl {

  void throw_asynchronous() {
    if (m_asyncExceptions.empty())
      return;
    exception_list list = detail::create_exception_list(m_asyncExceptions);
    if (m_asyncHandler.has_value())
      m_asyncHandler.value()(list);
    else if (m_context.m_asyncHandler.has_value())
      m_context.m_asyncHandler.value()(list);
    m_asyncExceptions.clear();
  }

  std::vector<event> m_events;
  context m_context{get_default_context()};
  device m_device;
  std::optional<async_handler> m_asyncHandler;
  std::vector<std::exception_ptr> m_asyncExceptions;
  property_list m_props;
};

} // namespace sycl::detail

namespace sycl {

namespace info::queue {
struct context : detail::descriptor<sycl::context> {};
struct device : detail::descriptor<sycl::device> {};
} // namespace info::queue

class queue {
public:
  explicit queue(const property_list &propList = {}) {
    m_impl->m_props = propList;
  }

  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {}) {
    m_impl->m_asyncHandler = asyncHandler;
    m_impl->m_props = propList;
  }

  template <detail::DeviceSelector DeviceSelector>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {}) {
    m_impl->m_device = device{deviceSelector};
    m_impl->m_props = propList;
  }

  template <detail::DeviceSelector DeviceSelector>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {}) {
    m_impl->m_device = device{deviceSelector};
    m_impl->m_asyncHandler = asyncHandler;
    m_impl->m_props = propList;
  }

  explicit queue(const device &syclDevice, const property_list &propList = {}) {
    m_impl->m_device = syclDevice;
    m_impl->m_props = propList;
  }

  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {}) {
    m_impl->m_device = syclDevice;
    m_impl->m_asyncHandler = asyncHandler;
    m_impl->m_props = propList;
  }

  template <detail::DeviceSelector DeviceSelector>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const property_list &propList = {}) {
    m_impl->m_context = syclContext;
    m_impl->m_device = device{deviceSelector};
    m_impl->m_props = propList;
  }

  template <detail::DeviceSelector DeviceSelector>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {}) {
    m_impl->m_context = syclContext;
    m_impl->m_device = device{deviceSelector};
    m_impl->m_asyncHandler = asyncHandler;
    m_impl->m_props = propList;
  }

  explicit queue(const context &syclContext, const device &syclDevice,
                 const property_list &propList = {}) {
    m_impl->m_context = syclContext;
    m_impl->m_device = syclDevice;
    m_impl->m_props = propList;
  }

  explicit queue(const context &syclContext, const device &syclDevice,
                 const async_handler &asyncHandler,
                 const property_list &propList = {}) {
    m_impl->m_context = syclContext;
    m_impl->m_device = syclDevice;
    m_impl->m_asyncHandler = asyncHandler;
    m_impl->m_props = propList;
  }

  friend bool operator==(const queue &lhs, const queue &rhs) {
    return lhs.m_impl == rhs.m_impl;
  }

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  backend get_backend() const noexcept { return get_context().get_backend(); }

  context get_context() const { return m_impl->m_context; }

  device get_device() const { return m_impl->m_device; }

  bool is_in_order() const { return has_property<property::queue::in_order>(); }

  template <typename Param> typename Param::return_type get_info() const;

  template <>
  typename info::queue::context::return_type
  get_info<info::queue::context>() const {
    return get_context();
  }

  template <>
  typename info::queue::device::return_type
  get_info<info::queue::device>() const {
    return get_device();
  }

  template <typename Param>
  typename Param::return_type get_backend_info() const;

  template <typename T> event submit(T cgf) {
    handler h;
    cgf(h);
    if (is_in_order() && !m_impl->m_events.empty())
      h.m_task.m_dependencies.push_back(m_impl->m_events.back());
    auto event = detail::get_task_graph().add_task(
        h.m_task, m_impl->m_asyncExceptions,
        [m_impl = m_impl] { m_impl->throw_asynchronous(); });
    m_impl->m_events.push_back(event);
    return event;
  }

  template <typename T> event submit(T cgf, const queue &secondaryQueue) {
    return submit(cgf);
  }

  void wait() {
    for (auto &event : m_impl->m_events)
      event.wait();
  }

  void wait_and_throw() {
    wait();
    throw_asynchronous();
  }

  void throw_asynchronous() { m_impl->throw_asynchronous(); }

  /* -- convenience shortcuts -- */

  template <typename KernelName = detail::unnamed_kernel, typename KernelType>
  event single_task(const KernelType &kernelFunc) {
    return submit([&](handler &cgh) {
      cgh.single_task<KernelName, KernelType>(kernelFunc);
    });
  }

  template <typename KernelName = detail::unnamed_kernel, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.single_task<KernelName, KernelType>(kernelFunc);
    });
  }

  template <typename KernelName = detail::unnamed_kernel, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.single_task<KernelName, KernelType>(kernelFunc);
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(range<Dims> numWorkItems, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.parallel_for<KernelName, Dims, Rest...>(
          numWorkItems, std::forward<Rest...>(rest...));
    });
  }

  // Not in SYCL spec
  template <typename KernelName = detail::unnamed_kernel, typename... Rest>
  event parallel_for(std::size_t numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, 1, Rest...>(range(numWorkItems),
                                                std::forward<Rest...>(rest)...);
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(range<Dims> numWorkItems, event depEvent, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.parallel_for<KernelName, Dims, Rest...>(
          numWorkItems, std::forward<Rest...>(rest...));
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(range<Dims> numWorkItems,
                     const std::vector<event> &depEvents, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.parallel_for<KernelName, Dims, Rest...>(
          numWorkItems, std::forward<Rest...>(rest...));
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.parallel_for<KernelName, Dims, Rest...>(
          executionRange, std::forward<Rest...>(rest...));
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, event depEvent,
                     Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.parallel_for<KernelName, Dims, Rest...>(
          executionRange, std::forward<Rest...>(rest...));
    });
  }

  // Parameter pack acts as-if: Reductions&&... reductions, const KernelType
  // &kernelFunc
  template <typename KernelName = detail::unnamed_kernel, int Dims,
            typename... Rest>
  event parallel_for(nd_range<Dims> executionRange,
                     const std::vector<event> &depEvents, Rest &&...rest) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.parallel_for<KernelName, Dims, Rest...>(
          executionRange, std::forward<Rest...>(rest...));
    });
  }

  /* -- USM functions -- */

  event memcpy(void *dest, const void *src, std::size_t numBytes) {
    return submit([&](handler &cgh) { cgh.memcpy(dest, src, numBytes); });
  }
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.memcpy(dest, src, numBytes);
    });
  }
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.memcpy(dest, src, numBytes);
    });
  }

  template <typename T> event copy(const T *src, T *dest, std::size_t count) {
    return submit([&](handler &cgh) { cgh.copy(src, dest, count); });
  }
  template <typename T>
  event copy(const T *src, T *dest, std::size_t count, event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.copy(src, dest, count);
    });
  }
  template <typename T>
  event copy(const T *src, T *dest, std::size_t count,
             const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.copy(src, dest, count);
    });
  }

  event memset(void *ptr, int value, std::size_t numBytes) {
    return submit([&](handler &cgh) { cgh.memset(ptr, value, numBytes); });
  }
  event memset(void *ptr, int value, std::size_t numBytes, event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.memset(ptr, value, numBytes);
    });
  }
  event memset(void *ptr, int value, std::size_t numBytes,
               const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.memset(ptr, value, numBytes);
    });
  }

  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count) {
    return submit([&](handler &cgh) { cgh.fill(ptr, pattern, count); });
  }
  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count, event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.fill(ptr, pattern, count);
    });
  }
  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count,
             const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.fill(ptr, pattern, count);
    });
  }

  event prefetch(const void *ptr, std::size_t numBytes) {
    return submit([&](handler &cgh) { cgh.prefetch(ptr, numBytes); });
  }
  event prefetch(const void *ptr, std::size_t numBytes, event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.prefetch(ptr, numBytes);
    });
  }
  event prefetch(const void *ptr, std::size_t numBytes,
                 const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.prefetch(ptr, numBytes);
    });
  }

  event mem_advise(const void *ptr, std::size_t numBytes, int advice) {
    return submit([&](handler &cgh) { cgh.mem_advise(ptr, numBytes, advice); });
  }
  event mem_advise(const void *ptr, std::size_t numBytes, int advice,
                   event depEvent) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvent);
      cgh.mem_advise(ptr, numBytes, advice);
    });
  }
  event mem_advise(const void *ptr, std::size_t numBytes, int advice,
                   const std::vector<event> &depEvents) {
    return submit([&](handler &cgh) {
      cgh.depends_on(depEvents);
      cgh.mem_advise(ptr, numBytes, advice);
    });
  }

  /// Placeholder accessor shortcuts

  // Explicit copy functions

  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src,
             std::shared_ptr<DestT> dest) {
    return submit([&](handler &cgh) {
      cgh.require(src);
      cgh.copy(src, dest);
    });
  }

  template <typename SrcT, typename DestT, int DestDims, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  event copy(std::shared_ptr<SrcT> src,
             accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
    return submit([&](handler &cgh) {
      cgh.require(dest);
      cgh.copy(src, dest);
    });
  }

  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsPlaceholder, typename DestT>
  event copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsPlaceholder> src,
             DestT *dest) {
    return submit([&](handler &cgh) {
      cgh.require(src);
      cgh.copy(src, dest);
    });
  }

  template <typename SrcT, typename DestT, int DestDims, access_mode DestMode,
            target DestTgt, access::placeholder IsPlaceholder>
  event copy(const SrcT *src,
             accessor<DestT, DestDims, DestMode, DestTgt, IsPlaceholder> dest) {
    return submit([&](handler &cgh) {
      cgh.require(dest);
      cgh.copy(src, dest);
    });
  }

  template <typename SrcT, int SrcDims, access_mode SrcMode, target SrcTgt,
            access::placeholder IsSrcPlaceholder, typename DestT, int DestDims,
            access_mode DestMode, target DestTgt,
            access::placeholder IsDestPlaceholder>
  event
  copy(accessor<SrcT, SrcDims, SrcMode, SrcTgt, IsSrcPlaceholder> src,
       accessor<DestT, DestDims, DestMode, DestTgt, IsDestPlaceholder> dest) {
    return submit([&](handler &cgh) {
      cgh.require(src);
      cgh.require(dest);
      cgh.copy(src, dest);
    });
  }

  template <typename T, int Dims, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  event update_host(accessor<T, Dims, Mode, Tgt, IsPlaceholder> acc) {
    return submit([&](handler &cgh) {
      cgh.require(acc);
      cgh.update_host(acc);
    });
  }

  template <typename T, int Dims, access_mode Mode, target Tgt,
            access::placeholder IsPlaceholder>
  event fill(accessor<T, Dims, Mode, Tgt, IsPlaceholder> dest, const T &src) {
    return submit([&](handler &cgh) {
      cgh.require(dest);
      cgh.fill(dest, src);
    });
  }

private:
  friend std::hash<queue>;
  std::shared_ptr<detail::queue_impl> m_impl{
      std::make_shared<detail::queue_impl>()};
};

} // namespace sycl

template <> struct std::hash<sycl::queue> {
  std::size_t operator()(const sycl::queue &q) const noexcept {
    return std::hash<decltype(q.m_impl)>{}(q.m_impl);
  }
};
