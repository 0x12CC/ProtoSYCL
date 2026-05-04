#pragma once

#include <algorithm>
#include <barrier>

#include "device_event.hpp"
#include "h_item.hpp"
#include "id.hpp"
#include "item.hpp"
#include "memory_scope.hpp"
#include "multi_ptr.hpp"
#include "range.hpp"

namespace sycl {

template <int Dimensions = 1>
  requires(1 <= Dimensions && Dimensions <= 3)
class group;

namespace detail {

template <int Dimensions>
group<Dimensions> make_group(std::barrier<> *, item<Dimensions, false>,
                             item<Dimensions, false>);

template <int Dimensions>
std::barrier<> *get_barrier(sycl::group<Dimensions> &);

} // namespace detail

template <int Dimensions>
  requires(1 <= Dimensions && Dimensions <= 3)
class group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = std::size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  /* -- common interface members -- */

  // Not in spec
  id_type get_id() const { return get_group_id(); }

  // Not in spec
  std::size_t get_id(int dimension) const { return get_group_id(dimension); }

  // Not in spec
  range<Dimensions> get_global_range() const { return m_global.get_range(); }

  // Not in spec
  std::size_t get_global_range(int dimension) const {
    return get_global_range()[dimension];
  }

  id<Dimensions> get_group_id() const { return m_item.get_id(); }

  std::size_t get_group_id(int dimension) const {
    return get_group_id()[dimension];
  }

  id<Dimensions> get_local_id() const { return m_local.get_id(); }

  std::size_t get_local_id(int dimension) const {
    return get_local_id()[dimension];
  }

  range<Dimensions> get_local_range() const { return m_local.get_range(); }

  std::size_t get_local_range(int dimension) const {
    return get_local_range()[dimension];
  }

  range<Dimensions> get_group_range() const { return m_item.get_range(); }

  std::size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  range<Dimensions> get_max_local_range() const { return get_local_range(); }

  std::size_t operator[](int dimension) const {
    return get_group_id()[dimension];
  }

  // Not in spec
  std::size_t get_linear_id() const { return get_group_linear_id(); }

  std::size_t get_group_linear_id() const { return m_item.get_linear_id(); }

  std::size_t get_local_linear_id() const { return m_local.get_linear_id(); }

  std::size_t get_group_linear_range() const {
    return get_group_range().size();
  }

  std::size_t get_local_linear_range() const {
    return get_local_range().size();
  }

  bool leader() const { return get_local_linear_id() == 0; }

  template <typename WorkItemFunctionT>
  void parallel_for_work_item(const WorkItemFunctionT &func) const {
    parallel_for_work_item(m_local.get_range(), func);
  }

  template <typename WorkItemFunctionT>
  void parallel_for_work_item(range<Dimensions> logicalRange,
                              const WorkItemFunctionT &func) const {
    for (std::size_t idx = 0; idx < logicalRange.size(); idx++) {
      const auto logical_id = detail::unlinearize(idx, logicalRange);
      const auto logical_item = detail::make_item(logical_id, logicalRange);

      const auto physical_id = logical_id % id{m_local.get_range()};
      const auto physical_item =
          detail::make_item(physical_id, m_local.get_range());

      const auto global_id = m_global + physical_id;
      const auto global_item =
          detail::make_item(global_id, m_global.get_range());

      h_item<Dimensions> item = detail::make_h_item<Dimensions>(
          global_item, logical_item, physical_item);
      func(item);
    }
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event async_work_group_copy(local_ptr<DataT> dest,
                                     global_ptr<DataT> src,
                                     std::size_t numElements) const {
    std::copy_n(src, numElements, dest);
    return detail::make_device_event();
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event async_work_group_copy(global_ptr<DataT> dest,
                                     local_ptr<DataT> src,
                                     std::size_t numElements) const {
    std::copy_n(src, numElements, dest);
    return detail::make_device_event();
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event
  async_work_group_copy(local_ptr<DataT> dest, global_ptr<DataT> src,
                        std::size_t numElements, std::size_t srcStride) const {
    for (std::size_t i = 0; i < numElements; ++i)
      dest[i] = src[i * srcStride];
    return detail::make_device_event();
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event
  async_work_group_copy(global_ptr<DataT> dest, local_ptr<DataT> src,
                        std::size_t numElements, std::size_t destStride) const {
    for (std::size_t i = 0; i < numElements; ++i)
      dest[i * destStride] = src[i];
    return detail::make_device_event();
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                     decorated_global_ptr<SrcDataT> src,
                                     std::size_t numElements) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    std::copy_n(src, numElements, dest);
    return detail::make_device_event();
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                     decorated_local_ptr<SrcDataT> src,
                                     std::size_t numElements) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    std::copy_n(src, numElements, dest);
    return detail::make_device_event();
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                     decorated_global_ptr<SrcDataT> src,
                                     std::size_t numElements,
                                     std::size_t srcStride) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    for (std::size_t i = 0; i < numElements; ++i)
      dest[i] = src[i * srcStride];
    return detail::make_device_event();
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                     decorated_local_ptr<SrcDataT> src,
                                     std::size_t numElements,
                                     std::size_t destStride) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    for (std::size_t i = 0; i < numElements; ++i)
      dest[i * destStride] = src[i];
    return detail::make_device_event();
  }

  template <typename... EventTN> void wait_for(EventTN... events) const {
    (events.wait(), ...);
  }

  friend bool operator==(const group &, const group &) = default;

private:
  item<Dimensions, false>
  get_group_item(const item<Dimensions, false> &global,
                 const item<Dimensions, false> &local) const {
    const range<Dimensions> group_range = global.get_range() / local.get_range();
    const id<Dimensions> group_id = global.get_id() / id{local.get_range()};
    return detail::make_item(group_id, group_range);
  }

  group(std::barrier<> *barrier, item<Dimensions, false> global,
        item<Dimensions, false> local)
      : m_barrier{barrier}, m_global{global}, m_local{local},
        m_item{get_group_item(global, local)} {}

  std::barrier<> *m_barrier;
  item<Dimensions, false> m_global;
  item<Dimensions, false> m_local;
  item<Dimensions, false> m_item;

  friend group<Dimensions>
  detail::make_group<Dimensions>(std::barrier<> *, item<Dimensions, false>,
                                 item<Dimensions, false>);

  friend std::barrier<> *detail::get_barrier<Dimensions>(group<Dimensions> &);
};

template <int Dimensions>
group<Dimensions> detail::make_group(std::barrier<> *barrier,
                                     item<Dimensions, false> global,
                                     item<Dimensions, false> local) {
  return group<Dimensions>{barrier, global, local};
}

template <int Dimensions>
std::barrier<> *detail::get_barrier(sycl::group<Dimensions> &group) {
  return group.m_barrier;
}

} // namespace sycl
