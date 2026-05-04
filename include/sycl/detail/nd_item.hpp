#pragma once

#include "id.hpp"
#include "device_event.hpp"
#include "group.hpp"
#include "multi_ptr.hpp"
#include "nd_range.hpp"
#include "sub_group.hpp"

namespace sycl {

template <int Dimensions> class nd_item;

namespace detail {

template <int Dimensions>
nd_item<Dimensions> make_nd_item(std::barrier<> *, id<Dimensions>,
                                 id<Dimensions>, nd_range<Dimensions>);

} // namespace detail

template <int Dimensions> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  nd_item() = delete;

  friend bool operator==(const nd_item &, const nd_item &) = default;

  id<Dimensions> get_global_id() const { return m_global; }

  std::size_t get_global_id(int dimension) const {
    return get_global_id()[dimension];
  }

  std::size_t get_global_linear_id() const {
    return detail::linearize(get_global_id(), m_range.get_global_range());
  }

  id<Dimensions> get_local_id() const { return m_local; }

  std::size_t get_local_id(int dimension) const {
    return get_local_id()[dimension];
  }

  std::size_t get_local_linear_id() const {
    return detail::linearize(get_local_id(), m_range.get_local_range());
  }

  group<Dimensions> get_group() const {
    return detail::make_group<Dimensions>(
        m_barrier, detail::make_item(m_global, m_range.get_global_range()),
        detail::make_item(m_local, m_range.get_local_range()));
  }

  sub_group get_sub_group() const {
    constexpr std::uint32_t sub_group_size = 32;
    const std::uint32_t local_id = get_local_linear_id() % sub_group_size;
    const std::uint32_t local_range =
        m_range.get_local_range().size() % sub_group_size;
    const std::uint32_t group_id = get_local_linear_id() / sub_group_size;
    const std::uint32_t number_of_sub_groups =
        (m_range.get_local_range().size() + sub_group_size - 1) /
        sub_group_size;
    return detail::make_sub_group(local_id, local_range, group_id,
                                  number_of_sub_groups, sub_group_size);
  }

  std::size_t get_group(int dimension) const {
    return get_group().get_id()[dimension];
  }

  std::size_t get_group_linear_id() const {
    return get_group().get_group_linear_id();
  }

  range<Dimensions> get_group_range() const {
    return get_group().get_group_range();
  }

  std::size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  range<Dimensions> get_global_range() const {
    return m_range.get_global_range();
  }

  std::size_t get_global_range(int dimension) const {
    return get_global_range()[dimension];
  }

  range<Dimensions> get_local_range() const {
    return m_range.get_local_range();
  }

  std::size_t get_local_range(int dimension) const {
    return get_local_range()[dimension];
  }

  // Deprecated in SYCL 2020.
  id<Dimensions> get_offset() const { return m_range.get_offset(); }

  nd_range<Dimensions> get_nd_range() const { return m_range; }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event async_work_group_copy(local_ptr<DataT> dest,
                                     global_ptr<DataT> src,
                                     std::size_t numElements) const {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event async_work_group_copy(global_ptr<DataT> dest,
                                     local_ptr<DataT> src,
                                     std::size_t numElements) const {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event
  async_work_group_copy(local_ptr<DataT> dest, global_ptr<DataT> src,
                        std::size_t numElements, std::size_t srcStride) const {
    return get_group().async_work_group_copy(dest, src, numElements, srcStride);
  }
  // Deprecated in SYCL 2020.
  template <typename DataT>
  device_event
  async_work_group_copy(global_ptr<DataT> dest, local_ptr<DataT> src,
                        std::size_t numElements, std::size_t destStride) const {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                     decorated_global_ptr<SrcDataT> src,
                                     std::size_t numElements) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                     decorated_local_ptr<SrcDataT> src,
                                     std::size_t numElements) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                     decorated_global_ptr<SrcDataT> src,
                                     std::size_t numElements,
                                     std::size_t srcStride) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    return get_group().async_work_group_copy(dest, src, numElements, srcStride);
  }

  template <typename DestDataT, typename SrcDataT>
  device_event async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                     decorated_local_ptr<SrcDataT> src,
                                     std::size_t numElements,
                                     std::size_t destStride) const
    requires(std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>)
  {
    return get_group().async_work_group_copy(dest, src, numElements,
                                             destStride);
  }

  template <typename... EventTN> void wait_for(EventTN... events) const {
    (events.wait(), ...);
  }

private:
  nd_item(std::barrier<> *barrier, id<Dimensions> globalID,
          id<Dimensions> localID, nd_range<Dimensions> range)
      : m_barrier{barrier}, m_global{globalID}, m_local{localID},
        m_range{range} {}

  template <int Dims>
  friend nd_item<Dims> detail::make_nd_item(std::barrier<> *, id<Dims>,
                                            id<Dims>, nd_range<Dims>);

  std::barrier<> *m_barrier;
  id<Dimensions> m_global;
  id<Dimensions> m_local;
  nd_range<Dimensions> m_range;
};

template <int Dimensions>
nd_item<Dimensions>
detail::make_nd_item(std::barrier<> *barrier, id<Dimensions> global,
                     id<Dimensions> local, nd_range<Dimensions> range) {
  return nd_item<Dimensions>{barrier, global, local, range};
}

} // namespace sycl
