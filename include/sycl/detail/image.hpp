#pragma once

#include "access.hpp"
#include "handler.hpp"
#include "property_list.hpp"
#include "range.hpp"
#include "vec.hpp"

namespace sycl {

enum class image_format {
  r8g8b8a8_unorm,
  r16g16b16a16_unorm,
  r8g8b8a8_sint,
  r16g16b16a16_sint,
  r32b32g32a32_sint,
  r8g8b8a8_uint,
  r16g16b16a16_uint,
  r32b32g32a32_uint,
  r16b16g16a16_sfloat,
  r32g32b32a32_sfloat,
  b8g8r8a8_unorm
};

enum class image_target { device, host_task };

enum class addressing_mode {
  mirrored_repeat,
  repeat,
  clamp_to_edge,
  clamp,
  none
};

enum class filtering_mode { nearest, linear };

enum class coordinate_normalization_mode { normalized, unnormalized };

struct image_sampler {
  addressing_mode addressing;
  coordinate_normalization_mode coordinate;
  filtering_mode filtering;
};

using image_allocator = std::allocator<std::byte>;

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class unsampled_image;

template <int Dimensions = 1, typename AllocatorT = sycl::image_allocator>
class sampled_image;

} // namespace sycl

namespace sycl::detail {

template <typename CoordT, typename ElemT, int Dimensions>
concept CoordTForDimensions =
    (Dimensions == 1 && std::is_same_v<CoordT, ElemT>) ||
    (Dimensions == 2 && std::is_same_v<CoordT, vec<ElemT, 2>>) ||
    (Dimensions == 3 && std::is_same_v<CoordT, vec<ElemT, 4>>);

inline constexpr std::size_t elem_size_from_image_format(image_format fmt) {
  switch (fmt) {
  case image_format::r8g8b8a8_unorm:
  case image_format::r8g8b8a8_sint:
  case image_format::r8g8b8a8_uint:
  case image_format::b8g8r8a8_unorm:
    return 4;
  case image_format::r16g16b16a16_unorm:
  case image_format::r16g16b16a16_sint:
  case image_format::r16g16b16a16_uint:
  case image_format::r16b16g16a16_sfloat:
    return 8;
  case image_format::r32b32g32a32_sint:
  case image_format::r32b32g32a32_uint:
  case image_format::r32g32b32a32_sfloat:
    return 16;
  default:
    return 0;
  }
}

template <int Dimensions, typename AllocatorT> struct unsampled_image_impl {
  std::vector<char> m_data;
  image_format m_format;
  range<Dimensions> m_range;
  std::conditional_t<(Dimensions > 1), range<Dimensions - 1>, std::monostate>
      m_pitch;
  AllocatorT m_allocator;
  void *m_hostPointer = nullptr;
  std::shared_ptr<void> m_sharedHostPointer;
  bool m_writeBack = false;
  property_list m_props;
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget>
struct unsampled_image_accessor_impl {
  std::vector<char> *m_data;
  handler *m_commandGroupHandler;
  range<Dimensions> m_range;
  property_list m_props;
};

template <typename DataT, int Dimensions, access_mode AccessMode>
struct host_unsampled_image_accessor_impl {
  std::vector<char> *m_data;
  range<Dimensions> m_range;
  property_list m_props;
};

template <int Dimensions, typename AllocatorT> struct sampled_image_impl {
  std::vector<char> m_data;
  image_format m_format;
  image_sampler m_sampler;
  range<Dimensions> m_range;
  std::conditional_t<(Dimensions > 1), range<Dimensions - 1>, std::monostate>
      m_pitch;
  AllocatorT m_allocator;
  const void *m_hostPointer = nullptr;
  std::shared_ptr<const void> m_sharedHostPointer;
  property_list m_props;
};

template <typename DataT, int Dimensions,
          image_target AccessTarget = image_target::device>
struct sampled_image_accessor_impl {
  std::vector<char> *m_data;
  handler *m_commandGroupHandler;
  property_list m_props;
};

template <typename DataT, int Dimensions>
struct host_sampled_image_accessor_impl {
  std::vector<char> *m_data;
  property_list m_props;
};

template <int Dimensions, typename AllocatorT>
std::shared_ptr<unsampled_image_impl<Dimensions, AllocatorT>>
get_image_impl(unsampled_image<Dimensions, AllocatorT> &);

template <int Dimensions, typename AllocatorT>
std::shared_ptr<sampled_image_impl<Dimensions, AllocatorT>>
get_image_impl(sampled_image<Dimensions, AllocatorT> &);

} // namespace sycl::detail

namespace sycl {

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget = image_target::device>
class unsampled_image_accessor;

template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write)>
class host_unsampled_image_accessor;

template <int Dimensions, typename AllocatorT> class unsampled_image {
public:
  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  const property_list &propList = {}) {
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  AllocatorT allocator, const property_list &propList = {}) {
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(image_format format, const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch, AllocatorT allocator,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const property_list &propList = {}) {
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef, AllocatorT allocator,
                  const property_list &propList = {}) {
    m_impl->m_hostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_hostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(void *hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch, AllocatorT allocator,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_hostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(std::shared_ptr<void> &hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const property_list &propList = {}) {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(std::shared_ptr<void> &hostPointer, image_format format,
                  const range<Dimensions> &rangeRef, AllocatorT allocator,
                  const property_list &propList = {}) {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(std::shared_ptr<void> &hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  unsampled_image(std::shared_ptr<void> &hostPointer, image_format format,
                  const range<Dimensions> &rangeRef,
                  const range<Dimensions - 1> &pitch, AllocatorT allocator,
                  const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_allocator = allocator;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  friend bool operator==(const unsampled_image &,
                         const unsampled_image &) = default;

  template <typename Property> bool has_property() const noexcept {
    return detail::has_property<Property>(m_impl->m_props);
  }

  template <typename Property> Property get_property() const {
    return detail::get_property<Property>(m_impl->m_props);
  }

  range<Dimensions> get_range() const { return m_impl->m_range; }

  range<Dimensions - 1> get_pitch() const
    requires(Dimensions > 1)
  {
    return m_impl->m_pitch;
  }

  std::size_t byte_size() const noexcept { return m_impl->m_data.size(); }

  std::size_t size() const noexcept { return m_impl->m_range.size(); }

  AllocatorT get_allocator() const { return m_impl->m_allocator; }

  template <typename DataT,
            access_mode Mode = (std::is_const_v<DataT>
                                    ? access_mode::read
                                    : access_mode::read_write),
            image_target Targ = image_target::device>
  unsampled_image_accessor<DataT, Dimensions, Mode, Targ>
  get_access(handler &commandGroupHandler, const property_list &propList = {});

  template <typename DataT, access_mode Mode = (std::is_const_v<DataT>
                                                    ? access_mode::read
                                                    : access_mode::read_write)>
  host_unsampled_image_accessor<DataT, Dimensions, Mode>
  get_host_access(const property_list &propList = {});

  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {}

  void set_write_back(bool flag = true) { m_impl->m_writeBack = flag; }

private:
  void allocate_image_data() {
    const std::size_t elem_size =
        detail::elem_size_from_image_format(m_impl->m_format);
    m_impl->m_data.resize(size() * elem_size);
  }

  friend std::hash<unsampled_image>;
  friend std::shared_ptr<detail::unsampled_image_impl<Dimensions, AllocatorT>>
  detail::get_image_impl<Dimensions, AllocatorT>(unsampled_image &);

  std::shared_ptr<detail::unsampled_image_impl<Dimensions, AllocatorT>> m_impl{
      std::make_shared<detail::unsampled_image_impl<Dimensions, AllocatorT>>()};
};

template <typename DataT, int Dimensions, access_mode AccessMode,
          image_target AccessTarget>
class unsampled_image_accessor {
public:
  using value_type =
      std::conditional_t<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  unsampled_image_accessor(unsampled_image<Dimensions, AllocatorT> &imageRef,
                           handler &commandGroupHandlerRef,
                           const property_list &propList = {}) {
    auto impl = detail::get_image_impl(imageRef);
    m_impl->m_data = &impl->m_data;
    m_impl->m_commandGroupHandler = &commandGroupHandlerRef;
    m_impl->m_range = imageRef.get_range();
    m_impl->m_props = propList;
  }

  friend bool operator==(const unsampled_image_accessor &,
                         const unsampled_image_accessor &) = default;

  /* -- property interface members -- */

  std::size_t size() const noexcept {
    return m_impl->m_data->size() / sizeof(DataT);
  }

  template <typename CoordT>
  DataT read(const CoordT &coords) const noexcept
    requires(AccessMode == access_mode::read &&
             detail::CoordTForDimensions<CoordT, int, Dimensions>)
  {
    return {};
  }

  template <typename CoordT>
  void write(const CoordT &coords, const DataT &color) const
    requires(AccessMode == access_mode::write &&
             detail::CoordTForDimensions<CoordT, int, Dimensions>)
  {}

private:
  friend std::hash<unsampled_image_accessor>;
  std::shared_ptr<detail::unsampled_image_accessor_impl<
      DataT, Dimensions, AccessMode, AccessTarget>>
      m_impl{std::make_shared<detail::unsampled_image_accessor_impl<
          DataT, Dimensions, AccessMode, AccessTarget>>()};
};

template <int Dimensions, typename AllocatorT>
template <typename DataT, access_mode Mode, image_target Targ>
unsampled_image_accessor<DataT, Dimensions, Mode, Targ>
unsampled_image<Dimensions, AllocatorT>::get_access(
    handler &commandGroupHandler, const property_list &propList) {
  return unsampled_image_accessor<DataT, Dimensions, Mode, Targ>(
      *this, commandGroupHandler, propList);
}

template <typename DataT, int Dimensions, access_mode AccessMode>
class host_unsampled_image_accessor {
public:
  using value_type =
      std::conditional<AccessMode == access_mode::read, const DataT, DataT>;
  using reference = value_type &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  host_unsampled_image_accessor(
      unsampled_image<Dimensions, AllocatorT> &imageRef,
      const property_list &propList = {}) {
    auto impl = detail::get_image_impl(imageRef);
    m_impl->m_data = &impl->m_data;
    m_impl->m_range = imageRef.get_range();
    m_impl->m_props = propList;
  }

  friend bool operator==(const host_unsampled_image_accessor &,
                         const host_unsampled_image_accessor &) = default;

  /* -- property interface members -- */

  std::size_t size() const noexcept {
    return m_impl->m_data->size() / sizeof(DataT);
  }

  template <typename CoordT>
  DataT read(const CoordT &coords) const noexcept
    requires(AccessMode == access_mode::read ||
             AccessMode == access_mode::read_write) &&
            detail::CoordTForDimensions<CoordT, int, Dimensions>
  {
    if constexpr (Dimensions == 1) {
      const std::size_t address = coords * sizeof(DataT);
      DataT result;
      std::memcpy(&result, &m_impl->m_data->data()[address], sizeof(DataT));
      return result;
    }
  }

  template <typename CoordT>
  void write(const CoordT &coords, const DataT &color) const
    requires(AccessMode == access_mode::write ||
             AccessMode == access_mode::read_write) &&
            detail::CoordTForDimensions<CoordT, int, Dimensions>
  {
    if constexpr (Dimensions == 1) {
      const std::size_t address = coords * sizeof(DataT);
      std::memcpy(&m_impl->m_data->data()[address], &color, sizeof(DataT));
    }
  }

private:
  friend std::hash<host_unsampled_image_accessor>;
  std::shared_ptr<
      detail::host_unsampled_image_accessor_impl<DataT, Dimensions, AccessMode>>
      m_impl{std::make_shared<detail::host_unsampled_image_accessor_impl<
          DataT, Dimensions, AccessMode>>()};
};

template <int Dimensions, typename AllocatorT>
template <typename DataT, access_mode Mode>
host_unsampled_image_accessor<DataT, Dimensions, Mode>
sycl::unsampled_image<Dimensions, AllocatorT>::get_host_access(
    const property_list &propList) {
  return host_unsampled_image_accessor<DataT, Dimensions, Mode>(*this,
                                                                propList);
}

template <typename DataT, int Dimensions,
          image_target AccessTarget = image_target::device>
class sampled_image_accessor;

template <typename DataT, int Dimensions> class host_sampled_image_accessor;

template <int Dimensions, typename AllocatorT> class sampled_image {
public:
  sampled_image(const void *hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const property_list &propList = {}) {
    m_impl->m_hostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_sampler = sampler;
    m_impl->m_range = rangeRef;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  sampled_image(const void *hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const range<Dimensions - 1> &pitch,
                const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_hostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_sampler = sampler;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  sampled_image(std::shared_ptr<const void> &hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const property_list &propList = {}) {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_sampler = sampler;
    m_impl->m_range = rangeRef;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  sampled_image(std::shared_ptr<const void> &hostPointer, image_format format,
                image_sampler sampler, const range<Dimensions> &rangeRef,
                const range<Dimensions - 1> &pitch,
                const property_list &propList = {})
    requires(Dimensions > 1)
  {
    m_impl->m_sharedHostPointer = hostPointer;
    m_impl->m_format = format;
    m_impl->m_sampler = sampler;
    m_impl->m_range = rangeRef;
    m_impl->m_pitch = pitch;
    m_impl->m_props = propList;
    allocate_image_data();
  }

  /* -- common interface members -- */

  friend bool operator==(const sampled_image &,
                         const sampled_image &) = default;

  /* -- property interface members -- */

  range<Dimensions> get_range() const { return m_impl->m_range; }

  range<Dimensions - 1> get_pitch() const
    requires(Dimensions > 1)
  {
    return m_impl->m_pitch;
  }

  std::size_t byte_size() const noexcept { return m_impl->m_data.size(); }

  std::size_t size() const noexcept { return get_range().size(); }

  template <typename DataT, image_target Targ = image_target::device>
  sampled_image_accessor<DataT, Dimensions, Targ>
  get_access(handler &commandGroupHandler, const property_list &propList = {});

  template <typename DataT>
  host_sampled_image_accessor<DataT, Dimensions>
  get_host_access(const property_list &propList = {});

private:
  void allocate_image_data() {
    const std::size_t elem_size =
        detail::elem_size_from_image_format(m_impl->m_format);
    m_impl->m_data.resize(size() * elem_size);
  }

  friend std::hash<sampled_image>;
  friend std::shared_ptr<detail::sampled_image_impl<Dimensions, AllocatorT>>
  detail::get_image_impl<Dimensions, AllocatorT>(sampled_image &);

  std::shared_ptr<detail::sampled_image_impl<Dimensions, AllocatorT>> m_impl{
      std::make_shared<detail::sampled_image_impl<Dimensions, AllocatorT>>()};
};

template <typename DataT, int Dimensions, image_target AccessTarget>
class sampled_image_accessor {
public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  sampled_image_accessor(sampled_image<Dimensions, AllocatorT> &imageRef,
                         handler &commandGroupHandlerRef,
                         const property_list &propList = {}) {
    auto impl = detail::get_image_impl(imageRef);
    m_impl->m_data = &impl->m_data;
    m_impl->m_commandGroupHandler = &commandGroupHandlerRef;
    m_impl->m_props = propList;
  }

  friend bool operator==(const sampled_image_accessor &,
                         const sampled_image_accessor &) = default;

  /* -- property interface members -- */

  std::size_t size() const noexcept {
    return m_impl->m_data->size() / sizeof(DataT);
  }

  template <typename CoordT>
  DataT read(const CoordT &coords) const noexcept
    requires(detail::CoordTForDimensions<CoordT, float, Dimensions>)
  {
    return {};
  }

private:
  friend std::hash<sampled_image_accessor>;
  std::shared_ptr<
      detail::sampled_image_accessor_impl<DataT, Dimensions, AccessTarget>>
      m_impl{std::make_shared<detail::sampled_image_accessor_impl<
          DataT, Dimensions, AccessTarget>>()};
};

template <int Dimensions, typename AllocatorT>
template <typename DataT, image_target Targ>
sampled_image_accessor<DataT, Dimensions, Targ>
sampled_image<Dimensions, AllocatorT>::get_access(
    handler &commandGroupHandler, const property_list &propList) {
  return sampled_image_accessor<DataT, Dimensions, Targ>(
      *this, commandGroupHandler, propList);
}

template <typename DataT, int Dimensions> class host_sampled_image_accessor {
public:
  using value_type = const DataT;
  using reference = const DataT &;
  using const_reference = const DataT &;

  template <typename AllocatorT>
  host_sampled_image_accessor(sampled_image<Dimensions, AllocatorT> &imageRef,
                              const property_list &propList = {}) {
    auto impl = detail::get_image_impl(imageRef);
    m_impl->m_data = &impl->m_data;
    m_impl->m_props = propList;
  }

  friend bool operator==(const host_sampled_image_accessor &,
                         const host_sampled_image_accessor &) = default;

  /* -- property interface members -- */

  std::size_t size() const noexcept {
    return m_impl->m_data->size() / sizeof(DataT);
  }

  template <typename CoordT>
  DataT read(const CoordT &coords) const noexcept
    requires(detail::CoordTForDimensions<CoordT, float, Dimensions>)
  {
    return {};
  }

private:
  friend std::hash<host_sampled_image_accessor>;
  std::shared_ptr<detail::host_sampled_image_accessor_impl<DataT, Dimensions>>
      m_impl{std::make_shared<
          detail::host_sampled_image_accessor_impl<DataT, Dimensions>>()};
};

template <int Dimensions, typename AllocatorT>
template <typename DataT>
host_sampled_image_accessor<DataT, Dimensions>
sampled_image<Dimensions, AllocatorT>::get_host_access(
    const property_list &propList) {
  return host_sampled_image_accessor<DataT, Dimensions>(*this, propList);
}

} // namespace sycl

template <int Dimensions, typename AllocatorT>
std::shared_ptr<sycl::detail::unsampled_image_impl<Dimensions, AllocatorT>>
sycl::detail::get_image_impl(unsampled_image<Dimensions, AllocatorT> &image) {
  return image.m_impl;
}

template <int Dimensions, typename AllocatorT>
std::shared_ptr<sycl::detail::sampled_image_impl<Dimensions, AllocatorT>>
sycl::detail::get_image_impl(sampled_image<Dimensions, AllocatorT> &image) {
  return image.m_impl;
}

template <int Dimensions, typename AllocatorT>
struct std::hash<sycl::unsampled_image<Dimensions, AllocatorT>> {
  std::size_t operator()(const sycl::unsampled_image<Dimensions, AllocatorT>
                             &image) const noexcept {
    return std::hash<decltype(image.m_impl)>{}(image.m_impl);
  }
};

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::image_target AccessTarget>
struct std::hash<sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                                AccessTarget>> {
  std::size_t
  operator()(const sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                                  AccessTarget> &accessor)
      const noexcept {
    return std::hash<decltype(accessor.m_impl)>{}(accessor.m_impl);
  }
};

template <typename DataT, int Dimensions>
struct std::hash<sycl::host_unsampled_image_accessor<DataT, Dimensions>> {
  std::size_t operator()(
      const sycl::host_unsampled_image_accessor<DataT, Dimensions> &accessor)
      const noexcept {
    return std::hash<decltype(accessor.m_impl)>{}(accessor.m_impl);
  }
};

template <int Dimensions, typename AllocatorT>
struct std::hash<sycl::sampled_image<Dimensions, AllocatorT>> {
  std::size_t operator()(
      const sycl::sampled_image<Dimensions, AllocatorT> &image) const noexcept {
    return std::hash<decltype(image.m_impl)>{}(image.m_impl);
  }
};

template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
struct std::hash<
    sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>> {
  std::size_t
  operator()(const sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>
                 &accessor) const noexcept {
    return std::hash<decltype(accessor.m_impl)>{}(accessor.m_impl);
  }
};

template <typename DataT, int Dimensions>
struct std::hash<sycl::host_sampled_image_accessor<DataT, Dimensions>> {
  std::size_t operator()(
      const sycl::host_sampled_image_accessor<DataT, Dimensions> &accessor)
      const noexcept {
    return std::hash<decltype(accessor.m_impl)>{}(accessor.m_impl);
  }
};
