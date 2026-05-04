#pragma once

#include "property_list.hpp"
#include "queue.hpp"

namespace sycl {

namespace usm {
enum class alloc { host, device, shared, unknown };
}

namespace detail {

struct usm_allocation {
  void *ptr;
  std::size_t size;
  sycl::usm::alloc kind;
};

std::unordered_map<void *, usm_allocation> &get_usm_allocation_map();

void register_usm_allocation(void *ptr, std::size_t size,
                             sycl::usm::alloc kind);

void unregister_usm_allocation(void *ptr);

} // namespace detail

template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  usm_allocator() = delete;
  usm_allocator(const context &syclContext, const device &syclDevice,
                const property_list &propList = {})
      : m_context(syclContext), m_device(syclDevice), m_propList(propList) {}
  usm_allocator(const queue &syclQueue, const property_list &propList = {})
      : m_context(syclQueue.get_context()), m_device(syclQueue.get_device()),
        m_propList(propList) {}
  usm_allocator(const usm_allocator &other) = default;
  usm_allocator(usm_allocator &&) noexcept = default;
  usm_allocator &operator=(const usm_allocator &) = default;
  usm_allocator &operator=(usm_allocator &&) = default;

  template <class U>
  usm_allocator(usm_allocator<U, AllocKind, Alignment> const &other) noexcept
      : m_context(other.m_context), m_device(other.m_device),
        m_propList(other.m_propList) {}

  /// Allocate memory
  T *allocate(std::size_t count) {
    T *ptr = new T[count];
    detail::register_usm_allocation(ptr, count * sizeof(T), AllocKind);
    return ptr;
  }

  /// Deallocate memory
  void deallocate(T *Ptr, std::size_t count) {
    detail::unregister_usm_allocation(Ptr);
    delete[] Ptr;
  }

  /// Equality Comparison
  ///
  /// Allocators only compare equal if they are of the same USM kind, alignment,
  /// context, and device
  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    return (AllocKind == AllocKindU) && (Alignment == AlignmentU) &&
           (lhs.m_context == rhs.m_context) && (lhs.m_device == rhs.m_device);
  }

  /// Inequality Comparison
  /// Allocators only compare unequal if they are not of the same USM kind,
  /// alignment, context, or device
  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    return !(lhs == rhs);
  }

private:
  template <typename U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend class usm_allocator;

  context m_context;
  device m_device;
  property_list m_propList;
};

void *malloc_device(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext,
                    const property_list &propList = {});

template <typename T>
T *malloc_device(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  T *ptr = new T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::device);
  return ptr;
}

void *malloc_device(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList = {});

template <typename T>
T *malloc_device(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList = {});

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const device &syclDevice, const context &syclContext,
                        const property_list &propList = {}) {
  T *ptr = new (std::align_val_t(alignment)) T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::device);
  return ptr;
}

void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList = {});

template <typename T>
T *aligned_alloc_device(std::size_t alignment, std::size_t count,
                        const queue &syclQueue,
                        const property_list &propList = {}) {
  return aligned_alloc_device<T>(alignment, count, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}

void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList = {});

template <typename T>
T *malloc_host(std::size_t count, const context &syclContext,
               const property_list &propList = {}) {
  T *ptr = new T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::host);
  return ptr;
}

void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                  const property_list &propList = {});

template <typename T>
T *malloc_host(std::size_t count, const queue &syclQueue,
               const property_list &propList = {}) {
  return malloc_host<T>(count, syclQueue.get_context(), propList);
}

void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const context &syclContext,
                         const property_list &propList = {});

template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                      const context &syclContext,
                      const property_list &propList = {}) {
  T *ptr = new (std::align_val_t(alignment)) T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::host);
  return ptr;
}

void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const queue &syclQueue,
                         const property_list &propList = {});

template <typename T>
T *aligned_alloc_host(std::size_t alignment, std::size_t count,
                      const queue &syclQueue,
                      const property_list &propList = {}) {
  return aligned_alloc_host<T>(alignment, count, syclQueue.get_context(),
                               propList);
}

void *malloc_shared(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext,
                    const property_list &propList = {});

template <typename T>
T *malloc_shared(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  T *ptr = new T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::shared);
  return ptr;
}

void *malloc_shared(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList = {});

template <typename T>
T *malloc_shared(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_shared<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList = {});

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const device &syclDevice, const context &syclContext,
                        const property_list &propList = {}) {
  T *ptr = new (std::align_val_t(alignment)) T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), usm::alloc::shared);
  return ptr;
}

void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList = {});

template <typename T>
T *aligned_alloc_shared(std::size_t alignment, std::size_t count,
                        const queue &syclQueue,
                        const property_list &propList = {}) {
  return aligned_alloc_shared<T>(alignment, count, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}

void *malloc(std::size_t numBytes, const device &syclDevice,
             const context &syclContext, usm::alloc kind,
             const property_list &propList = {});

template <typename T>
T *malloc(std::size_t count, const device &syclDevice,
          const context &syclContext, usm::alloc kind,
          const property_list &propList = {}) {
  T *ptr = new T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), kind);
  return ptr;
}

void *malloc(std::size_t numBytes, const queue &syclQueue, usm::alloc kind,
             const property_list &propList = {});

template <typename T>
T *malloc(std::size_t count, const queue &syclQueue, usm::alloc kind,
          const property_list &propList = {}) {
  return malloc<T>(count, syclQueue.get_device(), syclQueue.get_context(), kind,
                   propList);
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const device &syclDevice, const context &syclContext,
                    usm::alloc kind, const property_list &propList = {});

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count,
                 const device &syclDevice, const context &syclContext,
                 usm::alloc kind, const property_list &propList = {}) {
  T *ptr = new (std::align_val_t(alignment)) T[count];
  detail::register_usm_allocation(ptr, count * sizeof(T), kind);
  return ptr;
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const queue &syclQueue, usm::alloc kind,
                    const property_list &propList = {});

template <typename T>
T *aligned_alloc(std::size_t alignment, std::size_t count,
                 const queue &syclQueue, usm::alloc kind,
                 const property_list &propList = {}) {
  return aligned_alloc<T>(alignment, count, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}

void free(void *ptr, const context &syclContext);

void free(void *ptr, const queue &syclQueue);

usm::alloc get_pointer_type(const void *ptr, const context &syclContext);

device get_pointer_device(const void *ptr, const context &syclContext);

} // namespace sycl
