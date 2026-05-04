#include "sycl/detail/usm.hpp"

namespace sycl {

namespace detail {

std::unordered_map<void *, usm_allocation> &get_usm_allocation_map() {
  static std::unordered_map<void *, usm_allocation> usm_allocation_map;
  return usm_allocation_map;
}

void register_usm_allocation(void *ptr, std::size_t size,
                             sycl::usm::alloc kind) {
  usm_allocation allocation{ptr, size, kind};
  get_usm_allocation_map()[ptr] = allocation;
}

void unregister_usm_allocation(void *ptr) {
  get_usm_allocation_map().erase(ptr);
}

} // namespace detail

void *malloc_device(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext, const property_list &propList) {
  void *ptr = std::malloc(numBytes);
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::device);
  return ptr;
}

void *malloc_device(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList) {
  return malloc_device(numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), propList);
}

void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList) {
  void *ptr = new (std::align_val_t(alignment)) char[numBytes];
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::device);
  return ptr;
}

void *aligned_alloc_device(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList) {
  return aligned_alloc_device(alignment, numBytes, syclQueue.get_device(),
                              syclQueue.get_context(), propList);
}

void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList) {
  void *ptr = std::malloc(numBytes);
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::host);
  return ptr;
}

void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                  const property_list &propList) {
  return malloc_host(numBytes, syclQueue.get_context(), propList);
}

void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const context &syclContext,
                         const property_list &propList) {
  void *ptr = new (std::align_val_t(alignment)) char[numBytes];
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::host);
  return ptr;
}

void *aligned_alloc_host(std::size_t alignment, std::size_t numBytes,
                         const queue &syclQueue,
                         const property_list &propList) {
  return aligned_alloc_host(alignment, numBytes, syclQueue.get_context(),
                            propList);
}

void *malloc_shared(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext, const property_list &propList) {
  void *ptr = std::malloc(numBytes);
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::shared);
  return ptr;
}

void *malloc_shared(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList) {
  return malloc_shared(numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), propList);
}

void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const device &syclDevice, const context &syclContext,
                           const property_list &propList) {
  void *ptr = new (std::align_val_t(alignment)) char[numBytes];
  detail::register_usm_allocation(ptr, numBytes, usm::alloc::shared);
  return ptr;
}

void *aligned_alloc_shared(std::size_t alignment, std::size_t numBytes,
                           const queue &syclQueue,
                           const property_list &propList) {
  return aligned_alloc_shared(alignment, numBytes, syclQueue.get_device(),
                              syclQueue.get_context(), propList);
}

void *malloc(std::size_t numBytes, const device &syclDevice,
             const context &syclContext, usm::alloc kind,
             const property_list &propList) {
  return std::malloc(numBytes);
}

void *malloc(std::size_t numBytes, const queue &syclQueue, usm::alloc kind,
             const property_list &propList) {
  return malloc(numBytes, syclQueue.get_device(), syclQueue.get_context(), kind,
                propList);
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const device &syclDevice, const context &syclContext,
                    usm::alloc kind, const property_list &propList) {
  void *ptr = new (std::align_val_t(alignment)) char[numBytes];
  detail::register_usm_allocation(ptr, numBytes, kind);
  return ptr;
}

void *aligned_alloc(std::size_t alignment, std::size_t numBytes,
                    const queue &syclQueue, usm::alloc kind,
                    const property_list &propList) {
  return aligned_alloc(alignment, numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), kind, propList);
}

void free(void *ptr, const context &syclContext) {
  detail::unregister_usm_allocation(ptr);
  std::free(ptr);
}

void free(void *ptr, const queue &syclQueue) {
  free(ptr, syclQueue.get_context());
}

usm::alloc get_pointer_type(const void *ptr, const context &syclContext) {
  auto ptr_it = detail::get_usm_allocation_map().find(const_cast<void *>(ptr));
  if (ptr_it != detail::get_usm_allocation_map().end())
    return ptr_it->second.kind;
  return usm::alloc::unknown;
}

device get_pointer_device(const void *ptr, const context &syclContext) {
  auto ptr_it = detail::get_usm_allocation_map().find(const_cast<void *>(ptr));
  if (ptr_it == detail::get_usm_allocation_map().end())
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "sycl::get_pointer_device called with non-USM pointer");
  return device{};
}
} // namespace sycl