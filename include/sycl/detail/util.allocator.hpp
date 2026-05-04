#pragma once

#include <memory>
#include <type_traits>

namespace sycl::detail {

template <typename T>
  requires(!std::is_const_v<T>)
class allocator {
public:
  template <typename AllocatorT>
  allocator(AllocatorT allocator)
      : m_impl{std::make_unique<impl<AllocatorT>>(allocator)} {}

  T *allocate(const std::size_t n) { return m_impl->allocate(n); }

  void deallocate(T *const p, std::size_t n) noexcept {
    m_impl->deallocate(p, n);
  }

private:
  class interface {
  public:
    virtual ~interface() = default;
    virtual T *allocate(const std::size_t) = 0;
    virtual void deallocate(T *const p, std::size_t) noexcept = 0;
  };

  template <typename AllocatorT> class impl final : public interface {
  public:
    impl(AllocatorT allocator) : m_allocator{allocator} {}

    T *allocate(const std::size_t n) override {
      return m_allocator.allocate(n);
    }

    void deallocate(T *const p, std::size_t n) noexcept override {
      m_allocator.deallocate(p, n);
    }

  private:
    AllocatorT m_allocator;
  };

  std::unique_ptr<interface> m_impl;
};

} // namespace sycl::detail
