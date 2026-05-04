#pragma once

#include <exception>
#include <system_error>

#include "context.hpp"
#include "errc.hpp"

namespace sycl {

namespace detail {

class error_category final : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }

  std::string message(int condition) const override {
    switch (static_cast<sycl::errc>(condition)) {
    case sycl::errc::success:
      return "success";
    case sycl::errc::runtime:
      return "runtime";
    case sycl::errc::kernel:
      return "kernel";
    case sycl::errc::accessor:
      return "accessor";
    case sycl::errc::nd_range:
      return "nd_range";
    case sycl::errc::event:
      return "event";
    case sycl::errc::kernel_argument:
      return "kernel argument";
    case sycl::errc::build:
      return "build";
    case sycl::errc::invalid:
      return "invalid";
    case sycl::errc::memory_allocation:
      return "memory allocation";
    case sycl::errc::platform:
      return "platform";
    case sycl::errc::profiling:
      return "profiling";
    case sycl::errc::feature_not_supported:
      return "feature not supported";
    case sycl::errc::kernel_not_supported:
      return "kernel not supported";
    case sycl::errc::backend_mismatch:
      return "backend mismatch";
    default:
      return "unknown";
    }
  }
};

static const error_category error_category_v{};

} // namespace detail

std::error_code make_error_code(errc e) noexcept;

const std::error_category &sycl_category() noexcept;

class exception : public virtual std::exception {
public:
  exception(std::error_code ec, const std::string &what_arg)
      : m_error{ec, what_arg} {}

  exception(std::error_code ec, const char *what_arg) : m_error{ec, what_arg} {}

  exception(std::error_code ec) : m_error{ec} {}

  exception(int ev, const std::error_category &ecat,
            const std::string &what_arg)
      : m_error{ev, ecat, what_arg} {}

  exception(int ev, const std::error_category &ecat, const char *what_arg)
      : m_error{ev, ecat, what_arg} {}

  exception(int ev, const std::error_category &ecat) : m_error{ev, ecat} {}

  exception(context ctx, std::error_code ec, const std::string &what_arg)
      : m_error{ec, what_arg}, m_context{ctx} {}

  exception(context ctx, std::error_code ec, const char *what_arg)
      : m_error{ec, what_arg}, m_context{ctx} {}

  exception(context ctx, std::error_code ec) : m_error{ec}, m_context{ctx} {}

  exception(context ctx, int ev, const std::error_category &ecat,
            const std::string &what_arg)
      : m_error{ev, ecat, what_arg}, m_context{ctx} {}

  exception(context ctx, int ev, const std::error_category &ecat,
            const char *what_arg)
      : m_error{ev, ecat, what_arg}, m_context{ctx} {}

  exception(context ctx, int ev, const std::error_category &ecat)
      : m_error{ev, ecat}, m_context{ctx} {}

  const std::error_code &code() const noexcept { return m_error.code(); }

  const std::error_category &category() const noexcept {
    return code().category();
  }

  const char *what() const noexcept { return m_error.what(); }

  bool has_context() const noexcept { return m_context.has_value(); }

  context get_context() const {
    if (!m_context.has_value())
      throw exception(make_error_code(errc::invalid),
                      "Exception has no associated context");
    return m_context.value();
  }

private:
  std::system_error m_error;
  std::optional<context> m_context;
};

namespace detail {
exception_list create_exception_list(std::vector<std::exception_ptr>);
}

class exception_list {
  // Used as a container for a list of asynchronous exceptions
public:
  using value_type = std::exception_ptr;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = std::vector<value_type>::const_iterator;
  using const_iterator = std::vector<value_type>::const_iterator;

  size_type size() const { return m_elements.size(); }
  iterator begin() const { return m_elements.begin(); }
  iterator end() const { return m_elements.end(); }

private:
  friend exception_list
      detail::create_exception_list(std::vector<std::exception_ptr>);

  std::vector<value_type> m_elements;
};

exception_list
detail::create_exception_list(std::vector<std::exception_ptr> exceptions);

} // namespace sycl

template <> struct std::is_error_code_enum<sycl::errc> : true_type {};

[[noreturn]] void sycl::detail::throw_sycl_exception(errc code,
                                                     const char *message);
